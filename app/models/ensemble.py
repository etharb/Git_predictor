# -*- coding: utf-8 -*-
"""
EnsemblePredictor (kompat + rozšířený)
--------------------------------------
• L1:   xgb / lstm / hrm  (+ p_meta z hrm_meta.pt)
• L2:   hrm_meta_L2.pt (+ vol. Platt kalibrace meta_L2.calib.pkl)
• L3:   supervisor_L3.pt nebo bag (supervisor_bag.json) (+ vol. Platt)
      + volitelné multi-hlavy: abstain / uncert / softmax3 / hsel

Kompatibilita:
• Respektuje supervisor.json: sets_l1/sets_l2, use_raw_in_l3/use_l1_in_l3/use_l2_in_l3, seq_len.
• Načítá thresholds/regime parametry z meta.json/supervisor checkpointů.
• Když něco chybí, vrátí co má (bez pádu).

Výstup predict_detail(df) obsahuje:
  p_xgb, p_lstm, p_hrm, p_meta, p_l2, p_l3,
  p_abstain, p_uncert, p3, hsel, p_ens
"""

import os, json, math
import numpy as np
import joblib
import torch
import torch.nn as nn

from typing import Dict, List, Optional

from .hrm_model import HRMHead
from .lstm_model import SmallLSTM
from ..utils.features import FEATURE_COLS, make_lstm_sequences

# ----- MLP kompat pro starý meta_gate.json -----
class MetaMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128, drop: float = 0.10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(drop),
            nn.Linear(hidden, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x).view(-1)

    @torch.no_grad()
    def infer1(self, x1d_np: np.ndarray) -> float:
        t = torch.from_numpy(x1d_np.astype(np.float32).reshape(1, -1))
        return float(self.forward(t).cpu().numpy().reshape(-1)[0])

# ----------------- utilky -----------------
def _softmax(z):
    z = np.array(z, dtype=np.float32)
    z = z - np.max(z)
    e = np.exp(z)
    s = e.sum()
    return (e / s) if s > 0 else np.ones_like(e) / len(e)

def _safe_make_lstm_sequences(df_feat_recent, seq_len: int):
    out = make_lstm_sequences(df_feat_recent, seq_len=seq_len)
    if isinstance(out, (list, tuple)):
        return out[0]
    return out

def _apply_platt_or_isotonic(cal, p: float) -> float:
    if cal is None or p is None:
        return p
    if hasattr(cal, "transform"):  # isotonic
        return float(cal.transform(np.array([p]))[0])
    if hasattr(cal, "predict_proba"):  # Platt (LogReg) na logitech
        eps = 1e-6
        pp = np.clip(p, eps, 1 - eps)
        logit = np.log(pp / (1 - pp)).reshape(-1, 1)
        return float(cal.predict_proba(logit)[:, 1][0])
    return p

def _sigmoid_to_logit(p: float) -> float:
    eps = 1e-6
    pp = np.clip(p, eps, 1 - eps)
    return float(np.log(pp/(1-pp)))

def _compute_regime_vector(df_feat, t_idx: int) -> np.ndarray:
    """
    Režimové featury:
      • sin/cos minuty a hodiny (pokud index je epoch sekundy)
      • 60s realized vol z 'close' (pokud existuje)
      • 60s SMA volume (pokud existuje)
    Bezpečné – pokud sloupce nejsou, dané prvky se jen nepřidají.
    """
    cols = []

    # časová sin/cos
    if df_feat.index.dtype.kind in "iu":  # integer epoch seconds
        ts = int(df_feat.index[t_idx])
        minute = (ts // 60) % 60
        hour = (ts // 3600) % 24
        ang_m = 2 * math.pi * (minute / 60.0)
        ang_h = 2 * math.pi * (hour / 24.0)
        cols.extend([math.sin(ang_m), math.cos(ang_m), math.sin(ang_h), math.cos(ang_h)])

    # 60s vol z close
    if "close" in df_feat.columns:
        start = max(0, t_idx - 59)
        c = df_feat["close"].values[start:t_idx+1].astype(np.float32)
        if len(c) >= 2:
            r = np.diff(c) / (c[:-1] + 1e-12)
            cols.append(float(np.std(r)))
        else:
            cols.append(0.0)

    # 60s SMA volume
    if "volume" in df_feat.columns:
        start = max(0, t_idx - 59)
        v = df_feat["volume"].values[start:t_idx+1].astype(np.float32)
        cols.append(float(np.mean(v)))

    return np.array(cols, dtype=np.float32)

# --- KOMPATIBILNÍ nahrání HRM checkpointů (legacy vs. multi-head) ---
def _load_hrm_state_dict_compat(model: nn.Module, state_dict: dict):
    """
    Umožní:
      - staré checkpointy: klíče "head.*"
      - nové checkpointy:  klíče "head_updown.*" (+ volitelné hlavy head_abstain.*, head_uncert.*, head_softmax3.*, head_hsel.*)
    Přemapuje "head." -> "head_updown." a použije strict=False, takže chybějící volitelné hlavy nevadí.
    """
    raw = state_dict or {}
    has_abstain  = any(k.startswith("head_abstain")  for k in raw.keys())
    has_uncert   = any(k.startswith("head_uncert")   for k in raw.keys())
    has_softmax3 = any(k.startswith("head_softmax3") for k in raw.keys())
    has_hsel     = any(k.startswith("head_hsel")     for k in raw.keys())

    # přemapuj legacy "head." -> "head_updown."
    sd = {}
    for k, v in raw.items():
        if k.startswith("head."):
            sd["head_updown" + k[len("head"):]] = v
        else:
            sd[k] = v

    try:
        model.load_state_dict(sd, strict=True)
    except Exception:
        model.load_state_dict(sd, strict=False)

    setattr(model, "_has_abstain",  bool(has_abstain))
    setattr(model, "_has_uncert",   bool(has_uncert))
    setattr(model, "_has_softmax3", bool(has_softmax3))
    setattr(model, "_has_hsel",     bool(has_hsel))


# ===========================================================
#                    HLAVNÍ TŘÍDA ENSEMBLU
# ===========================================================
class EnsemblePredictor:
    """
    L1/L2/L3 ansámbl s prioritami: L3 > L2 > (MLP gate) > L1 meta > průměr (HRM/LSTM/XGB).

    L3 supervisor (HRMHead) umí míchat:
      - L1 blok: p_L1(t..t-K+1) napříč sadami (povolitelný)
      - L2 blok: p_L2(t..t-K+1) z více L2-hlav (různé 'target' sady) (povolitelný)
      - RAW blok: sekvence p_xgb/p_lstm/p_hrm (povolitelný)
      - volitelně režimové featury (čas/volatilita/objem) – stejné jako v tréninku
    """
    def __init__(self, weights_dir: str | None):
        self.dir = os.path.abspath(weights_dir) if weights_dir else None

        # base (aktuální sada)
        self.xgb = None;  self.xgb_cal = None
        self.lstm = None; self.lstm_cal = None
        self.hrm = None
        self.hrm_meta = None

        # L2 gate (aktivní sada)
        self.l2_gate = None
        self.l2_seq_len = 60
        self.l1_seq_len = 60
        self.l2_sets: List[str] = []
        self.l2_use_raw = False
        self.l2_cal = None

        # L3 supervisor
        self.l3_supervisor = None          # single model
        self.l3_bag: Optional[dict] = None # {"models":[...], "calib":obj} – volitelné
        self.l3_seq_len = 60
        self.l3_use_raw = False
        self.l3_use_l1 = True
        self.l3_use_l2 = True
        self.l3_use_regime = False
        self.l3_regime_mean = None
        self.l3_regime_std = None
        self.l3_sets: List[str] = []       # pro L1/RAW blok
        self.l3_l2_heads: List[str] = []   # sady, jejichž L2 výstupy budeme míchat v L3
        self.l3_cal = None

        # cache
        self._l2_sets_cache = {}   # aktivní L2 config
        self._l3_sets_cache = {}   # L3 L1/RAW blok (stejné jako L2)
        self._l2_envs = {}         # sdir -> {l2_model, ...}

        # MLP meta-gate kompat
        self.meta_gate = None
        self.meta_gate_desc = None
        self.gate_components = []
        self.loaded_components = {}
        self.gate_mode = "auto"
        self.gate_alpha = 0.3
        self.gate_gains = {}
        self._last_comp_probs = {}

        self.horizon_sec = None
        self.seq_len = 60

        self._load_all()

    # ---------- utils ----------
    def _load_model_from_set(self, set_dir, mtype):
        out = {"model": None, "calib": None, "seq_len": None, "in_features": len(FEATURE_COLS)}
        if mtype == "xgb":
            xm = os.path.join(set_dir, "xgb.model")
            if os.path.exists(xm):
                out["model"] = joblib.load(xm)
            xc = os.path.join(set_dir, "xgb.calib.pkl")
            if os.path.exists(xc):
                out["calib"] = joblib.load(xc)

        elif mtype == "lstm":
            lm = os.path.join(set_dir, "lstm.pt")
            if os.path.exists(lm):
                ckpt = torch.load(lm, map_location="cpu")
                out["seq_len"] = int(ckpt.get("seq_len") or 60)
                out["in_features"] = int(ckpt.get("in_features") or len(FEATURE_COLS))
                mdl = SmallLSTM(in_features=out["in_features"], hidden=48, num_layers=1)
                mdl.load_state_dict(ckpt['state_dict'])
                mdl.eval()
                out["model"] = mdl
            lc = os.path.join(set_dir, "lstm.calib.pkl")
            if os.path.exists(lc):
                out["calib"] = joblib.load(lc)

        elif mtype == "hrm":
            hm = os.path.join(set_dir, "hrm.pt")
            if os.path.exists(hm):
                ckpt = torch.load(hm, map_location="cpu")
                out["seq_len"] = int(ckpt.get("seq_len") or 60)
                out["in_features"] = int(ckpt.get("in_features") or len(FEATURE_COLS))
                mdl = HRMHead(in_features=int(ckpt['in_features']),
                              hidden_low=int(ckpt['hidden_low']),
                              hidden_high=int(ckpt['hidden_high']),
                              high_period=int(ckpt['high_period']))
                _load_hrm_state_dict_compat(mdl, ckpt.get('state_dict') or {})
                mdl.eval()
                out["model"] = mdl

        elif mtype == "hrm_meta":  # L1 v dané sadě
            hmm = os.path.join(set_dir, "hrm_meta.pt")
            if os.path.exists(hmm):
                ckpt = torch.load(hmm, map_location="cpu")
                mdl = HRMHead(in_features=int(ckpt['in_features']),
                              hidden_low=int(ckpt['hidden_low']),
                              hidden_high=int(ckpt['high_period']),
                              high_period=int(ckpt['high_period']))
                _load_hrm_state_dict_compat(mdl, ckpt.get('state_dict') or {})
                mdl.eval()
                out["model"] = mdl
                out["in_features"] = int(ckpt['in_features'])
                out["seq_len"] = int(ckpt.get("seq_len") or 60)

        elif mtype in ("l2", "hrm_meta_L2"):  # L2 head v sadě
            p2 = os.path.join(set_dir, "hrm_meta_L2.pt")
            if os.path.exists(p2):
                ckpt = torch.load(p2, map_location="cpu")
                mdl = HRMHead(in_features=int(ckpt['in_features']),
                              hidden_low=int(ckpt['hidden_low']),
                              hidden_high=int(ckpt['hidden_high']),
                              high_period=int(ckpt['high_period']))
                _load_hrm_state_dict_compat(mdl, ckpt.get('state_dict') or {})
                mdl.eval()
                out["model"] = mdl
                out["in_features"] = int(ckpt['in_features'])
                out["seq_len"] = int(ckpt.get("seq_len") or 60)
            calp = os.path.join(set_dir, "meta_L2.calib.pkl")
            if os.path.exists(calp):
                try:
                    out["calib"] = joblib.load(calp)
                except Exception:
                    out["calib"] = None
        return out

    def _load_meta_gate_mlp(self):
        j = os.path.join(self.dir, "meta_gate.json")
        p = os.path.join(self.dir, "meta_gate.pt")
        if not (os.path.exists(j) and os.path.exists(p)):
            return
        try:
            with open(j, "r") as f:
                desc = json.load(f)
        except Exception:
            return
        ckpt = torch.load(p, map_location="cpu")
        mdl = MetaMLP(in_dim=int(ckpt.get("in_dim") or 8), hidden=128, drop=0.0)
        mdl.load_state_dict(ckpt["state_dict"])
        mdl.eval()

        self.meta_gate = mdl
        self.meta_gate_desc = desc
        self.gate_components = list(desc.get("components") or [])
        self.gate_regime_cols = list(ckpt.get("regime_cols") or [])
        self.gate_regime_mean = np.array(ckpt.get("regime_mean") or [], dtype=np.float32)
        self.gate_regime_std  = np.array(ckpt.get("regime_std") or [], dtype=np.float32)

        self.loaded_components = {}
        for comp in self.gate_components:
            key = comp["key"]; ctype = comp["type"]; cpath = comp["path"]
            info = self._load_model_from_set(cpath, ctype)
            if info["model"] is not None:
                self.loaded_components[key] = {**info, "type": ctype, "path": cpath}
        for comp in self.gate_components:
            self.gate_gains.setdefault(comp["key"], 1.0)

    def _load_l2_gate(self):
        meta_path = os.path.join(self.dir, "meta.json")
        l2_path = os.path.join(self.dir, "hrm_meta_L2.pt")
        if not (os.path.exists(meta_path) and os.path.exists(l2_path)):
            return
        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
        except Exception:
            return
        cfg = (meta or {}).get("meta_gate") or {}
        sets = list(cfg.get("sets") or [])
        if not sets:
            return

        ckpt = torch.load(l2_path, map_location="cpu")
        mdl = HRMHead(in_features=int(ckpt['in_features']),
                      hidden_low=int(ckpt['hidden_low']),
                      hidden_high=int(ckpt['hidden_high']),
                      high_period=int(ckpt['high_period']))
        _load_hrm_state_dict_compat(mdl, ckpt.get('state_dict') or {})
        mdl.eval()
        self.l2_gate = mdl

        self.l2_seq_len = int(ckpt.get("seq_len") or int(cfg.get("l2_seq_len") or 60))
        self.l1_seq_len = int(cfg.get("l1_seq_len") or 60)
        self.l2_sets = [os.path.abspath(s) for s in sets]
        self.l2_use_raw = bool(cfg.get("use_raw_in_l2", False))

        calp = os.path.join(self.dir, "meta_L2.calib.pkl")
        if os.path.exists(calp):
            try:
                self.l2_cal = joblib.load(calp)
            except Exception:
                self.l2_cal = None

        self._l2_sets_cache = {}
        for sdir in self.l2_sets:
            info = {
                "xgb":  self._load_model_from_set(sdir, "xgb"),
                "lstm": self._load_model_from_set(sdir, "lstm"),
                "hrm":  self._load_model_from_set(sdir, "hrm"),
                "l1":   self._load_model_from_set(sdir, "hrm_meta"),
            }
            self._l2_sets_cache[sdir] = info

    def _prepare_L2_env_for_head(self, head_dir: str):
        """Připrav prostředí pro 'cizí' L2 hlavu (jiný target set/horizont) – pro L3 vstup."""
        head_dir = os.path.abspath(head_dir)
        if head_dir in self._l2_envs:
            return
        meta_path = os.path.join(head_dir, "meta.json")
        l2_path = os.path.join(head_dir, "hrm_meta_L2.pt")
        if not (os.path.exists(meta_path) and os.path.exists(l2_path)):
            return
        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
        except Exception:
            return
        cfg = (meta or {}).get("meta_gate") or {}
        sets = list(cfg.get("sets") or [])
        if not sets:
            return

        l2_info = self._load_model_from_set(head_dir, "l2")
        if l2_info["model"] is None:
            return
        env = {
            "l2_model": l2_info["model"],
            "l2_cal":   l2_info.get("calib"),
            "l2_seq_len": int(l2_info.get("seq_len") or int(cfg.get("l2_seq_len") or 60)),
            "l1_seq_len": int(cfg.get("l1_seq_len") or 60),
            "use_raw": bool(cfg.get("use_raw_in_l2", False)),
            "sets": [os.path.abspath(s) for s in sets],
            "sets_cache": {},
        }
        for sdir in env["sets"]:
            env["sets_cache"][sdir] = {
                "xgb":  self._load_model_from_set(sdir, "xgb"),
                "lstm": self._load_model_from_set(sdir, "lstm"),
                "hrm":  self._load_model_from_set(sdir, "hrm"),
                "l1":   self._load_model_from_set(sdir, "hrm_meta"),
            }
        self._l2_envs[head_dir] = env

    def _load_l3_supervisor(self):
        sup_json = os.path.join(self.dir, "supervisor.json")
        sup_ckpt = os.path.join(self.dir, "supervisor_L3.pt")
        bag_json = os.path.join(self.dir, "supervisor_bag.json")

        # supervisor.json (vstupní konfigurace)
        supcfg = {}
        if os.path.exists(sup_json):
            try:
                with open(sup_json, "r") as f:
                    supcfg = json.load(f) or {}
            except Exception:
                supcfg = {}

        # L3 single
        if os.path.exists(sup_ckpt):
            ckpt = torch.load(sup_ckpt, map_location="cpu")
            mdl = HRMHead(in_features=int(ckpt['in_features']),
                          hidden_low=int(ckpt['hidden_low']),
                          hidden_high=int(ckpt['hidden_high']),
                          high_period=int(ckpt['high_period']),
                          use_abstain_head=bool(ckpt.get("use_abstain_head", True)),
                          use_softmax3=bool(ckpt.get("use_softmax3", False)),
                          use_uncert_head=bool(ckpt.get("use_uncert_head", False)),
                          num_hsel=int(ckpt.get("num_hsel", 0)))
            _load_hrm_state_dict_compat(mdl, ckpt.get('state_dict') or {})
            mdl.eval()
            self.l3_supervisor = mdl

            self.l3_seq_len   = int(ckpt.get("seq_len") or int(supcfg.get("seq_len") or 60))
            # preferuj supervisor.json flagy, fallback na ckpt
            self.l3_use_raw   = bool(supcfg.get("use_raw_in_l3", supcfg.get("use_raw_in_l2", False)))
            self.l3_use_l1    = bool(supcfg.get("use_l1_in_l3", True))
            self.l3_use_l2    = bool(supcfg.get("use_l2_in_l3", True))
            self.l3_use_regime = bool(ckpt.get("use_regime_in_l3", supcfg.get("use_regime_in_l3", False)))
            rm = ckpt.get("regime_mean"); rs = ckpt.get("regime_std")
            self.l3_regime_mean = np.array(rm, dtype=np.float32) if rm is not None else None
            self.l3_regime_std  = np.array(rs, dtype=np.float32) if rs is not None else None

            sets_l1 = supcfg.get("sets_l1")
            if sets_l1 and isinstance(sets_l1, list) and len(sets_l1) > 0:
                self.l3_sets = [os.path.abspath(s) for s in sets_l1]
            else:
                self.l3_sets = list(self.l2_sets)

            sets_l2 = supcfg.get("sets_l2") or []
            self.l3_l2_heads = [os.path.abspath(s) for s in sets_l2 if isinstance(s, str)]
            for sdir in self.l3_l2_heads:
                self._prepare_L2_env_for_head(sdir)

            calp = os.path.join(self.dir, "supervisor_L3.calib.pkl")
            if os.path.exists(calp):
                try:
                    self.l3_cal = joblib.load(calp)
                except Exception:
                    self.l3_cal = None

            self._l3_sets_cache = {}
            for sdir in self.l3_sets:
                self._l3_sets_cache[sdir] = {
                    "xgb":  self._load_model_from_set(sdir, "xgb"),
                    "lstm": self._load_model_from_set(sdir, "lstm"),
                    "hrm":  self._load_model_from_set(sdir, "hrm"),
                    "l1":   self._load_model_from_set(sdir, "hrm_meta"),
                }

        # L3 bag (volitelně)
        if os.path.exists(bag_json):
            try:
                cfg = json.load(open(bag_json, "r"))
                ckpts = cfg.get("checkpoints") or []
                models = []
                for p in ckpts:
                    if not os.path.exists(p):
                        continue
                    try:
                        ck = torch.load(p, map_location="cpu")
                        mdl = HRMHead(
                            in_features=int(ck['in_features']),
                            hidden_low=int(ck['hidden_low']),
                            hidden_high=int(ck['hidden_high']),
                            high_period=int(ck['high_period']),
                            use_abstain_head=bool(ck.get("use_abstain_head", True)),
                            use_softmax3=bool(ck.get("use_softmax3", False)),
                            use_uncert_head=bool(ck.get("use_uncert_head", False)),
                            num_hsel=int(ck.get("num_hsel", 0))
                        )
                        _load_hrm_state_dict_compat(mdl, ck.get('state_dict') or {})
                        mdl.eval()
                        models.append({"model": mdl})
                    except Exception:
                        pass
                if models:
                    cal = None
                    calp = os.path.join(self.dir, "supervisor_L3.calib.pkl")
                    if os.path.exists(calp):
                        try:
                            cal = joblib.load(calp)
                        except Exception:
                            cal = None
                    self.l3_bag = {"models": models, "calib": cal}
            except Exception:
                pass

    def _load_all(self):
        if not self.dir or not os.path.isdir(self.dir):
            return

        meta_path = os.path.join(self.dir, "meta.json")
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r") as f:
                    m = json.load(f)
                self.horizon_sec = int(m.get("horizon_sec") or 1)
                self.seq_len = int(m.get("seq_len") or 60)
            except Exception:
                pass

        # base
        xm = os.path.join(self.dir, "xgb.model")
        if os.path.exists(xm):
            self.xgb = joblib.load(xm)
        xc = os.path.join(self.dir, "xgb.calib.pkl")
        if os.path.exists(xc):
            self.xgb_cal = joblib.load(xc)

        lm = os.path.join(self.dir, "lstm.pt")
        if os.path.exists(lm):
            ckpt = torch.load(lm, map_location="cpu")
            self.seq_len = int(ckpt.get("seq_len") or self.seq_len)
            self.lstm = SmallLSTM(in_features=int(ckpt['in_features']), hidden=48, num_layers=1)
            self.lstm.load_state_dict(ckpt['state_dict'])
            self.lstm.eval()
        lc = os.path.join(self.dir, "lstm.calib.pkl")
        if os.path.exists(lc):
            self.lstm_cal = joblib.load(lc)

        hm = os.path.join(self.dir, "hrm.pt")
        if os.path.exists(hm):
            ckpt = torch.load(hm, map_location="cpu")
            self.hrm = HRMHead(in_features=int(ckpt['in_features']),
                               hidden_low=int(ckpt['hidden_low']),
                               hidden_high=int(ckpt['hidden_high']),
                               high_period=int(ckpt['high_period']))
            _load_hrm_state_dict_compat(self.hrm, ckpt.get('state_dict') or {})
            self.hrm.eval()

        hmm = os.path.join(self.dir, "hrm_meta.pt")
        if os.path.exists(hmm):
            ckpt = torch.load(hmm, map_location="cpu")
            self.hrm_meta = HRMHead(in_features=int(ckpt['in_features']),
                                    hidden_low=int(ckpt['hidden_low']),
                                    hidden_high=int(ckpt['hidden_high']),
                                    high_period=int(ckpt['high_period']))
            _load_hrm_state_dict_compat(self.hrm_meta, ckpt.get('state_dict') or {})
            self.hrm_meta.eval()

        self._load_l2_gate()
        self._load_l3_supervisor()
        self._load_meta_gate_mlp()

    # ---------- gate tuning (MLP) ----------
    def gate_get_state(self):
        return {
            "mode": self.gate_mode,
            "alpha": float(self.gate_alpha),
            "components": [c["key"] for c in self.gate_components],
            "gains": {k: float(self.gate_gains.get(k, 1.0)) for k in [c["key"] for c in self.gate_components]},
            "last_probs": {k: (float(v) if v is not None else None) for k, v in self._last_comp_probs.items()},
            "l2_ready": bool(self.l2_gate is not None),
            "l2_sets": list(self.l2_sets),
            "l2_lens": {"l1_seq_len": int(self.l1_seq_len), "l2_seq_len": int(self.l2_seq_len)},
            "l2_use_raw": bool(self.l2_use_raw),
            "l3_ready": bool(self.l3_supervisor is not None or self.l3_bag is not None),
            "l3_sets": list(self.l3_sets),
            "l3_l2_heads": list(self.l3_l2_heads),
            "l3_lens": {"l3_seq_len": int(self.l3_seq_len)},
            "l3_use_raw": bool(self.l3_use_raw),
            "l3_use_l1": bool(self.l3_use_l1),
            "l3_use_l2": bool(self.l3_use_l2),
            "l3_use_regime": bool(self.l3_use_regime),
        }

    def gate_set_state(self, mode=None, alpha=None, gains=None, reset=False):
        if reset:
            self.gate_mode = "auto"
            self.gate_alpha = 0.3
            self.gate_gains = {c["key"]: 1.0 for c in self.gate_components}
            return
        if mode in ("auto", "manual", "blend"):
            self.gate_mode = mode
        if isinstance(alpha, (int, float)):
            self.gate_alpha = max(0.0, min(1.0, float(alpha)))
        if isinstance(gains, dict):
            for k, v in gains.items():
                try:
                    self.gate_gains[k] = float(v)
                except Exception:
                    pass

    # ---------- MLP gate vstup ----------
    def _build_gate_vector_mlp(self, df_feat_recent):
        """
        Vrací: (x_vec_np, probs_map, ok)
          - x_vec_np: 1D vektor [p_comp1, p_comp2, ..., regime_zscores...]
          - probs_map: dict {key -> p_comp}
          - ok: bool, zda máme dost komponent pro inference
        """
        if self.meta_gate is None or self.meta_gate_desc is None:
            return None, {}, False
        if df_feat_recent is None or len(df_feat_recent) == 0:
            return None, {}, False

        keys = [c["key"] for c in self.gate_components]
        probs_map = {}
        have = 0
        for c in self.gate_components:
            k = c["key"]
            info = self.loaded_components.get(k)
            if not info:
                probs_map[k] = None
                continue
            try:
                p = self._predict_component_once(k, info, df_feat_recent)
            except Exception:
                p = None
            probs_map[k] = p
            if p is not None:
                have += 1

        # režimové featury (z-score)
        reg_vals = []
        if getattr(self, "gate_regime_cols", None) and len(self.gate_regime_cols) > 0:
            last = df_feat_recent.tail(1)
            for col in self.gate_regime_cols:
                if col in last.columns:
                    v = float(last[col].values[0])
                else:
                    v = 0.0
                reg_vals.append(v)
            reg_vals = np.array(reg_vals, dtype=np.float32)
            mu = self.gate_regime_mean.astype(np.float32) if getattr(self, "gate_regime_mean", None) is not None else np.zeros_like(reg_vals)
            sd = self.gate_regime_std.astype(np.float32) if getattr(self, "gate_regime_std", None) is not None else np.ones_like(reg_vals)
            sd = np.where(sd <= 1e-9, 1.0, sd)
            reg_z = (reg_vals - mu) / sd
        else:
            reg_z = np.array([], dtype=np.float32)

        comp_p = []
        for k in keys:
            p = probs_map.get(k)
            comp_p.append(0.5 if (p is None or (isinstance(p, float) and not np.isfinite(p))) else float(p))
        x_vec = np.concatenate([np.array(comp_p, dtype=np.float32), reg_z.astype(np.float32)], axis=0)

        ok = (have >= 2)
        return x_vec, probs_map, ok

    # ---------- per-model rolls ----------
    def _roll_xgb(self, mdl, cal, df_feat_recent, K: int):
        if mdl is None or len(df_feat_recent) < 1 or K <= 0:
            return None
        X = df_feat_recent[FEATURE_COLS].values.astype(np.float32)
        if len(X) < K:
            return None
        p = mdl.predict_proba(X[-K:])[:, 1].astype(np.float32)
        if cal is not None:
            p = np.array([_apply_platt_or_isotonic(cal, float(pi)) for pi in p], dtype=np.float32)
        return p

    def _roll_lstm(self, mdl, cal, df_feat_recent, q: int, K: int):
        if mdl is None or len(df_feat_recent) < q or K <= 0:
            return None
        Xseq = _safe_make_lstm_sequences(df_feat_recent, seq_len=q)
        if Xseq is None or len(Xseq) < K:
            return None
        with torch.no_grad():
            p = mdl(torch.from_numpy(Xseq[-K:])).cpu().numpy().reshape(-1).astype(np.float32)
        if cal is not None:
            p = np.array([_apply_platt_or_isotonic(cal, float(pi)) for pi in p], dtype=np.float32)
        return p

    def _roll_hrm(self, mdl, df_feat_recent, q: int, K: int):
        if mdl is None or len(df_feat_recent) < q or K <= 0:
            return None
        Xseq = _safe_make_lstm_sequences(df_feat_recent, seq_len=q)
        if Xseq is None or len(Xseq) < K:
            return None
        with torch.no_grad():
            p = mdl.infer(torch.from_numpy(Xseq[-K:])).cpu().numpy().reshape(-1).astype(np.float32)
        return p

    def _predict_component_once(self, key, comp_info, df_feat_recent):
        if comp_info["model"] is None:
            return None
        typ = comp_info["type"]
        if typ == "xgb":
            if len(df_feat_recent) < 1:
                return None
            X = df_feat_recent[FEATURE_COLS].values[-1:].astype(np.float32)
            try:
                p = float(comp_info["model"].predict_proba(X)[:, 1][0])
            except Exception:
                return None
            return _apply_platt_or_isotonic(comp_info.get("calib"), p)

        elif typ == "lstm":
            q = int(comp_info.get("seq_len") or 60)
            if len(df_feat_recent) < q:
                return None
            Xseq = _safe_make_lstm_sequences(df_feat_recent, seq_len=q)
            if Xseq is None or len(Xseq) == 0:
                return None
            with torch.no_grad():
                p = float(comp_info["model"](torch.from_numpy(Xseq[-1:])).cpu().numpy().reshape(-1)[0])
            return _apply_platt_or_isotonic(comp_info.get("calib"), p)

        elif typ == "hrm":
            q = int(comp_info.get("seq_len") or 60)
            if len(df_feat_recent) < q:
                return None
            Xseq = _safe_make_lstm_sequences(df_feat_recent, seq_len=q)
            if Xseq is None or len(Xseq) == 0:
                return None
            with torch.no_grad():
                p = float(comp_info["model"].infer(torch.from_numpy(Xseq[-1:])).cpu().numpy().reshape(-1)[0])
            return p

        return None

    # ---------- L2 builder pro aktivní sadu ----------
    def _build_L2_inputs(self, df_feat_recent, K_override: int | None = None):
        if (self.l2_gate is None) or (not self.l2_sets):
            return False, None
        K = int(self.l2_seq_len if K_override is None else K_override)
        K_raw = int(self.l1_seq_len) + int(K) - 1
        per_set = []
        for sdir in self.l2_sets:
            cache = self._l2_sets_cache.get(sdir) or {}
            info_l1  = cache.get("l1", {})
            info_xgb = cache.get("xgb", {})
            info_lstm= cache.get("lstm", {})
            info_hrm = cache.get("hrm", {})

            p_series = []
            px = self._roll_xgb(info_xgb.get("model"), info_xgb.get("calib"), df_feat_recent, K_raw)
            if px is not None: p_series.append(px.reshape(-1,1))
            ql = int(info_lstm.get("seq_len") or 60) if info_lstm else 60
            pl = self._roll_lstm(info_lstm.get("model"), info_lstm.get("calib"), df_feat_recent, ql, K_raw)
            if pl is not None: p_series.append(pl.reshape(-1,1))
            qh = int(info_hrm.get("seq_len") or 60) if info_hrm else 60
            ph = self._roll_hrm(info_hrm.get("model"), df_feat_recent, qh, K_raw)
            if ph is not None: p_series.append(ph.reshape(-1,1))
            if not p_series:
                return False, None

            raw_mat = np.concatenate(p_series, axis=1)
            if raw_mat.shape[0] < K_raw:
                return False, None

            raw_tail = raw_mat[-K_raw:, :]
            l1_mdl = info_l1.get("model")
            l1_in = int(info_l1.get("in_features") or raw_tail.shape[1])
            if (l1_mdl is None) or (l1_in != raw_tail.shape[1]):
                return False, None
            L1 = int(self.l1_seq_len)
            Xs = np.stack([raw_tail[t-L1+1:t+1, :] for t in range(L1-1, L1-1+K)], axis=0).astype(np.float32)
            with torch.no_grad():
                p_l1 = l1_mdl.infer(torch.from_numpy(Xs)).cpu().numpy().reshape(-1).astype(np.float32)
            rec = {"l1_series": p_l1}
            if self.l2_use_raw:
                rec["raw_series"] = raw_tail[-K:, :]
            per_set.append(rec)

        cols = [rec["l1_series"].reshape(K,1) for rec in per_set]
        if self.l2_use_raw:
            for rec in per_set:
                cols.append(rec["raw_series"])

        mat = np.concatenate(cols, axis=1)
        return True, mat.astype(np.float32).reshape(1, K, -1)

    # ---------- L2 série pro cizí hlavu (pro L3 vstup) ----------
    def _roll_L2_series_for_env(self, env, df_feat_recent, K_out: int):
        L1 = int(env["l1_seq_len"])
        L2 = int(env["l2_seq_len"])
        use_raw = bool(env["use_raw"])
        T_total = L2 + K_out - 1
        raw_len = L1 + T_total - 1

        per_set = []
        for sdir in env["sets"]:
            cache = env["sets_cache"][sdir]
            info_l1  = cache.get("l1", {})
            info_xgb = cache.get("xgb", {})
            info_lstm= cache.get("lstm", {})
            info_hrm = cache.get("hrm", {})

            p_series = []
            px = self._roll_xgb(info_xgb.get("model"), info_xgb.get("calib"), df_feat_recent, raw_len)
            if px is not None: p_series.append(px.reshape(-1,1))
            ql = int(info_lstm.get("seq_len") or 60) if info_lstm else 60
            pl = self._roll_lstm(info_lstm.get("model"), info_lstm.get("calib"), df_feat_recent, ql, raw_len)
            if pl is not None: p_series.append(pl.reshape(-1,1))
            qh = int(info_hrm.get("seq_len") or 60) if info_hrm else 60
            ph = self._roll_hrm(info_hrm.get("model"), df_feat_recent, qh, raw_len)
            if ph is not None: p_series.append(ph.reshape(-1,1))
            if not p_series:
                return None

            raw_mat = np.concatenate(p_series, axis=1)
            if raw_mat.shape[0] < raw_len:
                return None
            l1_mdl = info_l1.get("model")
            l1_in = int(info_l1.get("in_features") or raw_mat.shape[1])
            if (l1_mdl is None) or (l1_in != raw_mat.shape[1]):
                return None
            Xs = np.stack([raw_mat[t-L1+1:t+1, :] for t in range(L1-1, L1-1+T_total)], axis=0).astype(np.float32)
            with torch.no_grad():
                p_l1 = l1_mdl.infer(torch.from_numpy(Xs)).cpu().numpy().reshape(-1).astype(np.float32)  # (T_total,)
            rec = {"p_l1_series": p_l1}
            if use_raw:
                rec["raw_mat"] = raw_mat
            per_set.append(rec)

        X_l2 = []
        for i in range(K_out):
            cols = [rec["p_l1_series"][i:i+L2].reshape(L2,1) for rec in per_set]  # L1 blok
            if use_raw:
                for rec in per_set:
                    cols.append(rec["raw_mat"][i:i+L2, :])  # RAW blok
            mat = np.concatenate(cols, axis=1)  # (L2, C_total)
            X_l2.append(mat)
        X_l2 = np.stack(X_l2, axis=0).astype(np.float32)  # (K_out, L2, C_total)

        with torch.no_grad():
            p = env["l2_model"].infer(torch.from_numpy(X_l2)).cpu().numpy().reshape(-1).astype(np.float32)
        if env.get("l2_cal") is not None:
            p = np.array([_apply_platt_or_isotonic(env["l2_cal"], float(pi)) for pi in p], dtype=np.float32)
        return p  # (K_out,)

    # ---------- L3 vstup ----------
    def _build_L3_inputs(self, df_feat_recent):
        # L3 může být single nebo bag – ale vstupy jsou stejné (dané supervisor.json)
        if (self.l3_supervisor is None) and (self.l3_bag is None):
            return False, None
        K = int(self.l3_seq_len)

        cols_all = []

        # (A) L1 + RAW blok (podle sets_l1)
        if self.l3_use_l1 or self.l3_use_raw:
            K_raw = int(self.l1_seq_len) + K - 1
            for sdir in self.l3_sets:
                cache = self._l3_sets_cache.get(sdir) or {}
                info_l1  = cache.get("l1", {})
                info_xgb = cache.get("xgb", {})
                info_lstm= cache.get("lstm", {})
                info_hrm = cache.get("hrm", {})

                p_series = []
                px = self._roll_xgb(info_xgb.get("model"), info_xgb.get("calib"), df_feat_recent, K_raw)
                if px is not None: p_series.append(px.reshape(-1,1))
                ql = int(info_lstm.get("seq_len") or 60) if info_lstm else 60
                pl = self._roll_lstm(info_lstm.get("model"), info_lstm.get("calib"), df_feat_recent, ql, K_raw)
                if pl is not None: p_series.append(pl.reshape(-1,1))
                qh = int(info_hrm.get("seq_len") or 60) if info_hrm else 60
                ph = self._roll_hrm(info_hrm.get("model"), df_feat_recent, qh, K_raw)
                if ph is not None: p_series.append(ph.reshape(-1,1))
                if not p_series:
                    return False, None

                raw_mat = np.concatenate(p_series, axis=1)
                if raw_mat.shape[0] < K_raw:
                    return False, None
                raw_tail = raw_mat[-K_raw:, :]

                rec_cols = []
                if self.l3_use_l1:
                    l1_mdl = info_l1.get("model")
                    l1_in = int(info_l1.get("in_features") or raw_tail.shape[1])
                    if (l1_mdl is None) or (l1_in != raw_tail.shape[1]):
                        return False, None
                    L1 = int(self.l1_seq_len)
                    Xs = np.stack([raw_tail[t-L1+1:t+1, :] for t in range(L1-1, L1-1+K)], axis=0).astype(np.float32)
                    with torch.no_grad():
                        p_l1 = l1_mdl.infer(torch.from_numpy(Xs)).cpu().numpy().reshape(-1).astype(np.float32)
                    rec_cols.append(p_l1.reshape(K,1))
                if self.l3_use_raw:
                    rec_cols.append(raw_tail[-K:, :])

                if rec_cols:
                    cols_all.extend(rec_cols)

        # (B) L2 blok (více hlav)
        if self.l3_use_l2 and len(self.l3_l2_heads) > 0:
            for head_dir in self.l3_l2_heads:
                env = self._l2_envs.get(head_dir)
                if not env:
                    continue
                p_l2 = self._roll_L2_series_for_env(env, df_feat_recent, K_out=K)
                if p_l2 is not None:
                    cols_all.append(p_l2.reshape(K,1))

        # (C) režimové featury — poslední r-vector tilovat přes K (pokud dim. sedí)
        if self.l3_use_regime and df_feat_recent is not None and len(df_feat_recent) > 0:
            rvec = _compute_regime_vector(df_feat_recent, len(df_feat_recent)-1)  # (R,)
            if (self.l3_regime_mean is not None and self.l3_regime_std is not None and
                rvec.size == getattr(self.l3_regime_mean, "size", 0) == getattr(self.l3_regime_std, "size", 0)):
                rvec = (rvec - self.l3_regime_mean) / (self.l3_regime_std + 1e-12)
                reg_blk = np.tile(rvec.reshape(1, -1), (K, 1)).astype(np.float32)
                cols_all.append(reg_blk)

        if not cols_all:
            return False, None
        mat = np.concatenate(cols_all, axis=1)  # (K, C_total)
        return True, mat.astype(np.float32).reshape(1, K, -1)

    # ---------- public ----------
    def ready(self) -> bool:
        have_base = (self.xgb is not None) or (self.lstm is not None) or (self.hrm is not None) or (self.hrm_meta is not None)
        have_gate_mlp = (self.meta_gate is not None) and len(self.loaded_components) >= 2
        have_l2 = (self.l2_gate is not None) and len(self.l2_sets) >= 1
        have_l3 = (self.l3_supervisor is not None) or (self.l3_bag is not None)
        return have_base or have_gate_mlp or have_l2 or have_l3

    def predict_detail(self, df_feat_recent):
        # standardizovaný výstup
        out = {
            "p_xgb": None, "p_lstm": None, "p_hrm": None, "p_meta": None,
            "p_l2": None, "p_l3": None,
            "p_abstain": None, "p_uncert": None, "p3": None, "hsel": None,
            "p_ens": 0.5, "prob_up": None
        }
        if df_feat_recent is None or len(df_feat_recent) == 0:
            out["prob_up"] = out["p_ens"]
            return out

        # base KPI
        px = None
        if self.xgb is not None and len(df_feat_recent) >= 1:
            X = df_feat_recent[FEATURE_COLS].values[-1:].astype(np.float32)
            try:
                px_raw = float(self.xgb.predict_proba(X)[:, 1][0])
            except Exception:
                px_raw = None
            px = _apply_platt_or_isotonic(self.xgb_cal, px_raw) if px_raw is not None else None
        out["p_xgb"] = px

        pl = None
        if self.lstm is not None and len(df_feat_recent) >= self.seq_len:
            Xseq = _safe_make_lstm_sequences(df_feat_recent, seq_len=int(self.seq_len))
            if Xseq is not None and len(Xseq) > 0:
                with torch.no_grad():
                    p = float(self.lstm(torch.from_numpy(Xseq[-1:])).cpu().numpy().reshape(-1)[0])
                pl = _apply_platt_or_isotonic(self.lstm_cal, p)
        out["p_lstm"] = pl

        ph = None
        if self.hrm is not None and len(df_feat_recent) >= self.seq_len:
            Xseq = _safe_make_lstm_sequences(df_feat_recent, seq_len=int(self.seq_len))
            if Xseq is not None and len(Xseq) > 0:
                with torch.no_grad():
                    ph = float(self.hrm.infer(torch.from_numpy(Xseq[-1:])).cpu().numpy().reshape(-1)[0])
        out["p_hrm"] = ph

        pm = None
        comps_basic = [p for p in [px, pl, ph] if p is not None]
        if self.hrm_meta is not None and len(comps_basic) >= 2:
            pmat = np.array(comps_basic, dtype=np.float32).reshape(1, -1)
            with torch.no_grad():
                pm = float(self.hrm_meta.infer(torch.from_numpy(pmat)).cpu().numpy()[0])
        out["p_meta"] = pm

        # L2 (aktivní)
        p_gate_l2 = None
        ok_L2, inputs_l2 = self._build_L2_inputs(df_feat_recent)
        if ok_L2:
            with torch.no_grad():
                p = float(self.l2_gate.infer(torch.from_numpy(inputs_l2)).cpu().numpy().reshape(-1)[0])
            if self.l2_cal is not None:
                p = _apply_platt_or_isotonic(self.l2_cal, p)
            p_gate_l2 = p
        out["p_l2"] = p_gate_l2

        # L3 (single nebo bagging)
        p_supervisor = None
        p_abs = None
        p_unc = None
        p3 = None
        hsel = None

        ok_L3, inputs_l3 = self._build_L3_inputs(df_feat_recent)
        if ok_L3:
            # BAG
            if self.l3_bag is not None and self.l3_bag.get("models"):
                ps = []
                for it in self.l3_bag["models"]:
                    mdl = it["model"]
                    with torch.no_grad():
                        p = float(mdl(torch.from_numpy(inputs_l3)).detach().cpu().numpy().reshape(-1)[-1])
                    ps.append(p)
                if ps:
                    p_supervisor = float(np.mean(ps))
                    cal = self.l3_bag.get("calib")
                    if cal is not None:
                        p_supervisor = _apply_platt_or_isotonic(cal, p_supervisor)

            # SINGLE
            elif self.l3_supervisor is not None:
                mdl = self.l3_supervisor
                with torch.no_grad():
                    # hlavní UP pravděpodobnost
                    p_supervisor = float(mdl(torch.from_numpy(inputs_l3)).detach().cpu().numpy().reshape(-1)[-1])

                    # volitelné hlavy (pokud je model má)
                    if hasattr(mdl, "forward_heads"):
                        outs = mdl.forward_heads(torch.from_numpy(inputs_l3))
                        if "abstain" in outs:
                            p_abs = float(outs["abstain"][-1].detach().cpu().numpy().reshape(-1)[0])
                        if "uncert" in outs:
                            p_unc = float(outs["uncert"][-1].detach().cpu().numpy().reshape(-1)[0])
                        if "softmax3" in outs:
                            logit3 = outs["softmax3"][-1].detach().cpu().numpy().reshape(-1)
                            e = np.exp(logit3 - np.max(logit3))
                            p3 = (e / (e.sum() + 1e-12)).astype(np.float32).tolist()
                        if "hsel" in outs:
                            hsel = outs["hsel"][-1].detach().cpu().numpy().reshape(-1).astype(np.float32).tolist()

                if self.l3_cal is not None and p_supervisor is not None:
                    p_supervisor = _apply_platt_or_isotonic(self.l3_cal, p_supervisor)

        out["p_l3"] = p_supervisor
        out["p_abstain"] = p_abs
        out["p_uncert"] = p_unc
        out["p3"] = p3
        out["hsel"] = hsel

        # MLP gate (původní laditelný)
        p_gate_mlp = None
        p_manual = None
        x_vec, probs_map, ok = self._build_gate_vector_mlp(df_feat_recent)
        if ok:
            self._last_comp_probs = dict(probs_map)
            try:
                p_gate_mlp = self.meta_gate.infer1(x_vec)
            except Exception:
                p_gate_mlp = None
            try:
                keys = [c["key"] for c in self.gate_components]
                pvals = np.array([probs_map[k] if probs_map[k] is not None else 0.5 for k in keys], dtype=np.float32)
                gains = np.array([float(self.gate_gains.get(k, 1.0)) for k in keys], dtype=np.float32)
                w = _softmax(np.log(np.maximum(gains, 1e-6)))
                p_manual = float((w * pvals).sum())
            except Exception:
                p_manual = None
        else:
            self._last_comp_probs = {}

        # priority & fallback do p_ens
        def _fallback_basic():
            if pm is not None:
                return float(pm)
            else:
                avail = [p for p in [ph, pl, px] if p is not None]
                return float(np.mean(avail)) if len(avail) else 0.5

        if p_supervisor is not None:
            p_ens = float(p_supervisor)
        elif p_gate_l2 is not None:
            p_ens = float(p_gate_l2)
        else:
            def _fallback_mlp():
                return _fallback_basic()
            if self.gate_mode == "auto":
                p_ens = float(p_gate_mlp) if p_gate_mlp is not None else _fallback_mlp()
            elif self.gate_mode == "manual":
                p_ens = float(p_manual) if p_manual is not None else _fallback_mlp()
            else:
                if (p_gate_mlp is not None) and (p_manual is not None):
                    a = float(self.gate_alpha)
                    p_ens = float((1.0 - a) * p_gate_mlp + a * p_manual)
                else:
                    p_ens = _fallback_mlp()

        out["p_ens"] = p_ens
        out["prob_up"] = p_ens  # alias pro kompatibilitu s main.py
        return out