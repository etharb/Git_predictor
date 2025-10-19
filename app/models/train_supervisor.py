# -*- coding: utf-8 -*-
"""
Trénink L3 "Supervisoru" – sekvenční HRMHead, který kombinuje:
  • L1 blok: p_meta_L1 napříč sadami (sets_l1)
  • RAW blok: p_xgb/p_lstm/p_hrm napříč sadami (volitelně)
  • L2 blok: výstupy z více L2 hlav (sets_l2)

Rozšíření podle zadání:
  • Triple-Barrier/abstain labeling (UP/DOWN/ABSTAIN) + okraj edge_min_bps
  • Cost/utility-aware váhy (fee_bps + slip_bps) -> ovlivní ztrátu
  • Volitelné pomocné hlavy v HRM: abstain, softmax3, uncertainty, horizon-selector
  • (Volitelně) režimové featury (time-of-day, vol, atd.) přimíchané do L3 sekvence

VÝSTUPY
-------
  • supervisor_L3.pt          – HRMHead checkpoint (+metadata v dictu)
  • supervisor_L3.calib.pkl   – volitelná Platt kalibrace (LogReg na validaci)
  • supervisor.json           – popis konfigurace (seq_len, use_*_in_l3, sets_l1, sets_l2, ...)
  • (volitelně) meta.json     – doplněné doporučené prahy
"""

import os, json, argparse, math, joblib
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from .hrm_model import HRMHead
from .lstm_model import SmallLSTM
from .ensemble import EnsemblePredictor  # kompatibilní loader logika
from ..utils.features import FEATURE_COLS, compute_features, make_lstm_sequences

# ---- utils ----

def _read_json(path: str) -> dict:
    if os.path.exists(path):
        try:
            return json.load(open(path, "r"))
        except Exception:
            return {}
    return {}

def _write_json(path: str, data: dict):
    try:
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        return True
    except Exception:
        return False

def _read_meta_l2(out_dir: str) -> dict:
    return _read_json(os.path.join(out_dir, "meta.json"))

def _read_supervisor_json(out_dir: str) -> dict:
    return _read_json(os.path.join(out_dir, "supervisor.json"))

def _get_l2_sets_from_meta(meta_json: dict) -> List[str]:
    mg = (meta_json.get("meta_gate") or {})
    sets = mg.get("sets") or []
    return [os.path.abspath(s) for s in sets]

def _get_horizon_from_meta(meta_json: dict, default_hz: int = 1) -> int:
    try:
        return int(meta_json.get("horizon_sec") or default_hz)
    except Exception:
        return default_hz

def _get_l1_seq_len_from_meta(meta_json: dict, default_l1: int = 60) -> int:
    mg = (meta_json.get("meta_gate") or {})
    try:
        return int(mg.get("l1_seq_len") or default_l1)
    except Exception:
        return default_l1

def _ensure_features(df: pd.DataFrame) -> pd.DataFrame:
    """Když v df chybí FEAT sloupce, tak je spočítáme z OHLCV."""
    if all(col in df.columns for col in FEATURE_COLS):
        if df.index.name != "ts" and "ts" in df.columns:
            df = df.set_index("ts")
        return df.sort_index()
    if df.index.name != "ts" and "ts" in df.columns:
        df = df.set_index("ts")
    df = df.sort_index()
    return compute_features(df)

def _sigmoid_to_logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    pp = np.clip(p, eps, 1.0 - eps)
    return np.log(pp / (1.0 - pp))

def _safe_numpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()


# --- kompatibilní nahrání HRM checkpointů (staré vs nové názvy hlav) ---
def _load_hrm_state_dict_compat(model: nn.Module, state_dict: dict):
    """
    Přemapuje legacy klíče 'head.*' -> 'head_updown.*' a načte state_dict se strict=False,
    aby se tolerovaly volitelné hlavy (abstain/uncert/softmax3/hsel).
    """
    raw = state_dict or {}
    sd = {}
    for k, v in raw.items():
        if k.startswith("head."):
            sd["head_updown" + k[len("head"):]] = v
        else:
            sd[k] = v

    # pokus s strict=True (když sedí), jinak strict=False (toleruje volitelné hlavy)
    try:
        model.load_state_dict(sd, strict=True)
    except Exception:
        model.load_state_dict(sd, strict=False)
        
# ---- načítání modelů v sadě (stejně jako v EnsemblePredictor) ----

def _load_model_from_set(set_dir: str, mtype: str) -> dict:
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
            mdl.load_state_dict(ckpt['state_dict']); mdl.eval()
            out["model"] = mdl
        lc = os.path.join(set_dir, "lstm.calib.pkl")
        if os.path.exists(lc):
            out["calib"] = joblib.load(lc)
    elif mtype == "hrm":
        hm = os.path.join(set_dir, "hrm.pt")
        if os.path.exists(hm):
            ckpt = torch.load(hm, map_location="cpu")
            mdl = HRMHead(in_features=int(ckpt['in_features']),
                          hidden_low=int(ckpt['hidden_low']),
                          hidden_high=int(ckpt['hidden_high']),
                          high_period=int(ckpt['high_period']))
            _load_hrm_state_dict_compat(mdl, ckpt.get('state_dict') or {})
            mdl.eval()
            out["model"] = mdl
    elif mtype == "hrm_meta":
        hmm = os.path.join(set_dir, "hrm_meta.pt")
        if os.path.exists(hmm):
            ckpt = torch.load(hmm, map_location="cpu")
            mdl = HRMHead(in_features=int(ckpt['in_features']),
                          hidden_low=int(ckpt['hidden_low']),
                          hidden_high=int(ckpt['hidden_high']),
                          high_period=int(ckpt['high_period']))
            _load_hrm_state_dict_compat(mdl, ckpt.get('state_dict') or {})
            mdl.eval()
            out["model"] = mdl
            out["in_features"] = int(ckpt['in_features'])
            out["seq_len"] = int(ckpt.get("seq_len") or 60)
    elif mtype in ("l2", "hrm_meta_L2"):
        p2 = os.path.join(set_dir, "hrm_meta_L2.pt")
        if os.path.exists(p2):
            ckpt = torch.load(p2, map_location="cpu")
            mdl = HRMHead(in_features=int(ckpt['in_features']),
                          hidden_low=int(ckpt['hidden_low']),
                          hidden_high=int(ckpt['hidden_high']),
                          high_period=int(ckpt['high_period']))
            mdl.load_state_dict(ckpt['state_dict']); mdl.eval()
            out["model"] = mdl
            out["in_features"] = int(ckpt['in_features'])
            out["seq_len"] = int(ckpt.get("seq_len") or 60)
        calp = os.path.join(set_dir, "meta_L2.calib.pkl")
        if os.path.exists(calp):
            try: out["calib"] = joblib.load(calp)
            except Exception: out["calib"] = None
    return out

# ---- výpočet raw kanálů po celé délce (XGB/LSTM/HRM) ----

def _full_series_xgb(info: dict, df_feat: pd.DataFrame) -> np.ndarray | None:
    mdl = info.get("model")
    cal = info.get("calib")
    if mdl is None:
        return None
    try:
        X = df_feat[FEATURE_COLS].values.astype(np.float32)
    except Exception:
        return None

    p = None
    try:
        if hasattr(mdl, "predict_proba"):
            proba = mdl.predict_proba(X)
            if isinstance(proba, (list, tuple)):
                proba = np.asarray(proba)
            if hasattr(proba, "toarray"):
                proba = proba.toarray()
            proba = np.asarray(proba, dtype=np.float32)
            if proba.ndim == 1:
                p = proba
            elif proba.ndim == 2 and proba.shape[1] >= 2:
                p = proba[:, 1]
            elif proba.ndim == 2 and proba.shape[1] == 1:
                p = proba[:, 0]
        elif hasattr(mdl, "decision_function"):
            z = mdl.decision_function(X).astype(np.float32)
            p = 1.0 / (1.0 + np.exp(-z))
        elif hasattr(mdl, "predict"):
            preds = mdl.predict(X).astype(np.float32)
            # pokud jsou predikce binární (0/1), necháme je tak
            if np.unique(preds).size == 2:
                p = preds
            else:
                # fallback — rescale do <0,1>
                p = (preds - preds.min()) / (preds.max() - preds.min() + 1e-12)
    except Exception as e:
        print(f"[WARN] _full_series_xgb failed: {e}")
        return None

    if p is None:
        return None

    # Kalibrace
    if cal is not None:
        try:
            if hasattr(cal, "transform"):
                p = np.asarray(cal.transform(p), dtype=np.float32)
            elif hasattr(cal, "predict_proba"):
                eps = 1e-6
                pp = np.clip(p, eps, 1 - eps)
                logit = np.log(pp / (1 - pp)).reshape(-1, 1)
                p = cal.predict_proba(logit)[:, 1].astype(np.float32)
        except Exception as e:
            print(f"[WARN] calibration failed: {e}")

    return np.clip(p.astype(np.float32), 0.0, 1.0)
    
def _full_series_lstm(info: dict, df_feat: pd.DataFrame) -> np.ndarray:
    mdl = info.get("model"); cal = info.get("calib")
    if mdl is None: return None
    q = int(info.get("seq_len") or 60)
    Xseq = make_lstm_sequences(df_feat, seq_len=q)
    if isinstance(Xseq, (list, tuple)):
        Xseq = Xseq[0]
    if Xseq is None or len(Xseq)==0:
        return None
    with torch.no_grad():
        p = _safe_numpy(mdl(torch.from_numpy(Xseq))).reshape(-1).astype(np.float32)
    if cal is not None:
        if hasattr(cal, "transform"):
            p = cal.transform(p)
        elif hasattr(cal, "predict_proba"):
            eps = 1e-6
            pp = np.clip(p, eps, 1 - eps)
            logit = np.log(pp / (1 - pp)).reshape(-1, 1)
            p = cal.predict_proba(logit)[:, 1]
    out = np.full((len(df_feat),), np.nan, dtype=np.float32)
    out[q-1:] = p
    return out

def _full_series_hrm(info: dict, df_feat: pd.DataFrame) -> np.ndarray:
    mdl = info.get("model")
    if mdl is None: return None
    q = int(info.get("seq_len") or 60)
    Xseq = make_lstm_sequences(df_feat, seq_len=q)
    if isinstance(Xseq, (list, tuple)):
        Xseq = Xseq[0]
    if Xseq is None or len(Xseq)==0:
        return None
    with torch.no_grad():
        p = _safe_numpy(mdl.infer(torch.from_numpy(Xseq))).reshape(-1).astype(np.float32)
    out = np.full((len(df_feat),), np.nan, dtype=np.float32)
    out[q-1:] = p
    return out

def _build_raw_matrix_for_set(set_dir: str, df_feat: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    """
    Vrací (raw_mat, cols) kde raw_mat má shape (T, C_raw) s p_xgb,p_lstm,p_hrm (kde jsou k dispozici).
    Na začátku série jsou NaN, dokud nenaběhne potřebná délka pro daný model.
    """
    info_xgb = _load_model_from_set(set_dir, "xgb")
    info_lstm = _load_model_from_set(set_dir, "lstm")
    info_hrm = _load_model_from_set(set_dir, "hrm")

    cols = []
    names = []

    px = _full_series_xgb(info_xgb, df_feat)
    if px is not None:
        cols.append(px.reshape(-1, 1))
        names.append("xgb")

    pl = _full_series_lstm(info_lstm, df_feat)
    if pl is not None:
        cols.append(pl.reshape(-1, 1))
        names.append("lstm")

    ph = _full_series_hrm(info_hrm, df_feat)
    if ph is not None:
        cols.append(ph.reshape(-1, 1))
        names.append("hrm")

    if not cols:
        print(f"[WARN] V sadě {set_dir} nebyl nalezen žádný kanál (xgb/lstm/hrm). Vracíme prázdnou matici.")
        return np.zeros((len(df_feat), 0), dtype=np.float32), []

    raw = np.concatenate(cols, axis=1).astype(np.float32)
    return raw, names
    
def _compute_l1_series_for_set(set_dir: str, raw_mat: np.ndarray, df_feat_len: int, l1_seq_len: int) -> np.ndarray:
    """Spočti p_meta_L1[t] přes okno L1 x C_raw pomocí hrm_meta.pt v setu. Na místech bez dat → NaN."""
    info_l1 = _load_model_from_set(set_dir, "hrm_meta")
    l1 = info_l1.get("model")
    if l1 is None:
        raise RuntimeError(f"V sadě {set_dir} chybí hrm_meta.pt (L1).")

    need = int(info_l1.get("in_features") or raw_mat.shape[1])
    if need != raw_mat.shape[1]:
        raise RuntimeError(f"{set_dir}: hrm_meta.in_features={need} neodpovídá C_raw={raw_mat.shape[1]}.")

    T = int(df_feat_len)
    L1 = int(l1_seq_len)
    out = np.full((T,), np.nan, dtype=np.float32)

    xs = []
    idxs = []
    for t in range(L1-1, T):
        block = raw_mat[t-L1+1:t+1, :]
        if np.isnan(block).any():
            continue
        xs.append(block)
        idxs.append(t)

    if not xs:
        return out

    Xs = np.stack(xs, axis=0).astype(np.float32)
    with torch.no_grad():
        p = _safe_numpy(l1.infer(torch.from_numpy(Xs))).reshape(-1).astype(np.float32)
    for t, val in zip(idxs, p):
        out[t] = float(val)
    return out

# ---------- L2 heads: příprava a plná série p_L2(t) ----------
def _prepare_env_for_l2_head(head_dir: str, df_feat: pd.DataFrame, l1_seq_len_fallback: int = 60) -> dict:
    """
    Načte meta.json/head, zjistí L2 parametry a vstupní sady, spočítá RAW a L1 série pro tyto sady.
    Vrací env dict: {l2_model,l2_cal,l2_seq_len,l1_seq_len,use_raw, sets, raw_by_set, p_l1_by_set}
    """
    head_dir = os.path.abspath(head_dir)
    meta = _read_json(os.path.join(head_dir, "meta.json"))
    cfg = (meta.get("meta_gate") or {})
    sets = [os.path.abspath(s) for s in (cfg.get("sets") or [])]
    if not sets:
        raise RuntimeError(f"[L2 ENV] {head_dir}: meta_gate.sets je prázdný.")

    l2_info = _load_model_from_set(head_dir, "l2")
    if l2_info["model"] is None:
        raise RuntimeError(f"[L2 ENV] {head_dir}: chybí hrm_meta_L2.pt")

    L2 = int(l2_info.get("seq_len") or int(cfg.get("l2_seq_len") or 60))
    L1 = int(cfg.get("l1_seq_len") or l1_seq_len_fallback)
    use_raw = bool(cfg.get("use_raw_in_l2", False))

    raw_by_set = {}
    p_l1_by_set = {}
    for s in sets:
        raw_mat, _ = _build_raw_matrix_for_set(s, df_feat)
        raw_by_set[s] = raw_mat
        p_l1_by_set[s] = _compute_l1_series_for_set(s, raw_mat, len(df_feat), L1)

    env = {
        "l2_model": l2_info["model"],
        "l2_cal":   l2_info.get("calib"),
        "l2_seq_len": L2,
        "l1_seq_len": L1,
        "use_raw": use_raw,
        "sets": sets,
        "raw_by_set": raw_by_set,
        "p_l1_by_set": p_l1_by_set,
    }
    return env

def _full_series_l2_from_env(env: dict) -> np.ndarray:
    """
    Z env (kde jsou p_L1(t) napříč sadami + případné RAW) vyrobí plnou sérii p_L2(t) (NaN před naběhem).
    """
    sets = env["sets"]
    L1 = int(env["l1_seq_len"]); L2 = int(env["l2_seq_len"])
    use_raw = bool(env["use_raw"])
    p_l1_by_set = env["p_l1_by_set"]
    raw_by_set  = env["raw_by_set"]

    # slož (T, S) z L1 sérií
    T = min(len(p_l1_by_set[s]) for s in sets)
    P = np.stack([p_l1_by_set[s][:T] for s in sets], axis=1)  # (T,S)

    raw_cat = None
    if use_raw:
        raws = [raw_by_set[s][:T, :] for s in sets]
        raw_cat = np.concatenate(raws, axis=1).astype(np.float32)  # (T,sumCraw)

    # poskládej valid L2 okna
    X_list = []
    idxs = []
    for t in range(L2-1, T):
        block_l1 = P[t-L2+1:t+1, :]  # (L2,S)
        if np.isnan(block_l1).any():
            continue
        cols = [block_l1]
        if raw_cat is not None:
            block_raw = raw_cat[t-L2+1:t+1, :]  # (L2,sumCraw)
            if np.isnan(block_raw).any():
                continue
            cols.append(block_raw)
        X = np.concatenate(cols, axis=1).astype(np.float32)  # (L2, S+sumCraw)
        X_list.append(X)
        idxs.append(t)

    out = np.full((T,), np.nan, dtype=np.float32)
    if not X_list:
        return out

    X_all = np.stack(X_list, axis=0).astype(np.float32)  # (N, L2, C)
    mdl = env["l2_model"]
    with torch.no_grad():
        p = _safe_numpy(mdl.infer(torch.from_numpy(X_all))).reshape(-1).astype(np.float32)
    if env.get("l2_cal") is not None:
        cal = env["l2_cal"]
        if hasattr(cal, "transform"):  # isotonic (na vektoru)
            p = cal.transform(p)
        elif hasattr(cal, "predict_proba"):
            logit = _sigmoid_to_logit(p).reshape(-1,1)
            p = cal.predict_proba(logit)[:,1].astype(np.float32)

    for t, val in zip(idxs, p):
        out[t] = float(val)
    return out

# ---- Režimové featury (volitelné) ----

def _compute_regime_vector(df_feat: pd.DataFrame, t_idx: int) -> np.ndarray:
    """
    Velmi lehké režimové featury na čase t_idx (index po sort_index).
      • sin/cos z minuty a hodiny v rámci dne (pokud index je epocha v sekundách)
      • 60s realizovaná volatilita close diffů (rolling std)
      • 60s SMA volume (pokud sloupec existuje)
    Vrací vektor (R,), nebo prázdný vektor pokud nic neumíme spočítat.
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
    # 60s vol
    if "close" in df_feat.columns:
        start = max(0, t_idx - 59)
        c = df_feat["close"].values[start:t_idx+1].astype(np.float32)
        if len(c) >= 2:
            r = np.diff(c) / (c[:-1] + 1e-12)
            cols.append(float(np.std(r)))
        else:
            cols.append(0.0)
    # 60s vol SMA
    if "volume" in df_feat.columns:
        start = max(0, t_idx - 59)
        v = df_feat["volume"].values[start:t_idx+1].astype(np.float32)
        cols.append(float(np.mean(v)))
    return np.array(cols, dtype=np.float32)

# ---- skládání L3 oken ----

def _build_l3_windows(
    p_l1_by_set: Dict[str, np.ndarray],
    raw_by_set: Dict[str, np.ndarray],
    p_l2_by_head: Dict[str, np.ndarray],
    use_raw_in_l3: bool,
    use_l1_in_l3: bool,
    use_l2_in_l3: bool,
    l3_seq_len: int,
    df_feat: pd.DataFrame = None,
    use_regime_in_l3: bool = False,
    regime_mean: np.ndarray = None,
    regime_std: np.ndarray = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Z dostupných bloků (L1 napříč sadami, RAW napříč sadami, L2 napříč hlavami) + (volitelně) režimové featury
    sestaví pro každý čas t okno délky L3. Vrací (windows, valid_idx_end).
    """
    L3 = int(l3_seq_len)

    # (A) L1/RAW část
    set_keys = list(p_l1_by_set.keys())
    T_candidates = []
    if use_l1_in_l3 and set_keys:
        T_candidates.append(min(len(p_l1_by_set[k]) for k in set_keys))
    if use_raw_in_l3 and set_keys and raw_by_set:
        T_candidates.append(min(raw_by_set[k].shape[0] for k in set_keys))

    # (B) L2 část
    l2_keys = list(p_l2_by_head.keys()) if use_l2_in_l3 and p_l2_by_head else []
    if l2_keys:
        T_candidates.append(min(len(p_l2_by_head[k]) for k in l2_keys))

    if not T_candidates:
        return np.empty((0, L3, 0), dtype=np.float32), np.array([], dtype=np.int64)

    T = min(T_candidates)

    # připrav podkladové matice časem zarovnané na T
    P_l1 = None
    if use_l1_in_l3 and set_keys:
        P_l1 = np.stack([p_l1_by_set[k][:T] for k in set_keys], axis=1)  # (T,S)

    RAW = None
    if use_raw_in_l3 and set_keys and raw_by_set:
        raws = [raw_by_set[k][:T, :] for k in set_keys]
        RAW = np.concatenate(raws, axis=1).astype(np.float32)  # (T,sumCraw)

    P_l2 = None
    if l2_keys:
        P_l2 = np.stack([p_l2_by_head[k][:T] for k in l2_keys], axis=1).astype(np.float32)  # (T,H)

    # režimové featury (per-time)
    REG = None
    Rdim = 0
    if use_regime_in_l3 and df_feat is not None:
        regs = []
        for t in range(T):
            rvec = _compute_regime_vector(df_feat, t)  # (R,)
            if regime_mean is not None and regime_std is not None and rvec.size == regime_mean.size:
                rvec = (rvec - regime_mean) / (regime_std + 1e-12)
            regs.append(rvec.reshape(1, -1))
        REG = np.concatenate(regs, axis=0).astype(np.float32)  # (T,R)
        Rdim = REG.shape[1] if REG is not None else 0

    # generuj okna
    X_list, idx_end = [], []
    for t in range(L3-1, T):
        cols = []
        if P_l1 is not None:
            blk = P_l1[t-L3+1:t+1, :]
            if np.isnan(blk).any(): 
                continue
            cols.append(blk)
        if RAW is not None:
            blk = RAW[t-L3+1:t+1, :]
            if np.isnan(blk).any(): 
                continue
            cols.append(blk)
        if P_l2 is not None:
            blk = P_l2[t-L3+1:t+1, :]
            if np.isnan(blk).any(): 
                continue
            cols.append(blk)
        if REG is not None and Rdim > 0:
            # režimový vektor duplicujeme přes časové kroky okna
            reg_blk = np.tile(REG[t].reshape(1, -1), (L3, 1))  # (L3,R)
            cols.append(reg_blk)

        if not cols: 
            continue
        X = np.concatenate(cols, axis=1).astype(np.float32)  # (L3, C_total)
        X_list.append(X)
        idx_end.append(t)

    if not X_list:
        return np.empty((0, L3, 0), dtype=np.float32), np.array([], dtype=np.int64)

    X_all = np.stack(X_list, axis=0)
    return X_all, np.array(idx_end, dtype=np.int64)

# ---- Triple-Barrier + váhy ----

def _compute_returns(close: np.ndarray, idx_end: np.ndarray, horizon_sec: int) -> np.ndarray:
    """
    Jednoduchý výnos r(t) = (C[t+h]-C[t]) / C[t] pro každý idx_end.
    """
    r = np.full((len(idx_end),), np.nan, dtype=np.float32)
    T = len(close)
    for i, t in enumerate(idx_end):
        j = t + int(horizon_sec)
        if j >= T:
            continue
        c0 = float(close[t]); c1 = float(close[j])
        if c0 == 0: 
            continue
        r[i] = (c1 - c0) / c0
    return r
    
def _rolling_sigma(close: np.ndarray, win: int) -> np.ndarray:
    if win <= 1: 
        return np.full_like(close, fill_value=np.nan, dtype=np.float32)
    r = np.diff(close) / (close[:-1] + 1e-12)
    sig = pd.Series(r).rolling(win, min_periods=max(2, win//3)).std().values
    sig = np.concatenate([[np.nan], sig]).astype(np.float32)  # zarovnej na délku close
    return sig

def _triple_barrier_labels(returns: np.ndarray, up_pt: float, dn_pt: float, edge_min_bps: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Jednoduchá TB aproximace z koncového výnosu:
      UP        pokud r >= up_pt
      DOWN      pokud r <= -dn_pt
      ABSTAIN   jinak, nebo pokud |r| < edge_min (v bps)
    Vrací tuple: (y_softmax3 [0=DOWN,1=ABSTAIN,2=UP], y_updown (jen pro {UP,DOWN}), mask_updown (bool))
    """
    y3 = np.full_like(returns, fill_value=1, dtype=np.int64)  # default ABSTAIN
    mask_ud = np.zeros_like(returns, dtype=bool)
    y_ud = np.zeros_like(returns, dtype=np.int64)

    edge_min = float(edge_min_bps) / 10000.0 if edge_min_bps is not None else 0.0

    for i, r in enumerate(returns):
        if not np.isfinite(r):
            continue
        if abs(r) < edge_min:
            y3[i] = 1  # abstain
            continue
        if r >= float(up_pt):
            y3[i] = 2  # UP
            y_ud[i] = 1
            mask_ud[i] = True
        elif r <= -float(dn_pt):
            y3[i] = 0  # DOWN
            y_ud[i] = 0
            mask_ud[i] = True
        else:
            y3[i] = 1  # abstain
    return y3, y_ud, mask_ud

def _sample_weights_from_costs(returns: np.ndarray, fee_bps: float = 0.0, slip_bps: float = 0.0) -> np.ndarray:
    """
    vzorek vážíme dle "užitečnosti" = max(|r| - (fee+slip), 0)
    """
    c = (float(fee_bps or 0.0) + float(slip_bps or 0.0)) / 10000.0
    w = np.maximum(np.abs(returns) - c, 0.0)
    # stabilizace: posuň do <0.1, 1.0> (aby nuly vypadly, ale neexplodovalo to)
    w = 0.1 + (w / (w.max() + 1e-12))
    return w.astype(np.float32)

# ---- Dataset/Dataloader ----

class SeqDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, w: np.ndarray = None):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)
        self.w = (w.astype(np.float32) if w is not None else np.ones_like(self.y, dtype=np.float32))
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, i):
        return self.X[i], self.y[i], self.w[i]

# ---- trénink ----

def train_hmr_supervisor(
    X_train: np.ndarray, y_train: np.ndarray, w_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray, w_val: np.ndarray,
    in_features: int, seq_len: int,
    hidden_low: int = 64, hidden_high: int = 64, high_period: int = 10,
    use_abstain_head: bool = True,
    use_softmax3: bool = False,
    use_uncert_head: bool = False,
    num_hsel: int = 0,
    epochs: int = 4, batch_size: int = 256, lr: float = 1e-3,
    device: str = "cpu",
    class_weight_up: float = 1.0, class_weight_down: float = 1.0
) -> Tuple[HRMHead, Dict]:
    mdl = HRMHead(
        in_features=in_features,
        hidden_low=hidden_low, hidden_high=hidden_high, high_period=high_period,
        use_abstain_head=use_abstain_head,
        use_softmax3=use_softmax3,
        use_uncert_head=use_uncert_head,
        num_hsel=num_hsel
    ).to(device)

    opt = torch.optim.Adam(mdl.parameters(), lr=lr)
    bce = nn.BCEWithLogitsLoss(reduction="none")
    ce  = nn.CrossEntropyLoss(reduction="none")
    

    def _run_epoch(split: str) -> Tuple[float, float]:
        is_train = (split == "train")
        loader = DataLoader(SeqDataset(
            X_train if is_train else X_val,
            y_train if is_train else y_val,
            w_train if is_train else w_val
        ), batch_size=batch_size, shuffle=is_train, drop_last=False)

        tot_loss = 0.0; tot_n = 0
        for xb, yb, wb in loader:
            xb = xb.to(device); yb = yb.to(device); wb = wb.to(device)
            if is_train:
                mdl.train(); opt.zero_grad()
            else:
                mdl.eval()

            # Multi-head výstupy
            outs = mdl.forward_heads(xb)

            # yb: [y_ud, mask_ud, y3]
            y_ud = yb[:, 0]
            m_ud = yb[:, 1]
            y3   = yb[:, 2].long()

            # --- hlavní up/down hlava na LOGITECH ---
            # outs["updown"] teď bereme jako PRAVDĚPODOBNOST a převedeme ji na logit,
            # aby BCEWithLogitsLoss dostal správný vstup. (Pokud by model někdy vracel přímo logity,
            # můžeš sem dát přímo ty.)
            p_up = outs["updown"].clamp(1e-6, 1-1e-6)         # p \in (0,1)
            z_up = torch.logit(p_up, eps=1e-6)                # logit(p)

            if m_ud.sum() > 0:
                cw = torch.where(
                    y_ud > 0.5,
                    torch.full_like(y_ud, class_weight_up),
                    torch.full_like(y_ud, class_weight_down)
                )
                loss_ud = bce(z_up, y_ud) * m_ud * wb * cw     # <<< BCEWithLogitsLoss bere z_up
                loss_ud = loss_ud.sum() / (m_ud.sum() + 1e-12)
            else:
                loss_ud = torch.tensor(0.0, device=device)

            # --- softmax3 necháváme jak je (už používá CE na logitech) ---
            if "softmax3" in outs:
                p3 = outs["softmax3"]          # měly by být logity
                loss_s3 = ce(p3, y3) * wb
                loss_s3 = loss_s3.mean()
            else:
                loss_s3 = torch.tensor(0.0, device=device)

            # --- ABSTAIN hlava (také jako logit) ---
            if "abstain" in outs:
                y_abs = (y3 == 1).float()
                p_abs = outs["abstain"].clamp(1e-6, 1-1e-6)
                z_abs = torch.logit(p_abs, eps=1e-6)
                loss_abs = bce(z_abs, y_abs) * wb
                loss_abs = loss_abs.mean()
            else:
                loss_abs = torch.tensor(0.0, device=device)

            # --- UNCERTAINTY hlava (stejně) ---
            if "uncert" in outs:
                y_unc = (y3 == 1).float()
                p_unc = outs["uncert"].clamp(1e-6, 1-1e-6)
                z_unc = torch.logit(p_unc, eps=1e-6)
                loss_unc = bce(z_unc, y_unc) * wb
                loss_unc = loss_unc.mean()
            else:
                loss_unc = torch.tensor(0.0, device=device)

           
            # (5) hsel – beze změny
            loss_hsel = torch.tensor(0.0, device=device)
            loss = loss_ud + 0.5*loss_s3 + 0.25*loss_abs + 0.25*loss_unc + loss_hsel

            if is_train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(mdl.parameters(), max_norm=1.0)
                opt.step()

            tot_loss += float(loss.item()) * xb.size(0); tot_n += xb.size(0)

        return tot_loss / max(1, tot_n)

    best = {"val_loss": 1e9, "state": None}
    for ep in range(1, epochs+1):
        tr = _run_epoch("train")
        va = _run_epoch("val")
        print(f"[L3] epoch {ep:02d}  train={tr:.5f}  val={va:.5f}")
        if va < best["val_loss"]:
            best["val_loss"] = va
            best["state"] = {k: v.detach().cpu() for k, v in mdl.state_dict().items()}

    if best["state"] is not None:
        mdl.load_state_dict(best["state"])
    return mdl, best

# ---- meta.json thresholds helper ----

def _update_meta_thresholds(out_dir: str, thr_up: float = None, thr_down: float = None, thr_margin: float = None):
    if thr_up is None and thr_down is None and thr_margin is None:
        return
    path = os.path.join(out_dir, "meta.json")
    m = _read_json(path)
    rec = m.get("recommended") or m.get("recommended_thresholds") or {}
    if thr_up is not None:    rec["CONF_ENTER_UP"] = float(thr_up)
    if thr_down is not None:  rec["CONF_ENTER_DOWN"] = float(thr_down)
    if thr_margin is not None:rec["ABSTAIN_MARGIN"] = float(thr_margin)
    m["recommended"] = rec
    _write_json(path, m)
    print(f"[L3] meta.json thresholds updated at {path}: up={rec.get('CONF_ENTER_UP')} down={rec.get('CONF_ENTER_DOWN')} margin={rec.get('ABSTAIN_MARGIN')}")

# ---- hlavní skript ----

def main():
    ap = argparse.ArgumentParser(description="Train L3 Supervisor (HRMHead)")

    # povinné/hlavní
    ap.add_argument("--data", required=True, help="CSV/Parquet s 1s OHLCV nebo s již spočtenými FEATURE_COLS.")
    ap.add_argument("--out_dir", required=False, help="Cílový set (adresář), kam se uloží supervisor_L3.*")
    ap.add_argument("--target-set", dest="out_dir", required=False, help="Alias pro --out_dir (bez změny cesty).")

    # vstupy
    ap.add_argument("--sets", nargs="*", default=None, help="Alias pro --sets_l1 (L1/RAW vstupy).")
    ap.add_argument("--sets_l1", nargs="*", default=None, help="Seznam setů pro L1/RAW vstup.")
    ap.add_argument("--sets_l2", nargs="*", default=None, help="Seznam L2 hlav (adresáře s hrm_meta_L2.pt).")
    ap.add_argument("--l2-heads", nargs="*", dest="sets_l2", default=None, help="Alias pro --sets_l2.")

    # délky oken
    ap.add_argument("--l3_seq_len", type=int, default=60, help="Délka sekvence pro L3 okno.")
    ap.add_argument("--l3-seq-len", type=int, dest="l3_seq_len", help="Alias pro --l3_seq_len.")
    ap.add_argument("--l1-seq-len", type=int, dest="l1_seq_len_override", default=None, help="Přepiš l1_seq_len z meta.json (volitelné).")

    # použití bloků
    ap.add_argument("--use_raw_in_l3", action="store_true", help="Připojit i RAW kanály (xgb/lstm/hrm) do L3.")
    ap.add_argument("--use-l1-in-l3", action="store_true", dest="use_l1_in_l3", help="Přidat L1 blok (p_meta_L1 napříč sets_l1).")
    ap.add_argument("--use-l2-in-l3", action="store_true", dest="use_l2_in_l3", help="Přidat L2 blok (p_L2 napříč sets_l2).")
    ap.add_argument("--use-regime-in-l3", action="store_true", dest="use_regime_in_l3", help="Přidat jednoduché režimové featury.")

    # trénink parametry
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--batch", type=int, dest="batch_size", help="Alias pro --batch_size.")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden_low", type=int, default=64)
    ap.add_argument("--hidden_high", type=int, default=64)
    ap.add_argument("--hrm-hidden-low", type=int, dest="hidden_low", help="Alias pro --hidden_low.")
    ap.add_argument("--hrm-hidden-high", type=int, dest="hidden_high", help="Alias pro --hidden_high.")
    ap.add_argument("--high_period", type=int, default=10)
    ap.add_argument("--hrm-high-period", type=int, dest="high_period", help="Alias pro --high_period.")
    ap.add_argument("--valid_frac", type=float, default=0.2, help="Poslední podíl pro validaci (časový split).")

    # kalibrace + prahy
    ap.add_argument("--platt-l3", action="store_true", dest="platt_l3", help="Zapnout Platt kalibraci na validaci.")
    ap.add_argument("--iso-l3", action="store_true", dest="iso_l3", help="Zapnout isotonic kalibraci na validaci (fallback/alternativa).")
    ap.add_argument("--thr-up", type=float, dest="thr_up", default=None, help="Doporučený threshold LONG (zapíše se do meta.json).")
    ap.add_argument("--thr-down", type=float, dest="thr_down", default=None, help="Doporučený threshold SHORT (zapíše se do meta.json).")
    ap.add_argument("--thr-margin", type=float, dest="thr_margin", default=None, help="Abstain margin (zapíše se do meta.json).")

    # triple barrier + costs
    ap.add_argument("--triple-barrier", action="store_true", dest="triple_barrier", help="Použít triple-barrier značkování (UP/DOWN/ABSTAIN).")
    ap.add_argument("--up-pt", type=float, default=0.0005, help="Upper take-profit práh (relativně, např. 0.0005=5 bps).")
    ap.add_argument("--dn-pt", type=float, default=0.0005, help="Lower stop-loss práh (relativně).")
    ap.add_argument("--up-k-sigma", type=float, default=None, help="Místo fixního up-pt použij k * σ_roll (k>0).")
    ap.add_argument("--dn-k-sigma", type=float, default=None, help="Místo fixního dn-pt použij k * σ_roll (k>0).")
    ap.add_argument("--sigma-window", type=int, default=120, help="Okno pro σ (v sekundách).")
    ap.add_argument("--timeout-hz", type=int, default=None, help="Timeout horizont v sekundách (default=out_dir.horizon_sec).")
    ap.add_argument("--edge-min-bps", type=float, default=0.0, help="Min. absolutní výnos pro non-abstain (bps).")

    ap.add_argument("--fee-bps", type=float, default=0.0, help="Komise v bps pro vážení vzorků.")
    ap.add_argument("--slip-bps", type=float, default=0.0, help="Slippage v bps pro vážení vzorků.")
    ap.add_argument("--class-weight-up", type=float, default=1.0, help="BCE váha pro třídu UP.")
    ap.add_argument("--class-weight-down", type=float, default=1.0, help="BCE váha pro třídu DOWN.")

    # pomocné hlavy
    ap.add_argument("--use-uncert-head", action="store_true", dest="use_uncert_head", help="Zapnout uncertainty hlavu (kopie abstain).")
    ap.add_argument("--use-hsel-head", action="store_true", dest="use_hsel_head", help="Zapnout horizon-selector softmax (počet = len(sets_l2)).")
    ap.add_argument("--fast-sanity", action="store_true", dest="fast_sanity", help="Zrychlený běh: méně kontrol/printů.")

    args = ap.parse_args()

    if not args.out_dir:
        raise SystemExit("Udej --out_dir (nebo alias --target-set).")

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # meta (z L2 cílové sady) – kvůli l1_seq_len a horizon_sec
    meta = _read_meta_l2(out_dir)
    horizon_sec = _get_horizon_from_meta(meta, default_hz=1)
    l1_seq_len_meta  = _get_l1_seq_len_from_meta(meta, default_l1=60)
    l1_seq_len = int(args.l1_seq_len_override) if args.l1_seq_len_override else l1_seq_len_meta

    # supervisor.json fallback (pokud existuje a neudáš sets)
    sup_prev = _read_supervisor_json(out_dir)

    # sets_l1
    if args.sets_l1 is not None:
        sets_l1 = [os.path.abspath(s) for s in args.sets_l1]
    elif args.sets is not None:  # alias
        sets_l1 = [os.path.abspath(s) for s in args.sets]
    elif isinstance(sup_prev.get("sets_l1"), list) and len(sup_prev["sets_l1"])>0:
        sets_l1 = [os.path.abspath(s) for s in sup_prev["sets_l1"]]
    else:
        # fallback: meta_gate.sets z out_dir (stejné jako L2)
        sets_l1 = _get_l2_sets_from_meta(meta)

    # sets_l2 (L2 hlavy)
    if args.sets_l2 is not None:
        sets_l2 = [os.path.abspath(s) for s in args.sets_l2]
    elif isinstance(sup_prev.get("sets_l2"), list) and len(sup_prev["sets_l2"])>0:
        sets_l2 = [os.path.abspath(s) for s in sup_prev["sets_l2"]]
    else:
        sets_l2 = []  # explicitně prázdné, pokud neuvedeš

    print(f"[L3] out_dir: {out_dir}")
    print(f"[L3] sets_l1: {sets_l1}")
    print(f"[L3] sets_l2: {sets_l2}")
    print(f"[L3] use_l1_in_l3={args.use_l1_in_l3}  use_raw_in_l3={args.use_raw_in_l3}  use_l2_in_l3={args.use_l2_in_l3}  use_regime_in_l3={args.use_regime_in_l3}")
    print(f"[L3] l1_seq_len={l1_seq_len} (meta={l1_seq_len_meta}), l3_seq_len={args.l3_seq_len}, horizon_sec={horizon_sec}")

    # data
    path = os.path.abspath(args.data)
    if not os.path.exists(path):
        raise SystemExit(f"Dataset '{path}' neexistuje.")
    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    df_feat = _ensure_features(df)
    close = df_feat["close"].values.astype(np.float32) if "close" in df_feat.columns else df["close"].values.astype(np.float32)

    # --- připrav RAW a L1 série pro sets_l1 ---
    raw_by_set: Dict[str, np.ndarray] = {}
    pl1_by_set: Dict[str, np.ndarray] = {}
    if sets_l1:
        for s in sets_l1:
            raw_mat, raw_cols = _build_raw_matrix_for_set(s, df_feat)
            raw_by_set[s] = raw_mat
            pl1 = _compute_l1_series_for_set(s, raw_mat, len(df_feat), l1_seq_len)
            pl1_by_set[s] = pl1
            print(f"[L3] L1/RAW {s}: raw C={raw_mat.shape[1]}  L1 valid={np.isfinite(pl1).sum()}")

    # --- připrav plné série p_L2 pro každý head v sets_l2 ---
    p_l2_by_head: Dict[str, np.ndarray] = {}
    if sets_l2:
        for head in sets_l2:
            env = _prepare_env_for_l2_head(head, df_feat, l1_seq_len_fallback=l1_seq_len)
            p_l2 = _full_series_l2_from_env(env)
            p_l2_by_head[head] = p_l2
            print(f"[L3] L2 head {head}: valid={np.isfinite(p_l2).sum()}  L2={env['l2_seq_len']}  L1={env['l1_seq_len']}  use_raw={env['use_raw']}  S={len(env['sets'])}")

    # --- režimové featury (fit normalizace jen z trénovací části; tady uděláme z celé a info uložíme) ---
    regime_mean = None; regime_std = None
    if args.use_regime_in_l3:
        # přibližně spočteme průměr a std z celé řady
        regs = []
        for t in range(len(df_feat)):
            rvec = _compute_regime_vector(df_feat, t)
            regs.append(rvec.reshape(1, -1))
        R = np.concatenate(regs, axis=0).astype(np.float32) if regs else None
        if R is not None and R.shape[1] > 0:
            regime_mean = R.mean(axis=0)
            regime_std  = R.std(axis=0) + 1e-6  # proti dělení nulou

    # --- sestav L3 okna + indexy konců oken ---
    X_all, idx_end = _build_l3_windows(
        pl1_by_set, raw_by_set, p_l2_by_head,
        use_raw_in_l3=bool(args.use_raw_in_l3),
        use_l1_in_l3=bool(args.use_l1_in_l3),
        use_l2_in_l3=bool(args.use_l2_in_l3),
        l3_seq_len=int(args.l3_seq_len),
        df_feat=df_feat if args.use_regime_in_l3 else None,
        use_regime_in_l3=bool(args.use_regime_in_l3),
        regime_mean=regime_mean, regime_std=regime_std
    )
    if X_all.shape[0] == 0:
        raise SystemExit("Nepodařilo se sestavit žádné L3 okno (zkontroluj data a dostupnost kanálů).")
        
    # --- diagnostika surových L3 oken ---
    try:
        n_nan = int(np.isnan(X_all).sum())
        n_inf = int(np.isinf(X_all).sum())
        print(f"[L3][dbg] X_all windows: shape={X_all.shape}  NaN={n_nan}  Inf={n_inf}")
    except Exception:
        pass

    # --- labely + váhy ---
    rets = _compute_returns(close, idx_end, horizon_sec=int(args.timeout_hz or horizon_sec))
    if args.triple_barrier:
        # vol-scaled TB? -> přepočti up/dn prahy per-index
        up_pt = float(args.up_pt)
        dn_pt = float(args.dn_pt)
        if args.up_k_sigma or args.dn_k_sigma:
            sig = _rolling_sigma(close, int(args.sigma_window))
            # vyrob individuální prahy na idx_end, kde je sigma definovaná
            # mapování t -> idx v close: idx_end je "t"; σ[t] použijeme přímo
            k_up = float(args.up_k_sigma or 0.0)
            k_dn = float(args.dn_k_sigma or 0.0)
            # fallback: kde σ chybí, použij fixní up_pt/dn_pt
            up_arr = np.where(np.isfinite(sig[idx_end]), np.abs(k_up*sig[idx_end]), up_pt)
            dn_arr = np.where(np.isfinite(sig[idx_end]), np.abs(k_dn*sig[idx_end]), dn_pt)
            # spočítej y3/y_ud/m_ud element-wise
            y3 = np.full_like(rets, 1, dtype=np.int64); y_ud = np.zeros_like(rets, dtype=np.int64)
            m_ud = np.zeros_like(rets, dtype=bool)
            edge_min = float(args.edge_min_bps or 0.0)/10000.0
            for i, r in enumerate(rets):
                if not np.isfinite(r): 
                    continue
                if abs(r) < edge_min: 
                    y3[i]=1; continue
                upi = up_arr[i]; dni = dn_arr[i]
                if r >= upi: 
                    y3[i]=2; y_ud[i]=1; m_ud[i]=True
                elif r <= -dni:
                    y3[i]=0; y_ud[i]=0; m_ud[i]=True
                else:
                    y3[i]=1
        else:
            y3, y_ud, m_ud = _triple_barrier_labels(rets, up_pt=up_pt, dn_pt=dn_pt, edge_min_bps=float(args.edge_min_bps or 0.0))
            # slož cíl matici: [y_ud, mask_ud, y3]
            y_all = np.stack([y_ud.astype(np.float32), m_ud.astype(np.float32), y3.astype(np.float32)], axis=1)

    else:
        # klasika: UP pokud r>0, DOWN jinak; mask=1 všude; y3 jen pro formu
        y_ud = (rets > 0).astype(np.float32)
        m_ud = np.isfinite(rets).astype(np.float32)
        y3 = (1 + (rets > 0).astype(np.int64))  # 2=UP, 1=ABSTAIN(never), 0=DOWN -> dáme 2/0 a ABSTAIN nepoužijeme
        y_all = np.stack([y_ud, m_ud, y3.astype(np.float32)], axis=1)

    # --- finální maska PO okenizaci: pouze plně finita okna/labely/vratnosti ---
    maskX = np.isfinite(X_all).all(axis=(1,2))
    maskY = np.isfinite(y_all).all(axis=1)
    maskR = np.isfinite(rets)
    ok = maskX & maskY & maskR
    if ok.sum() == 0:
        raise SystemExit("Po odfiltrování finity nezbyl žádný sample. Zkontroluj zdrojové kanály.")
        
        
    X_all = X_all[ok]; y_all = y_all[ok]; idx_end = idx_end[ok]; rets = rets[ok]
    w_all = _sample_weights_from_costs(rets, fee_bps=float(args.fee_bps or 0.0), slip_bps=float(args.slip_bps or 0.0))

    if not args.fast_sanity:
        print(f"[L3][dbg] X_all windows: shape={X_all.shape}  NaN={np.isnan(X_all).sum()}  Inf={np.isinf(X_all).sum()}")
    print(f"[L3] dataset: X={X_all.shape}  y={y_all.shape}  (po odfiltrování)")
    # --- poslední sanity: any NaN/Inf -> 0 (už by neměly být, ale pro jistotu) ---
    X_all = np.nan_to_num(X_all, nan=0.0, posinf=0.0, neginf=0.0)
    y_all = np.nan_to_num(y_all, nan=0.0, posinf=0.0, neginf=0.0)
    

    # --- časový split (poslední valid_frac jako validace) ---
    N = X_all.shape[0]
    n_val = max(1, int(round(N * float(args.valid_frac))))
    n_tr = N - n_val
    X_tr, y_tr, w_tr = X_all[:n_tr], y_all[:n_tr], w_all[:n_tr]
    X_va, y_va, w_va = X_all[n_tr:], y_all[n_tr:], w_all[n_tr:]
    print(f"[L3] split: train={X_tr.shape[0]}  valid={X_va.shape[0]}")

    # --- model + trénink ---
    in_features = X_all.shape[2]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_hsel = len(sets_l2) if args.use_hsel_head else 0
    mdl, best = train_hmr_supervisor(
        X_tr, y_tr, w_tr, X_va, y_va, w_va,
        in_features=in_features, seq_len=int(args.l3_seq_len),
        hidden_low=int(args.hidden_low), hidden_high=int(args.hidden_high), high_period=int(args.high_period),
        use_abstain_head=True,
        use_softmax3=bool(args.triple_barrier),
        use_uncert_head=bool(args.use_uncert_head),
        num_hsel=int(num_hsel),
        epochs=int(args.epochs), batch_size=int(args.batch_size), lr=float(args.lr),
        device=device,
        class_weight_up=float(args.class_weight_up), class_weight_down=float(args.class_weight_down)
    )

    # --- eval + kalibrace (Platt na validaci) ---
    mdl.eval()
    with torch.no_grad():
        p_va_up = _safe_numpy(mdl(torch.from_numpy(X_va).to(device))).reshape(-1)

    def _acc_updown(p_up, y):
        # y[:,0]=y_ud, y[:,1]=mask_ud
        m = y[:,1] > 0.5
        if m.sum() == 0: return np.nan
        pred = (p_up[m] >= 0.5).astype(int)
        return (pred == y[m,0].astype(int)).mean()

    print(f"[L3] valid up/down acc={_acc_updown(p_va_up, y_va):.4f}")

    calib = None
    if args.platt_l3:
        try:
            from sklearn.linear_model import LogisticRegression
            logit_va = _sigmoid_to_logit(p_va_up).reshape(-1,1)
            lr = LogisticRegression(max_iter=1000)
            # jen na validních updown vzorcích
            m = y_va[:,1] > 0.5
            lr.fit(logit_va[m], y_va[m,0].astype(int))
            calib = lr
            joblib.dump(calib, os.path.join(out_dir, "supervisor_L3.calib.pkl"))
            print("[L3] Platt kalibrace uložena: supervisor_L3.calib.pkl")
        except Exception as e:
            print(f"[L3] Kalibraci se nepodařilo uložit: {e}")
    else:
        print("[L3] Platt kalibrace vypnuta (nezadán --platt-l3)")

    # Isotonic fallback / alternativa
    if args.iso_l3:
        try:
            from sklearn.isotonic import IsotonicRegression
            m = y_va[:,1] > 0.5
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(p_va_up[m], y_va[m,0].astype(int))
            joblib.dump(iso, os.path.join(out_dir, "supervisor_L3.iso.pkl"))
            print("[L3] Isotonic kalibrace uložena: supervisor_L3.iso.pkl")
        except Exception as e:
            print(f"[L3] Isotonic selhala: {e}")

    # --- ulož checkpoint ---
    ckpt = {
        "state_dict": {k: v.cpu() for k, v in mdl.state_dict().items()},
        "in_features": in_features,
        "hidden_low": int(args.hidden_low),
        "hidden_high": int(args.hidden_high),
        "high_period": int(args.high_period),
        "seq_len": int(args.l3_seq_len),
        "use_abstain_head": True,
        "use_softmax3": bool(args.triple_barrier),
        "use_uncert_head": bool(args.use_uncert_head),
        "num_hsel": int(num_hsel),
        "use_regime_in_l3": bool(args.use_regime_in_l3),
        "regime_mean": (regime_mean.tolist() if regime_mean is not None else None),
        "regime_std":  (regime_std.tolist() if regime_std is not None else None),
    }
    torch.save(ckpt, os.path.join(out_dir, "supervisor_L3.pt"))
    print(f"[L3] uložen: {os.path.join(out_dir, 'supervisor_L3.pt')}")

    # --- ulož supervisor.json ---
    sup_json = {
        "seq_len": int(args.l3_seq_len),
        "use_raw_in_l3": bool(args.use_raw_in_l3),
        "use_l1_in_l3":  bool(args.use_l1_in_l3),
        "use_l2_in_l3":  bool(args.use_l2_in_l3),
        "use_regime_in_l3": bool(args.use_regime_in_l3),
        "sets_l1": sets_l1,
        "sets_l2": sets_l2,
        "triple_barrier": bool(args.triple_barrier),
        "tb_params": {
            "up_pt": float(args.up_pt),
            "dn_pt": float(args.dn_pt),
            "timeout_hz": int(args.timeout_hz or horizon_sec),
            "edge_min_bps": float(args.edge_min_bps or 0.0)
        },
        "costs_bps": {
            "fee": float(args.fee_bps or 0.0),
            "slip": float(args.slip_bps or 0.0)
        },
        "comment": "Auto-generated by train_supervisor.py (L3 supervisor)"
    }
    if args.thr_up is not None or args.thr_down is not None or args.thr_margin is not None:
        sup_json["recommended"] = {}
        if args.thr_up is not None: sup_json["recommended"]["CONF_ENTER_UP"] = float(args.thr_up)
        if args.thr_down is not None: sup_json["recommended"]["CONF_ENTER_DOWN"] = float(args.thr_down)
        if args.thr_margin is not None: sup_json["recommended"]["ABSTAIN_MARGIN"] = float(args.thr_margin)

    _write_json(os.path.join(out_dir, "supervisor.json"), sup_json)
    print(f"[L3] uložen: {os.path.join(out_dir, 'supervisor.json')}")

    # --- (volitelně) update / auto-thresholds ---
    if args.thr_up is None or args.thr_down is None:
        # spočti auto práhy tak, aby maximalizovaly expected net edge na validaci
        fee = float(args.fee_bps or 0.0)/10000.0
        slip= float(args.slip_bps or 0.0)/10000.0
        cost = fee + slip
        m = y_va[:,1] > 0.5
        if m.sum()>100:
            rets_va = rets[n_tr:][m]  # edge na valid up/down oknech
            p_va = p_va_up[m]
            best = {"up":0.58,"dn":0.42,"edge":-1e9}
            # hrubá grid-search
            ups = np.linspace(0.52, 0.70, 19)
            dns = np.linspace(0.30, 0.48, 19)
            for up in ups:
                for dn in dns:
                    mask_buy  = p_va >= up
                    mask_sell = p_va <= dn
                    mask = mask_buy | mask_sell
                    if mask.mean() < 0.05: 
                        continue
                    # očekávaný netto edge po nákladech (ber znaménko podle směru)
                    r_sel = rets_va[mask]
                    sgn   = np.where(mask_buy[mask], 1.0, -1.0)
                    net   = (sgn * r_sel) - cost
                    exp_edge = np.mean(net)
                    if exp_edge > best["edge"]:
                        best = {"up":float(up), "dn":float(dn), "edge":float(exp_edge)}
            print(f"[L3] AUTO-THR valid net-edge best: up={best['up']:.3f} down={best['dn']:.3f} edge={best['edge']:.6f}")
            _update_meta_thresholds(out_dir, best["up"], best["dn"], args.thr_margin)
        else:
            _update_meta_thresholds(out_dir, args.thr_up, args.thr_down, args.thr_margin)
    else:
        _update_meta_thresholds(out_dir, args.thr_up, args.thr_down, args.thr_margin)
    
if __name__ == "__main__":
    main()