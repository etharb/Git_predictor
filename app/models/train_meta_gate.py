# app/models/train_meta_gate.py
# -*- coding: utf-8 -*-
"""
Dvouvrstvá meta kombinace:
  L1 (per-horizont):  HRMHead nad [p_xgb, p_lstm?, p_hrm?] -> p_meta_L1
  L2 (napříč h):      HRMHead nad [p_meta_L1(h1), p_meta_L1(h2), ...] (+ volitelně raw kanály) -> p_final
"""

import os, json, argparse, time
from typing import Dict, List, Optional, Tuple

import numpy as np
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, matthews_corrcoef

from .hrm_model import HRMHead  # stejné jako u dosavadní mety

# ------------------------------ Pomocné I/O ---------------------------------

def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def _load_npz_must(path: str) -> Dict[str, np.ndarray]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path} (očekávám oof_preds.npz)")
    data = np.load(path, allow_pickle=False)
    out = {k: data[k] for k in data.files}
    return out

def _read_meta_json(path_dir: str) -> Dict:
    p = os.path.join(path_dir, "meta.json")
    if not os.path.exists(p):
        return {}
    try:
        with open(p, "r") as f:
            return json.load(f)
    except Exception:
        return {}

def _write_meta_json(path_dir: str, meta: Dict):
    p = os.path.join(path_dir, "meta.json")
    try:
        with open(p, "w") as f:
            json.dump(meta, f)
        print(f"[META] Saved → {p} {meta}")
    except Exception as e:
        print(f"[META] failed to write {p}: {e}")

# ----------------------------- Utility: numerika -----------------------------

def _sanitize_probs(a: np.ndarray) -> np.ndarray:
    """NaN/Inf→0.5 a clamp do (1e-6, 1-1e-6)."""
    a = np.asarray(a, dtype=np.float64)
    a = np.nan_to_num(a, nan=0.5, posinf=1.0, neginf=0.0)
    a = np.clip(a, 1e-6, 1.0 - 1e-6)
    return a.astype(np.float32)

def _safe_auc(y: np.ndarray, p: np.ndarray) -> float:
    """AUC, který nepadá na NaN a jediné třídě. Vrací 0.5 jako fallback."""
    y = np.asarray(y).reshape(-1)
    p = np.asarray(p).reshape(-1)
    m = np.isfinite(p)
    if not m.any():
        return 0.5
    y2 = y[m]; p2 = p[m]
    if np.unique(y2).size < 2:
        return 0.5
    try:
        return float(roc_auc_score(y2, p2))
    except Exception:
        return 0.5

def _bce_loss_auto(out: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Pokud out není v [0,1] (nebo obsahuje nefin. hodnoty), beru ho jako logity
    a použiju BCEWithLogits. Jinak použiju BCE nad pravděpodobnostmi, ale
    pro jistotu clampnu do (1e-6, 1-1e-6) a ošetřím NaN/Inf.
    """
    if not torch.isfinite(out).all():
        out = torch.nan_to_num(out, nan=0.0, posinf=10.0, neginf=-10.0)

    mn = float(out.min())
    mx = float(out.max())

    if (mn < -1e-6) or (mx > 1.0 + 1e-6):
        return F.binary_cross_entropy_with_logits(out, target)

    out_prob = out.clamp(1e-6, 1.0 - 1e-6)
    return F.binary_cross_entropy(out_prob, target)

# ----------------------------- Utility: thresholds ---------------------------

def auto_thresholds(prob: np.ndarray, y: np.ndarray,
                    max_margin: float = 0.2,
                    min_action_coverage: float = 0.10,
                    min_abstain_coverage: float = 0.0,
                    min_margin: float = 0.0) -> Tuple[float,float,float,float,float]:
    """
    Najde margin m v [0, max_margin] s omezeními a max. MCC.
    Vrací: up, down, m, best_mcc, action_coverage
    """
    prob = _sanitize_probs(prob)
    y = np.asarray(y).astype(np.int64)
    best = {"m":0.0, "mcc":-2.0, "cov":0.0}
    for m in np.linspace(0.0, max_margin, 81):
        if m < min_margin:
            continue
        up = 0.5 + m; dn = 0.5 - m
        pred = np.full_like(y, -1)
        pred[prob >= up] = 1
        pred[prob <= dn] = 0
        mask = (pred != -1)
        cov_act = float(mask.mean()) if mask.size else 0.0
        cov_abs = 1.0 - cov_act
        if cov_act < min_action_coverage:
            continue
        if cov_abs < min_abstain_coverage:
            continue
        mcc = matthews_corrcoef(y[mask], pred[mask]) if mask.any() else -2.0
        if mcc > best["mcc"]:
            best = {"m":float(m), "mcc":float(mcc), "cov":float(cov_act)}
    up = 0.5 + best["m"]; dn = 0.5 - best["m"]
    return float(up), float(dn), float(best["m"]), float(best["mcc"]), float(best["cov"])

def _quantiles(a: np.ndarray) -> List[float]:
    return np.quantile(_sanitize_probs(a), [0.01,0.05,0.25,0.5,0.75,0.95,0.99]).round(4).tolist()

def _report_channel(name: str, raw: np.ndarray, cal: Optional[np.ndarray] = None):
    print(f"[REPORT] {name}:")
    print(f"  raw  q01/05/25/50/75/95/99: {_quantiles(raw)}")
    if cal is not None:
        print(f"  cal  q01/05/25/50/75/95/99: {_quantiles(cal)}")

# ----------------------------- Sekvence pravděpodobností --------------------

def make_prob_sequences(pmat: np.ndarray, y: np.ndarray, seq_len: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    pmat: (T, C)  y: (T,)  ->  Xs: (T-seq_len+1, seq_len, C), ys: (T-seq_len+1,)
    """
    assert pmat.ndim == 2
    T, C = pmat.shape
    if T < seq_len:
        return None, None
    Xs = np.stack([pmat[t-seq_len+1:t+1, :] for t in range(seq_len-1, T)], axis=0).astype(np.float32)
    ys = y[seq_len-1:].astype(np.int64)
    return Xs, ys

# ----------------------------- L1 trén / infer -------------------------------

def train_or_load_L1(set_dir: str,
                     y: np.ndarray,
                     channels: List[np.ndarray],
                     seq_len: int,
                     epochs: int,
                     batch: int,
                     high_period: int = 10,
                     hidden_low: int = 48,
                     hidden_high: int = 48,
                     retrain: bool = False) -> Tuple[np.ndarray, Dict]:
    """
    Vstup: y (T,), kanály [p_xgb, p_lstm?, p_hrm?] (každý (T,))
    Výstup:
      - p_meta_L1 (T-seq_len+1,)  — L1 pravděpodobnost zarovnaná k posledním časům
      - info dict { 'model_path':..., 'retrained':bool, 'in_features':C }
    """
    model_path = os.path.join(set_dir, "hrm_meta.pt")
    C = sum(1 for c in channels if c is not None)
    col_list = [ _sanitize_probs(c).reshape(-1,1) for c in channels if c is not None ]
    pmat = np.concatenate(col_list, axis=1)  # (T, C)

    # pokus: reuse existujícího modelu, pokud in_features sedí a retrain=False
    reuse = False
    if (not retrain) and os.path.exists(model_path):
        try:
            ckpt = torch.load(model_path, map_location="cpu")
            in_features = int(ckpt.get("in_features", -1))
            if in_features == C:
                reuse = True
        except Exception:
            reuse = False

    Xs, ys = make_prob_sequences(pmat, y, seq_len=seq_len)
    if Xs is None:
        raise RuntimeError(f"[L1] Not enough data for sequences in {set_dir} (T={len(y)}, seq_len={seq_len}).")

    device = torch.device("cpu")
    if reuse:
        # jen inference
        ckpt = torch.load(model_path, map_location="cpu")
        model = HRMHead(in_features=C,
                        hidden_low=int(ckpt.get("hidden_low", hidden_low)),
                        hidden_high=int(ckpt.get("hidden_high", hidden_high)),
                        high_period=int(ckpt.get("high_period", high_period))).to(device)
        # bezpečné nahrání
        sd = {k: (torch.as_tensor(v) if isinstance(v, np.ndarray) else v) for k, v in ckpt["state_dict"].items()}
        model.load_state_dict(sd, strict=False)
        model.eval()
        with torch.no_grad():
            dl = DataLoader(TensorDataset(torch.from_numpy(Xs), torch.from_numpy(ys)), batch_size=batch, shuffle=False)
            preds = []
            for xb, _ in dl:
                xb = xb.to(device)
                out = model.infer(xb)
                preds.append(out.detach().cpu().numpy())
            p_meta = _sanitize_probs(np.concatenate(preds, axis=0))
        print(f"[L1] Reused {model_path} (in_features={C}) → p_meta_L1 len={len(p_meta)}")
        return p_meta, {"model_path": model_path, "retrained": False, "in_features": C}

    # trénink L1
    model = HRMHead(in_features=C,
                    hidden_low=hidden_low,
                    hidden_high=hidden_high,
                    high_period=high_period).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    dl = DataLoader(TensorDataset(torch.from_numpy(Xs), torch.from_numpy(ys)), batch_size=batch, shuffle=True)

    best_auc, best_state = -1.0, None
    for ep in range(1, epochs+1):
        model.train()
        for xb, yb in dl:
            xb = xb.to(device); yb = yb.float().to(device)
            opt.zero_grad()
            out = model(xb)
            loss = _bce_loss_auto(out, yb)
            loss.backward(); opt.step()

        # valid = train (OOF by mělo být vyrobeno dřív; tady optimalizujeme jen přibližně)
        model.eval()
        with torch.no_grad():
            preds, yy = [], []
            for xb, yb in dl:
                xb = xb.to(device)
                out = model.infer(xb)
                preds.append(out.detach().cpu().numpy()); yy.append(yb.numpy())
            pp = _sanitize_probs(np.concatenate(preds)); yyy = np.concatenate(yy)
            auc = _safe_auc(yyy, pp)
        print(f"[L1] {os.path.basename(set_dir)} epoch {ep}: AUC={auc:.4f}")
        if auc > best_auc:
            best_auc = auc
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            torch.save({
                "state_dict": best_state,
                "in_features": C,
                "seq_len": seq_len,
                "hidden_low": int(hidden_low),
                "hidden_high": int(hidden_high),
                "high_period": int(high_period),
            }, model_path)
            print(f"[L1] ✓ checkpoint saved → {model_path}")

    if best_state is not None:
        model.load_state_dict(best_state, strict=False)

    # finální inference (na Xs)
    model.eval()
    with torch.no_grad():
        dl_eval = DataLoader(TensorDataset(torch.from_numpy(Xs), torch.from_numpy(ys)), batch_size=batch, shuffle=False)
        preds = []
        for xb, _ in dl_eval:
            xb = xb.to(device)
            out = model.infer(xb)
            preds.append(out.detach().cpu().numpy())
        p_meta = _sanitize_probs(np.concatenate(preds, axis=0))

    print(f"[L1] Trained {model_path} (in_features={C}) → p_meta_L1 len={len(p_meta)}")
    return p_meta, {"model_path": model_path, "retrained": True, "in_features": C}

# ----------------------------- L2 trén --------------------------------------

def train_L2_gate(inputs: np.ndarray, y: np.ndarray,
                  seq_len: int, epochs: int, batch: int,
                  high_period: int = 10, hidden_low: int = 64, hidden_high: int = 64,
                  out_path: str = "") -> Tuple[np.ndarray, Dict]:
    """
    inputs: (T, C_total)  – sloučené kanály (např. p_meta_L1 z více sad) [+ volitelně raw kanály]
    y:      (T,)          – label pro cílový horizont
    → naučí HRMHead, vrátí predikce (T-seq_len+1,) a uloží checkpoint do out_path (pokud zadaný).
    """
    Xs, ys = make_prob_sequences(inputs, y, seq_len=seq_len)
    if Xs is None:
        raise RuntimeError(f"[L2] Not enough data (T={len(y)}, seq_len={seq_len}).")
    C = inputs.shape[1]
    device = torch.device("cpu")
    model = HRMHead(in_features=C,
                    hidden_low=hidden_low,
                    hidden_high=hidden_high,
                    high_period=high_period).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    dl = DataLoader(TensorDataset(torch.from_numpy(Xs), torch.from_numpy(ys)), batch_size=batch, shuffle=True)

    best_auc, best_state = -1.0, None
    for ep in range(1, epochs+1):
        model.train()
        for xb, yb in dl:
            xb = xb.to(device); yb = yb.float().to(device)
            opt.zero_grad()
            out = model(xb)
            loss = _bce_loss_auto(out, yb)
            loss.backward(); opt.step()

        # again „train-as-valid“ (OOF doporučeno v upstream kroku)
        model.eval()
        with torch.no_grad():
            preds, yy = [], []
            for xb, yb in dl:
                xb = xb.to(device)
                out = model.infer(xb)
                preds.append(out.detach().cpu().numpy()); yy.append(yb.numpy())
            pp = _sanitize_probs(np.concatenate(preds)); yyy = np.concatenate(yy)
            auc = _safe_auc(yyy, pp)
        print(f"[L2] epoch {ep}: AUC={auc:.4f}")
        if auc > best_auc:
            best_auc = auc
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            if out_path:
                torch.save({
                    "state_dict": best_state,
                    "in_features": C,
                    "seq_len": seq_len,
                    "hidden_low": hidden_low,
                    "hidden_high": hidden_high,
                    "high_period": int(high_period),
                }, out_path)
                print(f"[L2] ✓ checkpoint saved → {out_path}")

    if best_state is not None:
        model.load_state_dict(best_state, strict=False)

    # finální inference (na Xs)
    model.eval()
    with torch.no_grad():
        dl_eval = DataLoader(TensorDataset(torch.from_numpy(Xs), torch.from_numpy(ys)), batch_size=batch, shuffle=False)
        preds = []
        for xb, _ in dl_eval:
            xb = xb.to(device)
            out = model.infer(xb)
            preds.append(out.detach().cpu().numpy())
        p_final = _sanitize_probs(np.concatenate(preds, axis=0))

    if out_path:
        torch.save({
            "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "in_features": C,
            "seq_len": seq_len,
            "hidden_low": hidden_low,
            "hidden_high": hidden_high,
            "high_period": int(high_period),
        }, out_path)
        print(f"[L2] ✓ final saved → {out_path}")

    return p_final, {"model_path": out_path, "in_features": C, "best_auc": float(best_auc)}

# ----------------------------- Platt kalibrace -------------------------------

def platt_fit_and_apply(p_valid: np.ndarray, y_valid: np.ndarray, save_path: Optional[str] = None):
    # Bezpečný fallback: když je ve vzorcích jen jedna třída, vrať identitu
    if p_valid is None or y_valid is None or len(p_valid) == 0 or len(y_valid) == 0:
        return lambda p: p
    if np.unique(y_valid).size < 2:
        print("[CAL] L2 Platt skipped (only one class in y).")
        return lambda p: p
    eps = 1e-6
    p_clip = np.clip(np.nan_to_num(p_valid, nan=0.5, posinf=1.0, neginf=0.0), eps, 1-eps)
    logit = np.log(p_clip/(1-p_clip)).reshape(-1,1)
    try:
        lr = LogisticRegression(max_iter=1000)
        lr.fit(logit, y_valid)
    except Exception as e:
        print(f"[CAL] L2 Platt failed: {e} — using identity.")
        return lambda p: p
    if save_path:
        joblib.dump(lr, save_path)
        print(f"[CAL] Platt saved → {save_path}")
    def _cal(p):
        pc = np.clip(np.nan_to_num(p, nan=0.5, posinf=1.0, neginf=0.0), eps, 1-eps)
        return lr.predict_proba(np.log(pc/(1-pc)).reshape(-1,1))[:,1]
    return _cal

# ----------------------------- Runtime MLP gate ------------------------------

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

def train_runtime_mlp_gate(X: np.ndarray, y: np.ndarray, hidden: int = 128, epochs: int = 20, batch: int = 512) -> Tuple[MetaMLP, float]:
    """
    X: (T, C_comp) — vektory složené z kanálů (p_xgb[/p_lstm/p_hrm] napříč sadami) bez sekvencí
    y: (T,)        — label cílové sady zarovnaný na X
    """
    device = torch.device("cpu")
    model = MetaMLP(in_dim=X.shape[1], hidden=int(hidden), drop=0.0).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    dl = DataLoader(TensorDataset(torch.from_numpy(X.astype(np.float32)), torch.from_numpy(y.astype(np.float32))), batch_size=batch, shuffle=True)

    best_auc, best_state = -1.0, None
    for ep in range(1, epochs+1):
        model.train()
        for xb, yb in dl:
            xb = xb.to(device); yb = yb.to(device)
            opt.zero_grad()
            p = model(xb)
            loss = _bce_loss_auto(p, yb)
            loss.backward(); opt.step()
        # „train-as-valid“
        model.eval()
        with torch.no_grad():
            preds, yy = [], []
            for xb, yb in dl:
                xb = xb.to(device)
                preds.append(model(xb).cpu().numpy().reshape(-1))
                yy.append(yb.cpu().numpy().reshape(-1))
            pp = _sanitize_probs(np.concatenate(preds)); yyy = np.concatenate(yy)
            auc = _safe_auc(yyy, pp)
        print(f"[MLP-GATE] epoch {ep}: AUC={auc:.4f}")
        if auc > best_auc:
            best_auc = auc
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state, strict=False)
    return model, float(best_auc)

def build_components_list(set_dirs: List[str]) -> List[Dict]:
    comps = []
    for d in set_dirs:
        base = os.path.basename(d.rstrip("/"))
        if os.path.exists(os.path.join(d, "xgb.model")):
            comps.append({"key": f"xgb@{base}", "type": "xgb", "path": d})
        if os.path.exists(os.path.join(d, "lstm.pt")):
            comps.append({"key": f"lstm@{base}", "type": "lstm", "path": d})
        if os.path.exists(os.path.join(d, "hrm.pt")):
            comps.append({"key": f"hrm@{base}", "type": "hrm", "path": d})
    return comps

def stack_component_matrix(loaded: Dict[str, Dict[str, np.ndarray]], set_dirs: List[str]) -> Tuple[np.ndarray, List[str]]:
    T_min = min(len(loaded[d]["y"]) for d in set_dirs)
    cols, names = [], []
    for d in set_dirs:
        base = os.path.basename(d.rstrip("/"))
        for name in ["p_xgb", "p_lstm", "p_hrm"]:
            v = loaded[d].get(name, None)
            if v is None:
                continue
            cols.append(_sanitize_probs(v[-T_min:]).reshape(-1,1))
            names.append(f"{name}@{base}")
    if not cols:
        raise RuntimeError("Nenalezeny žádné komponenty pro MLP gate (žádný p_xgb/p_lstm/p_hrm).")
    X = np.concatenate(cols, axis=1)
    return X, names

# ----------------------------- Načtení jedné sady ---------------------------

def load_set_oof(set_dir: str) -> Dict[str, np.ndarray]:
    npz = _load_npz_must(os.path.join(set_dir, "oof_preds.npz"))
    need = ["y", "p_xgb"]
    for k in need:
        if k not in npz:
            raise KeyError(f"{set_dir}/oof_preds.npz missing key: {k}")
    out = {"y": np.asarray(npz["y"]).astype(np.int64),
           "p_xgb": np.asarray(npz["p_xgb"]).astype(np.float32)}
    for opt in ["p_lstm", "p_hrm"]:
        if opt in npz:
            out[opt] = np.asarray(npz[opt]).astype(np.float32)
        else:
            out[opt] = None
    return out

# ----------------------------- Hlavní pipeline -------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-set", required=True, help="Cílová sada (kam uložit L2 + meta.json).")
    ap.add_argument("--sets", nargs="+", required=True, help="Seznam sad (včetně cílové), např. 15s 30s 60s (cesty).")
    ap.add_argument("--l1-seq-len", type=int, default=60)
    ap.add_argument("--l2-seq-len", type=int, default=60)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=256)

    ap.add_argument("--hrm-high-period", type=int, default=10)
    ap.add_argument("--hrm-hidden-low", type=int, default=64)
    ap.add_argument("--hrm-hidden-high", type=int, default=64)

    ap.add_argument("--retrain-l1", action="store_true", help="Vynutit přeučení L1 i pokud hrm_meta.pt existuje.")
    ap.add_argument("--use-raw-in-l2", action="store_true", help="Do L2 přidat i raw kanály (xgb/lstm/hrm) všech sad.")

    # threshold constraints
    ap.add_argument("--thr-min-margin", type=float, default=0.0)
    ap.add_argument("--thr-min-act", type=float, default=0.10)
    ap.add_argument("--thr-min-abstain", type=float, default=0.0)

    ap.add_argument("--platt-l2", action="store_true", help="Platt kalibrace L2 před výpočtem prahů.")

    # runtime MLP gate pro EnsemblePredictor (volitelné)
    ap.add_argument("--emit-mlp-gate", action="store_true", help="Vytrénuje a uloží meta_gate.pt + meta_gate.json pro online gating.")
    ap.add_argument("--mlp-hidden", type=int, default=128)

    args = ap.parse_args()

    target_dir = os.path.abspath(args.target_set)
    set_dirs = [os.path.abspath(s) for s in args.sets]
    if target_dir not in set_dirs:
        print("[WARN] target-set není v --sets; přidávám ho na konec.")
        set_dirs.append(target_dir)

    for d in set_dirs:
        if not os.path.isdir(d):
            raise NotADirectoryError(f"Set dir not found: {d}")
        if not os.path.exists(os.path.join(d, "oof_preds.npz")):
            raise FileNotFoundError(f"{d}/oof_preds.npz nenalezeno (nejprve vyrob OOF v train_offline.py).")

    # --- Načti OOF pro všechny sady ---
    loaded = {d: load_set_oof(d) for d in set_dirs}
    lens = [len(loaded[d]["y"]) for d in set_dirs]
    T_min = min(lens)
    for d in set_dirs:
        for k, v in loaded[d].items():
            if v is None: continue
            loaded[d][k] = v[-T_min:]

    # --- L1 pro každou sadu: reuse nebo trén ---
    l1_out = {}
    l1_info = {}
    for d in set_dirs:
        chans = [loaded[d]["p_xgb"], loaded[d]["p_lstm"], loaded[d]["p_hrm"]]
        p_meta, info = train_or_load_L1(
            set_dir=d,
            y=loaded[d]["y"],
            channels=chans,
            seq_len=args.l1_seq_len,
            epochs=args.epochs,
            batch=args.batch,
            high_period=args.hrm_high_period,
            hidden_low=args.hrm_hidden_low,
            hidden_high=args.hrm_hidden_high,
            retrain=args.retrain_l1
        )
        l1_out[d] = p_meta
        l1_info[d] = info

    # --- Složení vstupu pro L2 ---
    l1_lens = [len(l1_out[d]) for d in set_dirs]
    TL1_min = min(l1_lens)
    l1_stack = [l1_out[d][-TL1_min:].reshape(-1,1) for d in set_dirs]
    l2_inputs = np.concatenate(l1_stack, axis=1)  # (TL1_min, n_sets)

    # label pro L2
    y_target = loaded[target_dir]["y"][-TL1_min:]

    # volitelně přidat raw kanály
    if args.use_raw_in_l2:
        raw_cols = []
        for d in set_dirs:
            for name in ["p_xgb", "p_lstm", "p_hrm"]:
                v = loaded[d].get(name, None)
                if v is None: 
                    continue
                raw_cols.append(_sanitize_probs(v[-(TL1_min + args.l1_seq_len - 1):]).reshape(-1,1))
        if raw_cols:
            raw_mat = np.concatenate(raw_cols, axis=1)  # (TL1_min + l1_seq_len - 1, C_raw)
            raw_mat_tail = raw_mat[-TL1_min:, :]
            l2_inputs = np.concatenate([l2_inputs, raw_mat_tail], axis=1)

    print(f"[L2] inputs shape: {l2_inputs.shape}  (T={l2_inputs.shape[0]}, C={l2_inputs.shape[1]})")

    # --- Trén L2 (sekvenční HRM) ---
    l2_path = os.path.join(target_dir, "hrm_meta_L2.pt")
    p_l2, info_l2 = train_L2_gate(
        inputs=l2_inputs,
        y=y_target,
        seq_len=args.l2_seq_len,
        epochs=args.epochs,
        batch=args.batch,
        high_period=args.hrm_high_period,
        hidden_low=args.hrm_hidden_low,
        hidden_high=args.hrm_hidden_high,
        out_path=l2_path
    )

    # report L2
    print(f"[REPORT] class-balance target (y=1 ratio): {float(np.mean(y_target)):.3f}")
    _report_channel("L2 raw", p_l2, None)

    # --- Platt (volitelně) + thresholds ---
    p_for_thr = p_l2
    if args.platt_l2:
        cal = platt_fit_and_apply(p_l2, y_target[-len(p_l2):], save_path=os.path.join(target_dir, "meta_L2.calib.pkl"))
        p_cal = cal(p_l2)
        _report_channel("L2 cal", p_l2, p_cal)
        p_for_thr = p_cal

    up, dn, m, mcc, cov = auto_thresholds(
        p_for_thr, y_target[-len(p_for_thr):],
        max_margin=0.2,
        min_action_coverage=float(args.thr_min_act),
        min_abstain_coverage=float(args.thr_min_abstain),
        min_margin=float(args.thr_min_margin)
    )
    print(f"[AUTO-THR L2] up={up:.3f} down={dn:.3f} margin={m:.3f} MCC={mcc:.4f} coverage={cov:.2f}")

    # --- Update meta.json v cílové sadě ---
    meta = _read_meta_json(target_dir)
    if not isinstance(meta, dict):
        meta = {}
    meta["trained_at"] = int(time.time())
    meta["recommended"] = {
        "CONF_ENTER_UP": float(up),
        "CONF_ENTER_DOWN": float(dn),
        "ABSTAIN_MARGIN": float(m)
    }
    meta["meta_gate"] = {
        "l1_seq_len": int(args.l1_seq_len),
        "l2_seq_len": int(args.l2_seq_len),
        "use_raw_in_l2": bool(args.use_raw_in_l2),
        "platt_l2": bool(args.platt_l2),
        "sets": set_dirs,
        "info_l2": info_l2
    }
    _write_meta_json(target_dir, meta)

    # --- (VOLITELNĚ) Runtime MLP gate pro EnsemblePredictor ---
    if args.emit_mlp_gate:
        X_comp, comp_names = stack_component_matrix(loaded, set_dirs)
        y_comp = loaded[target_dir]["y"][-X_comp.shape[0]:]

        print(f"[MLP-GATE] X shape: {X_comp.shape}, y shape: {y_comp.shape}")
        mlp, auc_mlp = train_runtime_mlp_gate(X_comp, y_comp, hidden=int(args.mlp_hidden),
                                              epochs=max(12, args.epochs//2), batch=max(256, args.batch))

        ckpt = {
            "state_dict": {k: v.detach().cpu() for k, v in mlp.state_dict().items()},
            "in_dim": int(X_comp.shape[1]),
            "regime_cols": [],
            "regime_mean": [],
            "regime_std": [],
        }
        torch.save(ckpt, os.path.join(target_dir, "meta_gate.pt"))
        print(f"[MLP-GATE] saved → {os.path.join(target_dir, 'meta_gate.pt')} (AUC={auc_mlp:.4f})")

        comps = build_components_list(set_dirs)
        gate_json = {
            "components": comps,
            "note": "Generated by train_meta_gate.py --emit-mlp-gate. Inputs = probabilities of listed components in this order."
        }
        with open(os.path.join(target_dir, "meta_gate.json"), "w") as f:
            json.dump(gate_json, f)
        print(f"[MLP-GATE] saved → {os.path.join(target_dir, 'meta_gate.json')} (components={len(comps)})")

    print("[DONE] L2 gate hotovo.")

if __name__ == "__main__":
    main()