# app/models/train_meta.py
# -*- coding: utf-8 -*-
import os
import time
import json
import argparse
from typing import Tuple, List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score

from .hrm_model import HRMHead
from .ensemble import EnsemblePredictor
from ..utils.features import compute_features, FEATURE_COLS

BASE_DIR = os.path.dirname(__file__)
DEFAULT_WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
os.makedirs(DEFAULT_WEIGHTS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

def _ensure_outdir(outdir: str | None) -> str:
    if not outdir:
        return DEFAULT_WEIGHTS_DIR
    outdir = os.path.abspath(outdir)
    os.makedirs(outdir, exist_ok=True)
    return outdir

# ---------- Cache helpers ----------
def _parquet_available() -> bool:
    try:
        import pyarrow  # noqa
        return True
    except Exception:
        try:
            import fastparquet  # noqa
            return True
        except Exception:
            return False

def features_cache_path(symbol: str) -> str:
    base = os.path.join(DATA_DIR, f"{symbol.upper()}_features")
    return base + (".parquet" if _parquet_available() else ".csv")

def seconds_cache_path(symbol: str) -> str:
    base = os.path.join(DATA_DIR, f"{symbol.upper()}_1s")
    return base + (".parquet" if _parquet_available() else ".csv")

def load_seconds(symbol: str) -> pd.DataFrame:
    p = seconds_cache_path(symbol)
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    if p.endswith(".parquet"):
        df = pd.read_parquet(p)
    else:
        df = pd.read_csv(p)
    if "ts" in df.columns:
        df = df.set_index("ts")
    return df.sort_index()

def _save_features(symbol: str, dff: pd.DataFrame):
    pf = features_cache_path(symbol)
    if pf.endswith(".parquet"):
        dff.to_parquet(pf)
    else:
        dff.to_csv(pf, index=False)
    print(f"[CACHE] Saved features → {pf} ({len(dff)} rows)")

def load_or_build_features(symbol: str, horizon: int) -> pd.DataFrame:
    pf = features_cache_path(symbol)
    need_recompute = True
    if os.path.exists(pf):
        try:
            if pf.endswith(".parquet"):
                dff = pd.read_parquet(pf)
            else:
                dff = pd.read_csv(pf)
            cols_ok = all(c in dff.columns for c in FEATURE_COLS + ["y"])
            if cols_ok:
                need_recompute = False
                return dff.reset_index(drop=True)
        except Exception:
            need_recompute = True

    df1s = load_seconds(symbol)
    dff = compute_features(df1s, horizon=horizon)
    _save_features(symbol, dff)
    return dff.reset_index(drop=True)

# ---------- META dataset ----------
def build_meta_sequences(
    df_feat: pd.DataFrame,
    seq_len: int,
    need_hrm: bool,
) -> Tuple[np.ndarray, np.ndarray, int]:
    ens = EnsemblePredictor()  # načte AKTUÁLNÍ sadu z default/ENV
    core = ens.core

    # očekávané délky
    lstm_seq = 30
    try:
        ck = torch.load(os.path.join(core.WEIGHTS_DIR, "lstm.pt"), map_location="cpu")
        lstm_seq = int(ck.get("seq_len", lstm_seq))
    except Exception:
        pass

    hrm_seq = 60
    hrm_ready = core.hrm is not None
    if hrm_ready:
        try:
            ck = torch.load(os.path.join(core.WEIGHTS_DIR, "hrm.pt"), map_location="cpu")
            hrm_seq = int(ck.get("seq_len", hrm_seq))
        except Exception:
            pass

    xgb_ready = core.xgb is not None
    if need_hrm and not hrm_ready:
        raise SystemExit("META need_hrm=True, ale hrm.pt není k dispozici. Nejdřív natrénuj HRM v train_offline.")

    Xf = df_feat[FEATURE_COLS].values.astype(np.float32)
    y_all = df_feat["y"].values.astype(np.int64)
    n = len(df_feat)

    px = np.full(n, np.nan, dtype=np.float32)
    pl = np.full(n, np.nan, dtype=np.float32)
    ph = np.full(n, np.nan, dtype=np.float32)

    # XGB
    if xgb_ready:
        try:
            raw = core.xgb.predict_proba(Xf)[:, 1]
            if core.xgb_cal is not None:
                px = core.xgb_cal.transform(raw.reshape(-1))
            else:
                px = raw.astype(np.float32)
        except Exception:
            pass

    # LSTM
    if core.lstm is not None:
        L = lstm_seq
        with torch.no_grad():
            for i in range(L - 1, n):
                xs = torch.from_numpy(Xf[i - L + 1 : i + 1]).unsqueeze(0)
                try:
                    p = float(core.lstm(xs).cpu().numpy().reshape(-1)[0])
                    if core.lstm_cal is not None:
                        eps = 1e-6
                        pp = np.clip(p, eps, 1 - eps)
                        logit = np.log(pp / (1 - pp)).reshape(-1, 1)
                        p = float(core.lstm_cal.predict_proba(logit)[:, 1][0])
                    pl[i] = p
                except Exception:
                    continue

    # HRM
    if hrm_ready:
        H = hrm_seq
        with torch.no_grad():
            for i in range(H - 1, n):
                xs = torch.from_numpy(Xf[i - H + 1 : i + 1]).unsqueeze(0)
                try:
                    ph[i] = float(core.hrm.infer(xs).cpu().numpy().reshape(-1)[0])
                except Exception:
                    continue

    channels: List[str] = []
    if xgb_ready: channels.append("xgb")
    if core.lstm is not None: channels.append("lstm")
    if need_hrm:
        if hrm_ready: channels.append("hrm")
        else: raise SystemExit("META need_hrm=True, ale HRM není dostupné.")
    else:
        if hrm_ready: channels.append("hrm")

    C = len(channels)
    if C < 2:
        raise SystemExit("META: potřeba aspoň 2 kanály (např. XGB+LSTM).")

    stack = []
    for ch in channels:
        if ch == "xgb": stack.append(px)
        elif ch == "lstm": stack.append(pl)
        elif ch == "hrm": stack.append(ph)
    P = np.stack(stack, axis=1)  # (n, C)

    warm = seq_len - 1
    if "lstm" in channels:
        warm = max(warm, lstm_seq - 1)
    if "hrm" in channels:
        warm = max(warm, hrm_seq - 1)

    X_seq, y_seq = [], []
    for i in range(warm, n):
        window = P[i - seq_len + 1 : i + 1, :]
        if np.any(np.isnan(window)):
            continue
        X_seq.append(window.astype(np.float32))
        y_seq.append(y_all[i])

    X_seq = np.stack(X_seq, axis=0) if X_seq else np.empty((0, seq_len, C), dtype=np.float32)
    y_seq = np.array(y_seq, dtype=np.int64) if y_seq else np.empty((0,), dtype=np.int64)
    return X_seq, y_seq, C

def time_split(X: np.ndarray, y: np.ndarray, valid_frac=0.2):
    n = len(X)
    if n == 0:
        return X, X, y, y
    k = max(int(n * (1.0 - valid_frac)), 1)
    Xtr, Xte = X[:k], X[k:]
    ytr, yte = y[:k], y[k:]
    return Xtr, Xte, ytr, yte

def train_meta(
    X: np.ndarray, y: np.ndarray,
    epochs: int = 40,
    high_period: int = 10,
    hidden_low: int = 32,
    hidden_high: int = 32,
    batch: int = 256,
    outdir: str = DEFAULT_WEIGHTS_DIR,
) -> tuple[HRMHead, float]:
    device = torch.device("cpu")
    seq_len = X.shape[1]
    in_features = X.shape[2]

    Xtr, Xte, ytr, yte = time_split(X, y, valid_frac=0.2)
    if len(Xtr) == 0 or len(Xte) == 0:
        raise SystemExit("META: not enough data after time split.")

    model = HRMHead(
        in_features=in_features,
        hidden_low=hidden_low,
        hidden_high=hidden_high,
        high_period=high_period
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    dl_tr = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr)), batch_size=batch, shuffle=False)
    dl_te = DataLoader(TensorDataset(torch.from_numpy(Xte), torch.from_numpy(yte)), batch_size=batch, shuffle=False)

    best_auc = -1.0
    best_state = None

    for ep in range(1, epochs + 1):
        model.train()
        for xb, yb in dl_tr:
            xb = xb.to(device); yb = yb.float().to(device)
            opt.zero_grad()
            p = model(xb)
            loss = torch.nn.functional.binary_cross_entropy(p, yb)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            preds, ys = [], []
            for xb, yb in dl_te:
                xb = xb.to(device)
                preds.append(model.infer(xb).cpu().numpy())
                ys.append(yb.numpy())
            pp = np.concatenate(preds).reshape(-1)
            yy = np.concatenate(ys).reshape(-1)
            try:
                auc = roc_auc_score(yy, pp)
            except ValueError:
                auc = 0.5

        print(f"[META] epoch {ep}: AUC={auc:.4f}")
        if auc > best_auc:
            best_auc = auc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            torch.save({
                "state_dict": best_state,
                "seq_len": int(seq_len),
                "in_features": int(in_features),
                "hidden_low": int(hidden_low),
                "hidden_high": int(hidden_high),
                "high_period": int(high_period),
            }, os.path.join(outdir, "hrm_meta.pt"))
            print("[META] ✓ checkpoint saved")

    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save({
        "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "seq_len": int(seq_len),
        "in_features": int(in_features),
        "hidden_low": int(hidden_low),
        "hidden_high": int(hidden_high),
        "high_period": int(high_period),
    }, os.path.join(outdir, "hrm_meta.pt"))
    print("[META] ✓ final saved")

    return model, best_auc

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="ETHUSDT")
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--horizon", type=int, default=30)
    ap.add_argument("--meta-seq", type=int, default=60)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--need-hrm", action="store_true", help="Vyžadovat HRM kanál (XGB+LSTM+HRM). Bez něj trénink skončí.")
    ap.add_argument("--high-period", type=int, default=10)
    ap.add_argument("--hidden-low", type=int, default=32)
    ap.add_argument("--hidden-high", type=int, default=32)
    ap.add_argument("--outdir", type=str, default=None, help="Kam uložit váhy (např. app/models/weights/30s)")
    args = ap.parse_args()

    outdir = _ensure_outdir(args.outdir)

    dff = load_or_build_features(args.symbol, horizon=int(args.horizon))

    X, y, C = build_meta_sequences(
        dff,
        seq_len=int(args.meta_seq),
        need_hrm=bool(args.need_hrm),
    )
    print(f"[META] dataset={len(y)} seq={int(args.meta_seq)} in_features={C} need_hrm={bool(args.need_hrm)}")

    if len(y) == 0:
        raise SystemExit("META: prázdná data — zkontroluj checkpointy XGB/LSTM (/HRM) pro daný horizont.")

    model, auc = train_meta(
        X, y,
        epochs=int(args.epochs),
        high_period=max(5, int(args.high_period)),
        hidden_low=int(args.hidden_low),
        hidden_high=int(args.hidden_high),
        batch=256,
        outdir=outdir,
    )

    # meta.json v outdir – jen aktualizujeme horizon
    meta_path = os.path.join(outdir, "meta.json")
    meta = {}
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
        except Exception:
            meta = {}

    meta["horizon_sec"] = int(args.horizon)
    meta["trained_at"] = int(time.time())
    with open(meta_path, "w") as f:
        json.dump(meta, f)
    print(f"[META] Saved → {meta_path} {meta}")

if __name__ == "__main__":
    main()
