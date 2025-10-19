# diag_l3_bce.py
# -*- coding: utf-8 -*-
"""
Diagnostika L3 (HRM Supervisor) pro chybu:
RuntimeError: all elements of input should be between 0 and 1 (BCELoss)

Skript:
- Postaví stejné L3 vstupy jako train_supervisor (RAW+L1+L2).
- Vytáhne první batch z train splitu.
- Projede výstupy forward_heads() a ukáže, kde jsou hodnoty mimo <0,1>, NaN/Inf, atd.
- Zkontroluje také y/masky/softmax3.

Nastavení uprav v HLAVIČCE.
"""

import os, json, numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader

# ======================= HLAVIČKA (UPRAV) =======================

DATA_PATH = "app/data/train_1s.parquet"

OUT_DIR   = "app/models/weights/10s"  # cílový set (kvůli meta.json: horizon, l1_seq_len fallback)

SETS_L1 = [
    "app/models/weights/1s",
    "app/models/weights/5s",
    "app/models/weights/10s",
    "app/models/weights/15s",
    "app/models/weights/30s",
    "app/models/weights/60s",
    "app/models/weights/180s",
]
SETS_L2 = [
    "app/models/weights/1s",
    "app/models/weights/10s",
    "app/models/weights/15s",
    "app/models/weights/30s",
    "app/models/weights/60s",
    "app/models/weights/180s",
]

USE_RAW_IN_L3 = True
USE_L1_IN_L3  = True
USE_L2_IN_L3  = True

L1_SEQ_LEN = 60
L3_SEQ_LEN = 60

# Triple-barrier pro labely (musí ladit s tvým L3 horizonem)
TB_UP_PT       = 0.0005
TB_DN_PT       = 0.0005
TB_EDGE_MIN_BPS= 0.0

VALID_FRAC = 0.20
BATCH_SIZE = 256

# ===============================================================

# Importujeme stejné utility a třídy z train_supervisor
from app.models.train_supervisor import (
    _ensure_features, _read_meta_l2, _get_horizon_from_meta,
    _build_raw_matrix_for_set, _compute_l1_series_for_set,
    _prepare_env_for_l2_head, _full_series_l2_from_env,
    _build_l3_windows, _compute_returns, _triple_barrier_labels,
    _sample_weights_from_costs, HRMHead, SeqDataset
)

def _stats(name: str, tens: torch.Tensor):
    arr = tens.detach().cpu().numpy()
    bad_lt0 = int(np.sum(arr < 0))
    bad_gt1 = int(np.sum(arr > 1))
    n_nan   = int(np.sum(~np.isfinite(arr)))
    amin = float(np.nanmin(arr)) if arr.size else float("nan")
    amax = float(np.nanmax(arr)) if arr.size else float("nan")
    print(f"[{name}] shape={tuple(arr.shape)}  min={amin:.6f} max={amax:.6f}  <0:{bad_lt0}  >1:{bad_gt1}  nan/inf:{n_nan}")

def main():
    # ---------- data + meta ----------
    df = pd.read_parquet(DATA_PATH)
    df_feat = _ensure_features(df)

    out_dir_abs = os.path.abspath(OUT_DIR)
    meta = _read_meta_l2(out_dir_abs)
    horizon_sec = _get_horizon_from_meta(meta, default_hz=10)
    print(f"[META] out_dir={out_dir_abs}  horizon_sec={horizon_sec}")

    # ---------- L1/RAW ----------
    raw_by_set = {}
    pl1_by_set = {}
    for s in SETS_L1:
        s_abs = os.path.abspath(s)
        raw_mat, raw_cols = _build_raw_matrix_for_set(s_abs, df_feat)
        pl1 = _compute_l1_series_for_set(s_abs, raw_mat, len(df_feat), int(L1_SEQ_LEN))
        raw_by_set[s_abs] = raw_mat
        pl1_by_set[s_abs] = pl1
        print(f"[L1/RAW] {s_abs}: raw C={raw_mat.shape[1]}  L1_valid={int(np.isfinite(pl1).sum())}")

    # ---------- L2 heads ----------
    p_l2_by_head = {}
    for head in SETS_L2:
        head_abs = os.path.abspath(head)
        env = _prepare_env_for_l2_head(head_abs, df_feat, l1_seq_len_fallback=int(L1_SEQ_LEN))
        p_l2 = _full_series_l2_from_env(env)
        p_l2_by_head[head_abs] = p_l2
        print(f"[L2] {head_abs}: valid={int(np.isfinite(p_l2).sum())}  L2={env['l2_seq_len']}  L1={env['l1_seq_len']}  use_raw={env['use_raw']}  S={len(env['sets'])}")

    # ---------- L3 okna ----------
    X_all, idx_end = _build_l3_windows(
        pl1_by_set, raw_by_set, p_l2_by_head,
        use_raw_in_l3=bool(USE_RAW_IN_L3),
        use_l1_in_l3=bool(USE_L1_IN_L3),
        use_l2_in_l3=bool(USE_L2_IN_L3),
        l3_seq_len=int(L3_SEQ_LEN),
        df_feat=None, use_regime_in_l3=False
    )
    print(f"[BUILD] X_all={X_all.shape}, idx_end={len(idx_end)}")
    if X_all.shape[0] == 0:
        print("[ERR] Nepodařilo se sestavit žádné L3 okno.")
        return

    # ---------- labely + váhy (triple barrier) ----------
    close = (df_feat["close"].values.astype(np.float32)
             if "close" in df_feat.columns else df["close"].values.astype(np.float32))
    rets = _compute_returns(close, idx_end, horizon_sec=int(horizon_sec))
    y3, y_ud, m_ud = _triple_barrier_labels(
        rets, up_pt=float(TB_UP_PT), dn_pt=float(TB_DN_PT), edge_min_bps=float(TB_EDGE_MIN_BPS)
    )
    y_all = np.stack([y_ud.astype(np.float32), m_ud.astype(np.float32), y3.astype(np.float32)], axis=1)

    ok = np.isfinite(y_all).all(axis=1)
    X_all = X_all[ok]; y_all = y_all[ok]; rets = rets[ok]
    w_all = _sample_weights_from_costs(rets, fee_bps=0.0, slip_bps=0.0)
    print(f"[DATASET] X={X_all.shape} y={y_all.shape}  valid_rows={X_all.shape[0]}")

    # ---------- split ----------
    N = X_all.shape[0]
    n_val = max(1, int(round(N * float(VALID_FRAC))))
    n_tr = N - n_val
    X_tr, y_tr, w_tr = X_all[:n_tr], y_all[:n_tr], w_all[:n_tr]
    print(f"[SPLIT] train={len(X_tr)}  valid={len(X_all)-len(X_tr)}")

    # ---------- model (stejné parametry jako trénink) ----------
    in_features = X_all.shape[2]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mdl = HRMHead(
        in_features=int(in_features),
        hidden_low=96, hidden_high=96, high_period=12,
        use_abstain_head=True,
        use_softmax3=True,
        use_uncert_head=False,
        num_hsel=len(SETS_L2)
    ).to(device)
    mdl.train()

    # ---------- první batch ----------
    ds = SeqDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr), torch.from_numpy(w_tr))
    dl = DataLoader(ds, batch_size=int(BATCH_SIZE), shuffle=False, drop_last=False)
    xb, yb, wb = next(iter(dl))
    xb = xb.to(device); yb = yb.to(device); wb = wb.to(device)

    outs = mdl.forward_heads(xb)
    print("=== HEADS:", list(outs.keys()))

    # ---- up/down hlava
    z_up = outs["updown"].squeeze(-1)  # tvůj HRMHead vrací 1D (N,) nebo (N,1) – sjednotíme
    _stats("updown_raw", z_up)

    # Pravděpodobnost přes sigmoid + clamp (čistě diagnosticky)
    p_up = torch.sigmoid(z_up).clamp(1e-6, 1-1e-6)
    _stats("updown_prob(sigmoid)", p_up)

    # najdi problematické indexy
    arr_raw = z_up.detach().cpu().numpy()
    arr_prob = p_up.detach().cpu().numpy()
    bad_raw = np.where(~np.isfinite(arr_raw))[0]
    bad_prob = np.where((arr_prob < 0) | (arr_prob > 1) | (~np.isfinite(arr_prob)))[0]
    if bad_raw.size:
        print("[BAD updown_raw idx]", bad_raw[:10].tolist())
    if bad_prob.size:
        print("[BAD updown_prob idx]", bad_prob[:10].tolist())

    # ---- softmax3 (pokud je)
    if "softmax3" in outs:
        z3 = outs["softmax3"]
        _stats("softmax3_logits", z3)
        p3 = torch.softmax(z3, dim=1)
        _stats("softmax3_softmax", p3)
        row_sums = p3.detach().cpu().sum(dim=1).numpy()
        print(f"[softmax3] row_sums ~1  min={row_sums.min():.6f}  max={row_sums.max():.6f}")

    # ---- abstain/uncert (pokud jsou)
    for k in ("abstain", "uncert"):
        if k in outs:
            z = outs[k].squeeze(-1)
            _stats(f"{k}_raw", z)
            _stats(f"{k}_prob(sigmoid)", torch.sigmoid(z).clamp(1e-6, 1-1e-6))

    # ---- cíle a masky
    y_ud = yb[:, 0]
    m_ud = yb[:, 1]
    y3_t = yb[:, 2].long()

    y_ud_u = torch.unique(y_ud).detach().cpu().tolist()
    m_ud_u = torch.unique(m_ud).detach().cpu().tolist()
    y3_u   = torch.unique(y3_t).detach().cpu().tolist()

    print(f"[y_ud] min={float(y_ud.min()):.1f} max={float(y_ud.max()):.1f} unique={y_ud_u}")
    print(f"[mask_ud] ones={int((m_ud>0.5).sum())}/{len(m_ud)}  unique={m_ud_u}")
    print(f"[y3] unique={y3_u}  (0=DOWN,1=ABSTAIN,2=UP)")

    # Ověření tvarů pro loss
    print(f"[SHAPES] p_up={tuple(p_up.shape)}  y_ud={tuple(y_ud.shape)}  m_ud={tuple(m_ud.shape)} wb={tuple(wb.shape)}")

    print("=== DIAGNOSTIKA DOKONČENA ===")

if __name__ == "__main__":
    main()