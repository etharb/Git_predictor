# -*- coding: utf-8 -*-
"""
Offline training pro 1s featury s volitelným horizontem (--horizon N).

Modely:
- XGB (tabular) + kalibrace (Platt)
- LSTM (sekvenční) + Platt kalibrace (+ volitelně focal loss)
- HRM base (sekvenční, "hierarchical") nad featurami (volitelně)
- META HRM nad pravděpodobnostmi (volitelně)

Novinky (kroky 1–3):
1) Dead-zone a váhy vzorků (podle absolutní budoucí návratnosti) + focal loss pro sekvenční modely
2) Beze změny features.py – dead-zone váhy se počítají v tomto skriptu (z raw df)
3) OOF (časový K-fold) pro XGB/LSTM/HRM je volitelný přes --oof-splits; Platt kalibrace na OOF; rychlý report
   Auto-thr podporuje min_margin a min_abstain_coverage
   ULOŽENÍ OOF: outdir/oof_preds.npz (y, p_xgb, [p_lstm], [p_hrm]) – zarovnané délky

Poznámka: Držíme původní API a chování, nic nepřejmenováváme.
"""
import argparse, time, os, json
import numpy as np, pandas as pd, requests, joblib, xgboost as xgb, torch
from sklearn.metrics import roc_auc_score, matthews_corrcoef
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import torch.nn as nn

from .hrm_model import HRMHead
from .lstm_model import SmallLSTM
from ..utils.features import compute_features, FEATURE_COLS, make_lstm_sequences

BASE_DIR = os.path.dirname(__file__)
DEFAULT_WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
os.makedirs(DEFAULT_WEIGHTS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# ---------------------- utils ----------------------
def _ensure_outdir(outdir: str | None) -> str:
    if not outdir:
        return DEFAULT_WEIGHTS_DIR
    outdir = os.path.abspath(outdir)
    os.makedirs(outdir, exist_ok=True)
    return outdir

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

def _cache_base(symbol: str, suffix: str) -> str:
    # suffix např. "1s" nebo "features"
    return os.path.join(DATA_DIR, f"{symbol.upper()}_{suffix}")

def _cache_paths_for_base(base: str) -> tuple[str, str]:
    return base + ".parquet", base + ".csv"

def _resolve_cache_path(base: str) -> str:
    """
    Vrátí existující cestu (parquet/csv), nebo preferovanou (parquet když je dostupný), jinak csv.
    """
    p_parq, p_csv = _cache_paths_for_base(base)
    if os.path.exists(p_parq):
        return p_parq
    if os.path.exists(p_csv):
        return p_csv
    # nic neexistuje -> preferuj parquet pokud je k dispozici
    return p_parq if _parquet_available() else p_csv

def cache_path_1s(symbol: str) -> str:
    base = _cache_base(symbol, "1s")
    return _resolve_cache_path(base)

def cache_path_features(symbol: str) -> str:
    base = _cache_base(symbol, "features")
    return _resolve_cache_path(base)

def load_cache_1s(symbol: str) -> pd.DataFrame:
    path = cache_path_1s(symbol)
    if not os.path.exists(path):
        # fallback: když se něco pokazilo, zkus druhou příponu
        base = _cache_base(symbol, "1s")
        p_parq, p_csv = _cache_paths_for_base(base)
        alt = p_csv if path.endswith(".parquet") else p_parq
        if not os.path.exists(alt):
            raise FileNotFoundError(path)
        path = alt

    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
        if "ts" in df.columns:
            df = df.set_index("ts")
    return df.sort_index()

def save_cache_1s(symbol: str, df: pd.DataFrame):
    # Ukládej na stejnou příponu, jaká už existuje; jinak preferuj parquet
    base = _cache_base(symbol, "1s")
    path = _resolve_cache_path(base)
    df = df.sort_index()
    if path.endswith(".parquet"):
        df.to_parquet(path)
    else:
        df.reset_index().to_csv(path, index=False)
    print(f"[CACHE] Saved {symbol} seconds → {path} ({len(df)} rows)")

def save_features(symbol: str, dff: pd.DataFrame):
    base = _cache_base(symbol, "features")
    path = _resolve_cache_path(base)
    if path.endswith(".parquet"):
        dff.to_parquet(path)
    else:
        dff.to_csv(path, index=False)
    print(f"[CACHE] Saved features → {path} ({len(dff)} rows)")
# ---------------------- Download & aggregate ----------------------
BINANCE_AGG = "https://api.binance.com/api/v3/aggTrades"

def _fetch_agg_trades_window(symbol: str, start_ms: int, end_ms: int, session: requests.Session, page_limit: int = 1000):
    all_rows = []
    params = {"symbol": symbol, "startTime": start_ms, "endTime": end_ms, "limit": page_limit}
    last_id = None
    while True:
        if last_id is not None:
            params = {"symbol": symbol, "fromId": last_id + 1, "limit": page_limit}
        r = session.get(BINANCE_AGG, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        if not data:
            break
        all_rows.extend(data)
        last_id = data[-1]["a"]
        last_ts = data[-1]["T"]
        if last_ts >= end_ms:
            break
        time.sleep(0.02)
    return all_rows

def fetch_agg_trades_resumable(symbol='ETHUSDT', days=1, resume=False) -> pd.DataFrame:
    have = None
    if resume:
        try:
            have = load_cache_1s(symbol)
            print(f"[CACHE] Loaded cached {symbol} seconds: {have.index.min()}..{have.index.max()} ({len(have)} rows)")
        except FileNotFoundError:
            have = None

    end_ms = int(time.time() * 1000)
    start_ms = end_ms - days * 24 * 3600 * 1000
    if have is not None:
        start_ms = max(start_ms, (int(have.index.max()) + 1) * 1000)

    if have is not None and start_ms >= end_ms:
        print("[CACHE] Up-to-date, no REST download needed.")
        return have

    ses = requests.Session()
    recs = []
    win = 60 * 60 * 1000  # 1 hod
    cur = start_ms
    last_print = time.time()
    base_window = max(1, end_ms - start_ms)

    while cur < end_ms:
        chunk_end = min(cur + win, end_ms)
        data = _fetch_agg_trades_window(symbol, cur, chunk_end, ses, page_limit=1000)
        if data:
            ts = np.fromiter((d['T'] // 1000 for d in data), dtype=np.int64)
            price = np.fromiter((float(d['p']) for d in data), dtype=np.float64)
            qty = np.fromiter((float(d['q']) for d in data), dtype=np.float64)
            df = pd.DataFrame({'ts': ts, 'price': price, 'qty': qty})
            g = df.groupby('ts', sort=True)
            out = g.agg(open=('price','first'),
                        high=('price','max'),
                        low=('price','min'),
                        close=('price','last'),
                        volume=('qty','sum')).reset_index()
            out['buy_vol']  = out['volume'] * 0.5
            out['sell_vol'] = out['volume'] * 0.5
            recs.append(out)
        cur = chunk_end + 1

        if time.time() - last_print > 2:
            done = (chunk_end - start_ms) / base_window
            print(f"[REST] {symbol} {done*100:.1f}% … {chunk_end}")
            last_print = time.time()

        time.sleep(0.02)

    if recs:
        new = pd.concat(recs, ignore_index=True).sort_values('ts').set_index('ts')
        merged = pd.concat([have, new]) if have is not None else new
    else:
        merged = have if have is not None else pd.DataFrame(
            columns=['open','high','low','close','volume','buy_vol','sell_vol']
        ).set_index(pd.Index([], name='ts'))

    merged = merged[~merged.index.duplicated(keep='last')].sort_index()
    save_cache_1s(symbol, merged)
    return merged

# ---------------------- labels & weights ----------------------
def compute_future_return(df_ohlc: pd.DataFrame, horizon_sec: int) -> pd.Series:
    """
    Vrátí budoucí návratnost r_h(t) = (close(t+h)-close(t))/close(t) pro index ts (sekundy).
    Pokud chybí t+h, vrací NaN.
    """
    if df_ohlc is None or len(df_ohlc) == 0:
        return pd.Series(dtype=float)
    close = df_ohlc['close']
    fut = close.reindex(close.index + horizon_sec)
    r = (fut.values - close.values) / np.where(close.values != 0, close.values, 1.0)
    out = pd.Series(r, index=close.index)
    return out

def deadzone_weights_from_returns(r: pd.Series, eps: float, w_dead: float, w_full: float = 1.0) -> pd.Series:
    """
    Váha  w = w_dead  pro |r| < eps, jinak w_full. Na chybějící r → w_dead (konzervativně).
    """
    w = np.where(np.abs(r.values) < eps, w_dead, w_full)
    w[np.isnan(r.values)] = w_dead
    return pd.Series(w, index=r.index, dtype=float)

# ---------------------- time split / kfold ----------------------
def time_split_indices(n: int, valid_frac: float = 0.2, purge: int = 0):
    split = int(n * (1.0 - valid_frac))
    train_end = max(0, split - purge)
    return (0, train_end), (split, n)

def ts_kfold_windows(n: int, n_splits: int, purge: int = 0):
    """
    Časové K-fold „walk-forward“: fold k má valid okno [cut_k, cut_{k+1}), train=[0, cut_k - purge).
    """
    assert n_splits >= 2
    cuts = [int(n * i / n_splits) for i in range(n_splits)] + [n]
    for k in range(n_splits):
        j0, j1 = cuts[k], cuts[k+1]
        i0, i1 = 0, max(0, j0 - purge)
        yield (i0, i1), (j0, j1)

# ---------------------- auto thresholds ----------------------
def auto_thresholds(prob: np.ndarray, y: np.ndarray,
                    max_margin: float = 0.2,
                    min_action_coverage: float = 0.10,
                    min_abstain_coverage: float = 0.0,
                    min_margin: float = 0.0):
    """
    Hledá margin m v [0, max_margin] tak, aby:
      - akční pokrytí (p>=0.5+m || p<=0.5-m) ≥ min_action_coverage
      - abstain pokrytí (jinak) ≥ min_abstain_coverage
      - m ≥ min_margin
    a maximalizuje MCC.
    """
    best = {"m":0.0, "mcc":-2.0, "cov":0.0}
    for m in np.linspace(0.0, max_margin, 81):
        if m < min_margin:
            continue
        up = 0.5 + m; dn = 0.5 - m
        pred = np.full_like(y, -1)
        pred[prob >= up] = 1
        pred[prob <= dn] = 0
        mask = (pred != -1)
        cov_act = mask.mean()
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

def _quant_report(name, p_raw, p_cal=None):
    def q(a):
        a = np.asarray(a)
        return np.quantile(a, [0.01,0.05,0.25,0.5,0.75,0.95,0.99]).round(4).tolist()
    print(f"[REPORT] {name}:")
    print("         raw quantiles  1/5/25/50/75/95/99%:", q(p_raw))
    if p_cal is not None:
        print("         cal quantiles  1/5/25/50/75/95/99%:", q(p_cal))

# ---------------------- focal loss (sigmoid outputs) ----------------------
def focal_bce_prob(p: torch.Tensor, y: torch.Tensor, gamma: float = 1.5, eps: float = 1e-6, weight: torch.Tensor | None = None):
    """
    p = sigmoid výstup (pravděpodobnost), y ∈ {0,1}
    """
    p = torch.clamp(p, eps, 1 - eps)
    pt = torch.where(y > 0.5, p, 1.0 - p)
    loss = - (1 - pt).pow(gamma) * torch.log(pt)
    if weight is not None:
        loss = loss * weight
    return loss.mean()

# ---------------------- modely (time-split) ----------------------
def train_xgb_timesplit(df_feat: pd.DataFrame, outdir: str, valid_frac=0.2, sample_weights: pd.Series | None = None):
    X = df_feat[FEATURE_COLS].values.astype(np.float32)
    y = df_feat['y'].values.astype(np.int64)
    w = sample_weights.reindex(df_feat.index).values.astype(np.float32) if sample_weights is not None else None
    n = len(df_feat)
    (i0, i1), (j0, j1) = time_split_indices(n, valid_frac=valid_frac, purge=0)
    Xtr, ytr = X[i0:i1], y[i0:i1]
    Xva, yva = X[j0:j1], y[j0:j1]
    wtr = w[i0:i1] if w is not None else None
    clf = xgb.XGBClassifier(
        n_estimators=600, max_depth=6, learning_rate=0.03,
        subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
        tree_method='hist', random_state=42, n_jobs=4
    )
    clf.fit(Xtr, ytr, sample_weight=wtr)
    pva = clf.predict_proba(Xva)[:, 1]
    auc = roc_auc_score(yva, pva)
    print(f"[XGB] (time-split) AUC={auc:.4f}")
    joblib.dump(clf, os.path.join(outdir, 'xgb.model'))
    return clf, (Xva, yva, pva)

def train_lstm_timesplit(df_feat: pd.DataFrame, outdir: str, seq_len=60, epochs=8, batch=256, valid_frac=0.2,
                         gamma=0.0, sample_weights_full: pd.Series | None = None):
    Xseq, y = make_lstm_sequences(df_feat, seq_len=seq_len)  # (Nseq, L, F), (Nseq,)
    if len(Xseq) == 0:
        print("[LSTM] Not enough data for sequences — skipping.")
        return None, (None, None, None)

    # váhy pro sekvenční labely: vezmeme váhu pro indexy y (poslední prvek každé sekvence)
    wseq = None
    if sample_weights_full is not None:
        # make_lstm_sequences typicky dělá y = df_feat['y'][seq_len-1:]
        w_all = sample_weights_full.reindex(df_feat.index).values.astype(np.float32)
        wseq = w_all[seq_len-1:]

    n = len(Xseq)
    (i0, i1), (j0, j1) = time_split_indices(n, valid_frac=valid_frac, purge=seq_len)
    Xtr, ytr = Xseq[i0:i1], y[i0:i1]
    Xva, yva = Xseq[j0:j1], y[j0:j1]
    wtr = None if wseq is None else wseq[i0:i1]

    device = torch.device('cpu')
    model = SmallLSTM(in_features=Xseq.shape[-1], hidden=48, num_layers=1).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    dl_tr = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr)), batch_size=batch, shuffle=True)
    dl_va = DataLoader(TensorDataset(torch.from_numpy(Xva), torch.from_numpy(yva)), batch_size=batch, shuffle=False)

    best_auc, best_state = -1.0, None
    for ep in range(1, epochs + 1):
        model.train()
        for xb, yb in dl_tr:
            xb = xb.to(device); yb = yb.float().to(device)
            opt.zero_grad()
            p = model(xb)  # sigmoid pravděpodobnost
            if gamma and gamma > 0:
                ww = None
                if wtr is not None:
                    # přeneseme váhy pro tento batch podle indexů datloaderu — pro jednoduchost stejná průměrná váha
                    ww = torch.tensor(np.full(len(yb), wtr.mean(), dtype=np.float32), device=device)
                loss = focal_bce_prob(p, yb, gamma=float(gamma), weight=ww)
            else:
                loss = F.binary_cross_entropy(p, yb)
                if wtr is not None:
                    loss = (loss * float(np.mean(wtr)))  # jednoduchá škála, ať se nebortí LR
            loss.backward(); opt.step()

        model.eval()
        with torch.no_grad():
            preds, ys = [], []
            for xb, yb in dl_va:
                xb = xb.to(device)
                preds.append(model(xb).cpu().numpy()); ys.append(yb.numpy())
            pp = np.concatenate(preds); yy = np.concatenate(ys)
            auc = roc_auc_score(yy, pp)
            print(f"[LSTM] (time-split) epoch {ep}: AUC={auc:.4f}")
            if auc > best_auc:
                best_auc, best_state = auc, {k: v.cpu() for k, v in model.state_dict().items()}
                torch.save({'state_dict': best_state,
                            'in_features': Xseq.shape[-1],
                            'seq_len': seq_len},
                           os.path.join(outdir, 'lstm.pt'))
                print("[LSTM] ✓ checkpoint saved")

    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save({'state_dict': model.state_dict(),
                'in_features': Xseq.shape[-1],
                'seq_len': seq_len},
               os.path.join(outdir, 'lstm.pt'))
    print("[LSTM] ✓ final saved")

    model.eval()
    with torch.no_grad():
        preds = []
        for xb, _ in dl_va:
            xb = xb.to(device)
            preds.append(model(xb).cpu().numpy())
        pva = np.concatenate(preds)
    return model, (Xva, yva, pva)

def train_hrm_timesplit(df_feat: pd.DataFrame, outdir: str, seq_len=60, epochs=8, batch=256,
                        high_period=10, hidden_low=64, hidden_high=64, valid_frac=0.2,
                        gamma=0.0, sample_weights_full: pd.Series | None = None):
    Xseq, y = make_lstm_sequences(df_feat, seq_len=seq_len)
    if len(Xseq) == 0:
        print("[HRM] Not enough data for sequences — skipping.")
        return None, (None, None, None)

    wseq = None
    if sample_weights_full is not None:
        w_all = sample_weights_full.reindex(df_feat.index).values.astype(np.float32)
        wseq = w_all[seq_len-1:]

    n = len(Xseq)
    (i0, i1), (j0, j1) = time_split_indices(n, valid_frac=valid_frac, purge=seq_len)
    Xtr, ytr = Xseq[i0:i1], y[i0:i1]
    Xva, yva = Xseq[j0:j1], y[j0:j1]
    wtr = None if wseq is None else wseq[i0:i1]

    device = torch.device('cpu')
    model = HRMHead(in_features=Xseq.shape[-1],
                    hidden_low=hidden_low,
                    hidden_high=hidden_high,
                    high_period=high_period).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    dl_tr = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr)), batch_size=batch, shuffle=True)
    dl_va = DataLoader(TensorDataset(torch.from_numpy(Xva), torch.from_numpy(yva)), batch_size=batch, shuffle=False)

    best_auc, best_state = -1.0, None
    for ep in range(1, epochs+1):
        model.train()
        for xb, yb in dl_tr:
            xb = xb.to(device); yb = yb.float().to(device)
            opt.zero_grad()
            p = model(xb)
            if gamma and gamma > 0:
                ww = None
                if wtr is not None:
                    ww = torch.tensor(np.full(len(yb), wtr.mean(), dtype=np.float32), device=device)
                loss = focal_bce_prob(p, yb, gamma=float(gamma), weight=ww)
            else:
                loss = F.binary_cross_entropy(p, yb)
                if wtr is not None:
                    loss = (loss * float(np.mean(wtr)))
            loss.backward(); opt.step()

        model.eval()
        with torch.no_grad():
            preds, ys = [], []
            for xb, yb in dl_va:
                xb = xb.to(device)
                preds.append(model.infer(xb).cpu().numpy()); ys.append(yb.numpy())
            pp = np.concatenate(preds); yy = np.concatenate(ys)
            auc = roc_auc_score(yy, pp)
            print(f"[HRM] (time-split) epoch {ep}: AUC={auc:.4f}")
            if auc > best_auc:
                best_auc = auc
                best_state = {k: v.cpu() for k, v in model.state_dict().items()}
                torch.save({'state_dict': best_state,
                            'in_features': Xseq.shape[-1],
                            'seq_len': seq_len,
                            'hidden_low': int(hidden_low),
                            'hidden_high': int(hidden_high),
                            'high_period': int(high_period)},
                           os.path.join(outdir, 'hrm.pt'))
                print("[HRM] ✓ checkpoint saved")

    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save({'state_dict': model.state_dict(),
                'in_features': Xseq.shape[-1],
                'seq_len': seq_len,
                'hidden_low': int(hidden_low),
                'hidden_high': int(hidden_high),
                'high_period': int(high_period)},
               os.path.join(outdir, 'hrm.pt'))
    print("[HRM] ✓ final saved")

    model.eval()
    with torch.no_grad():
        preds = []
        for xb, _ in dl_va:
            xb = xb.to(device)
            preds.append(model.infer(xb).cpu().numpy())
        pva = np.concatenate(preds)
    return model, (Xva, yva, pva)

# ---------------------- OOF (volitelné) ----------------------
def oof_preds_xgb(df_feat: pd.DataFrame, n_splits: int, sample_weights: pd.Series | None = None):
    X = df_feat[FEATURE_COLS].values.astype(np.float32)
    y = df_feat['y'].values.astype(np.int64)
    w = sample_weights.reindex(df_feat.index).values.astype(np.float32) if sample_weights is not None else None
    n = len(df_feat)
    p_oof = np.full(n, np.nan, dtype=np.float32)

    for (i0, i1), (j0, j1) in ts_kfold_windows(n, n_splits=n_splits, purge=0):
        clf = xgb.XGBClassifier(
            n_estimators=600, max_depth=6, learning_rate=0.03,
            subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
            tree_method='hist', random_state=42, n_jobs=4
        )
        wtr = w[i0:i1] if w is not None else None
        clf.fit(X[i0:i1], y[i0:i1], sample_weight=wtr)
        p = clf.predict_proba(X[j0:j1])[:, 1]
        p_oof[j0:j1] = p

    mask = ~np.isnan(p_oof)
    auc = roc_auc_score(y[mask], p_oof[mask]) if mask.any() else float('nan')
    print(f"[XGB/OOF] AUC={auc:.4f} (n_splits={n_splits})")
    return p_oof, y

def oof_preds_seq(df_feat: pd.DataFrame, n_splits: int, seq_len: int, model_kind: str,
                  gamma: float = 0.0, sample_weights_full: pd.Series | None = None,
                  hrm_cfg=None):
    Xseq, y = make_lstm_sequences(df_feat, seq_len=seq_len)
    n = len(Xseq)
    if n == 0:
        print(f"[{model_kind}/OOF] Not enough data.")
        return None, None

    # váhy pro sekvenční labely (poslední krok sekvence)
    wseq = None
    if sample_weights_full is not None:
        w_all = sample_weights_full.reindex(df_feat.index).values.astype(np.float32)
        wseq = w_all[seq_len-1:]

    p_oof = np.full(n, np.nan, dtype=np.float32)
    skipped = 0

    for (i0, i1), (j0, j1) in ts_kfold_windows(n, n_splits=n_splits, purge=seq_len):
        n_tr = i1 - i0
        n_va = j1 - j0
        if n_tr <= 0 or n_va <= 0:
            skipped += 1
            continue  # přeskoč prázdný fold

        device = torch.device('cpu')
        if model_kind == "LSTM":
            model = SmallLSTM(in_features=Xseq.shape[-1], hidden=48, num_layers=1).to(device)
            fwd = lambda xb: model(xb)
        else:  # HRM
            hp = (hrm_cfg or {}).get("high_period", 10)
            hl = (hrm_cfg or {}).get("hidden_low", 64)
            hh = (hrm_cfg or {}).get("hidden_high", 64)
            model = HRMHead(in_features=Xseq.shape[-1], hidden_low=hl, hidden_high=hh, high_period=hp).to(device)
            fwd = lambda xb: model.infer(xb)

        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        dl_tr = DataLoader(TensorDataset(torch.from_numpy(Xseq[i0:i1]), torch.from_numpy(y[i0:i1])), batch_size=256, shuffle=True)
        dl_va = DataLoader(TensorDataset(torch.from_numpy(Xseq[j0:j1]), torch.from_numpy(y[j0:j1])), batch_size=256, shuffle=False)

        # kratší OOF trénink
        for ep in range(1, 7):
            model.train()
            for xb, yb in dl_tr:
                xb = xb.to(device); yb = yb.float().to(device)
                opt.zero_grad()
                p = model(xb)
                if gamma and gamma > 0:
                    ww = None
                    if wseq is not None:
                        ww = torch.tensor(np.full(len(yb), float(np.mean(wseq[i0:i1])), dtype=np.float32), device=device)
                    loss = focal_bce_prob(p, yb, gamma=float(gamma), weight=ww)
                else:
                    loss = F.binary_cross_entropy(p, yb)
                loss.backward(); opt.step()

        model.eval()
        with torch.no_grad():
            preds = []
            for xb, _ in dl_va:
                xb = xb.to(device)
                preds.append(fwd(xb).cpu().numpy())
            p = np.concatenate(preds).reshape(-1)
        p_oof[j0:j1] = p

    mask = ~np.isnan(p_oof)
    if mask.any():
        auc = roc_auc_score(y[mask], p_oof[mask])
        print(f"[{model_kind}/OOF] AUC={auc:.4f} (n_splits={n_splits}, skipped_folds={skipped})")
    else:
        print(f"[{model_kind}/OOF] No valid folds (all skipped).")
    return p_oof, y
    
# ---------------------- kalibrace ----------------------
def calibrate_platt_and_save(outdir: str, model_name: str, p_valid: np.ndarray, y_valid: np.ndarray):
    """
    Platt kalibrace (logistická regrese na logitech) pro XGB, LSTM i HRM.
    Bezpečný fallback: pokud y_valid má <2 třídy, kalibrace se přeskočí (vrací None).
    """
    if p_valid is None or y_valid is None or len(p_valid) == 0:
        return None
    # musí obsahovat obě třídy
    if np.unique(y_valid).size < 2:
        print(f"[CAL] {model_name.upper()} skipped (only one class in valid).")
        return None
    eps = 1e-6
    p_clip = np.clip(p_valid, eps, 1-eps)
    logit = np.log(p_clip/(1-p_clip)).reshape(-1,1)
    try:
        lr = LogisticRegression(max_iter=1000)
        lr.fit(logit, y_valid)
    except Exception as e:
        print(f"[CAL] {model_name.upper()} failed: {e} — skipping.")
        return None
    joblib.dump(lr, os.path.join(outdir, f"{model_name}.calib.pkl"))
    print(f"[CAL] {model_name.upper()} Platt ✓ saved")
    def _cal(p):
        p = np.clip(p, eps, 1-eps)
        log = np.log(p/(1-p)).reshape(-1,1)
        return lr.predict_proba(log)[:,1]
    return _cal
# ---------------------- meta HRM nad pravděpodobnostmi ----------------------
def _make_prob_sequences(pmat: np.ndarray, y: np.ndarray, seq_len: int):
    assert pmat.ndim == 2 and pmat.shape[1] in (2,3)
    N, C = pmat.shape
    if N < seq_len:
        return None, None
    Xs, ys = [], []
    for t in range(seq_len-1, N):
        Xs.append(pmat[t-seq_len+1:t+1, :])
        ys.append(y[t])
    Xs = np.stack(Xs, axis=0).astype(np.float32)
    ys = np.asarray(ys).astype(np.int64)
    return Xs, ys

def train_meta_hrm(outdir: str, p_xgb, p_lstm, p_hrm, y_valid, seq_len=60, epochs=8, batch=256,
                   high_period=10, hidden_low=32, hidden_high=32):
    cols = []
    if p_xgb is not None: cols.append(p_xgb.reshape(-1,1))
    if p_lstm is not None: cols.append(p_lstm.reshape(-1,1))
    if p_hrm is not None:  cols.append(p_hrm.reshape(-1,1))
    if len(cols) < 2:
        print("[META] Not enough components to train meta (need ≥2). Skipping.")
        return None, 0.0
    pmat = np.concatenate(cols, axis=1)
    Xs, ys = _make_prob_sequences(pmat, y_valid, seq_len=seq_len)
    if Xs is None:
        print("[META] Not enough data for sequences. Skipping.")
        return None, 0.0

    device = torch.device('cpu')
    model = HRMHead(in_features=pmat.shape[1],
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
            opt.zero_grad(); p = model(xb); loss = F.binary_cross_entropy(p, yb)
            loss.backward(); opt.step()

        model.eval()
        with torch.no_grad():
            preds, yy = [], []
            for xb, yb in dl:
                xb = xb.to(device)
                preds.append(model.infer(xb).cpu().numpy()); yy.append(yb.numpy())
            pp = np.concatenate(preds); yyy = np.concatenate(yy)
            auc = roc_auc_score(yyy, pp)
        print(f"[META] epoch {ep}: AUC={auc:.4f}")
        if auc > best_auc:
            best_auc = auc
            best_state = {k: v.cpu() for k,v in model.state_dict().items()}
            torch.save({'state_dict': best_state,
                        'in_features': pmat.shape[1],
                        'seq_len': seq_len,
                        'hidden_low': hidden_low,
                        'hidden_high': hidden_high,
                        'high_period': int(high_period)},
                       os.path.join(outdir, 'hrm_meta.pt'))
            print("[META] ✓ checkpoint saved")

    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save({'state_dict': model.state_dict(),
                'in_features': pmat.shape[1],
                'seq_len': seq_len,
                'hidden_low': hidden_low,
                'hidden_high': hidden_high,
                'high_period': int(high_period)},
               os.path.join(outdir, 'hrm_meta.pt'))
    print("[META] ✓ final saved")
    return model, best_auc

# ---------------------- ULOŽENÍ OOF (NOVÉ) ----------------------
def save_oof_npz(outdir: str,
                 y_base: np.ndarray,
                 p_xgb_base: np.ndarray,
                 seq_len: int,
                 p_lstm_oof: np.ndarray | None = None,
                 p_hrm_oof: np.ndarray | None = None,
                 y_seq_valid: np.ndarray | None = None,
                 p_lstm_valid: np.ndarray | None = None,
                 p_hrm_valid: np.ndarray | None = None,
                 y_tab_valid: np.ndarray | None = None,
                 p_xgb_valid: np.ndarray | None = None):
    """
    Uloží oof_preds.npz se zarovnáním délek kanálů. Preferuje OOF, jinak fallback na validační split.
    Výstupní klíče: 'y', 'p_xgb', volitelně 'p_lstm', 'p_hrm'
    """
    use_oof = (y_base is not None) and (p_xgb_base is not None)
    # --- případ s aspoň jedním sekvenčním kanálem ---
    if (p_lstm_oof is not None) or (p_hrm_oof is not None) or (p_lstm_valid is not None) or (p_hrm_valid is not None):
        if use_oof:
            # posuň XGB i y o (seq_len-1), aby korespondovaly sekvenčním predikcím
            y = y_base[seq_len-1:]
            px = p_xgb_base[seq_len-1:]
            cand = [len(px)]
            if p_lstm_oof is not None: cand.append(len(p_lstm_oof))
            if p_hrm_oof  is not None: cand.append(len(p_hrm_oof))
            L = min(cand)
            y_save  = y[-L:]
            px_save = px[-L:]
            pl_save = (p_lstm_oof[-L:] if p_lstm_oof is not None else None)
            ph_save = (p_hrm_oof[-L:]  if p_hrm_oof  is not None else None)
        else:
            # fallback na validační: vezmi sekvenční valid jako referenci
            lens = []
            if y_seq_valid is not None: lens.append(len(y_seq_valid))
            if p_lstm_valid is not None: lens.append(len(p_lstm_valid))
            if p_hrm_valid  is not None: lens.append(len(p_hrm_valid))
            if not lens:  # nemáme nic použitelného
                return
            L = min(lens)
            # y preferujeme sekvenční; když není, použij tabular a srovnej na konec
            if y_seq_valid is not None:
                y_save = y_seq_valid[-L:]
            elif y_tab_valid is not None:
                y_save = y_tab_valid[-L:]
            else:
                return
            # XGB valid zarovnej na stejný tail
            if p_xgb_valid is not None:
                px_save = p_xgb_valid[-L:]
            else:
                return
            pl_save = (p_lstm_valid[-L:] if p_lstm_valid is not None else None)
            ph_save = (p_hrm_valid[-L:]  if p_hrm_valid  is not None else None)
    else:
        # jen XGB
        if use_oof:
            y_save, px_save = y_base, p_xgb_base
        else:
            if (y_tab_valid is None) or (p_xgb_valid is None):
                return
            L = min(len(y_tab_valid), len(p_xgb_valid))
            y_save, px_save = y_tab_valid[-L:], p_xgb_valid[-L:]
        pl_save = None
        ph_save = None

    payload = {"y": y_save.astype(np.int64), "p_xgb": px_save.astype(np.float32)}
    if pl_save is not None:
        payload["p_lstm"] = pl_save.astype(np.float32)
    if ph_save is not None:
        payload["p_hrm"]  = ph_save.astype(np.float32)
    path = os.path.join(outdir, "oof_preds.npz")
    np.savez(path, **payload)
    log = ", ".join([f"{k}={len(v)}" for k, v in payload.items()])
    print(f"[OOF/SAVE] → {path} ({log})")

# ---------------------- Main ----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--symbol', default='ETHUSDT')
    ap.add_argument('--days', type=int, default=1)
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--no-download', action='store_true')
    ap.add_argument('--save-features', action='store_true')
    ap.add_argument('--epochs', type=int, default=8)
    ap.add_argument('--horizon', type=int, default=30)
    ap.add_argument('--seq-len', type=int, default=None,help='Sekvenční délka pro LSTM/HRM. Když není zadáno, použije se 60 pro horizon >= 30, jinak 30.')

    # validace / K-fold
    ap.add_argument('--valid-frac', type=float, default=0.2)
    ap.add_argument('--oof-splits', type=int, default=1, help=">1 => použije OOF časové K-foldy")

    # dead-zone a focal
    ap.add_argument('--deadzone-eps', type=float, default=3e-4, help="např. 0.0003 = 0.03%")
    ap.add_argument('--deadzone-weight', type=float, default=0.2)
    ap.add_argument('--focal-gamma', type=float, default=0.0, help=">0 aktivuje focal loss u LSTM/HRM (např. 1.5)")

    # HRM (base)
    ap.add_argument('--use-hrm', dest='use_hrm', action='store_true')
    ap.add_argument('--hrm-high-period', type=int, default=10)
    ap.add_argument('--hrm-hidden-low', type=int, default=64)
    ap.add_argument('--hrm-hidden-high', type=int, default=64)

    # META
    ap.add_argument('--train-meta', dest='train_meta', action='store_true')
    ap.add_argument('--meta-seq-len', type=int, default=60)
    ap.add_argument('--meta-epochs', type=int, default=8)
    ap.add_argument('--meta-high-period', type=int, default=10)
    ap.add_argument('--meta-hidden-low', type=int, default=32)
    ap.add_argument('--meta-hidden-high', type=int, default=32)

    # auto-threshold parametry
    ap.add_argument('--thr-min-margin', type=float, default=0.0)
    ap.add_argument('--thr-min-act', type=float, default=0.10)
    ap.add_argument('--thr-min-abstain', type=float, default=0.0)

    ap.add_argument('--outdir', type=str, default=None, help="Kam uložit váhy (např. app/models/weights/30s)")
    args = ap.parse_args()

    outdir = _ensure_outdir(args.outdir)

    # Data
    if args.no_download:
        df = load_cache_1s(args.symbol)
        print(f"[CACHE] Using cached seconds: {df.index.min()}..{df.index.max()} ({len(df)})")
    else:
        print(f"Fetching {args.days} day(s) of {args.symbol} agg trades (resume={args.resume}) ...")
        df = fetch_agg_trades_resumable(symbol=args.symbol, days=args.days, resume=args.resume)

    # Features
    print("Preparing features…")
    df_feat = compute_features(df, horizon=args.horizon)
    print(f"Feature rows: {len(df_feat)}")
    if args.save_features:
        save_features(args.symbol, df_feat)

    # ---------- DEAD-ZONE VÁHY ----------
    r_h = compute_future_return(df, horizon_sec=args.horizon)
    w_series = deadzone_weights_from_returns(r_h.reindex(df_feat.index), eps=float(args.deadzone_eps),
                                             w_dead=float(args.deadzone_weight), w_full=1.0)
    # report class-balance
    if 'y' in df_feat.columns:
        y_all = df_feat['y'].values.astype(np.int64)
        print(f"[REPORT] class-balance (y=1 ratio): {float(np.mean(y_all)):.3f}")

    # ---------- TRAIN (XGB/LSTM/HRM) ----------
    valid_frac = float(args.valid_frac)
    lstm_seq = int(args.seq_len) if args.seq_len else (60 if args.horizon >= 30 else 30)
    print(f"[SEQ] Using seq_len={lstm_seq} for LSTM/HRM (override via --seq-len)")

    # XGB (time-split)
    xgb_model, (_, yva_tab, pva_xgb_raw) = train_xgb_timesplit(df_feat, outdir=outdir, valid_frac=valid_frac,
                                                               sample_weights=w_series)

    # LSTM (time-split)
    lstm_model, (_, yva_seq, pva_lstm_raw) = train_lstm_timesplit(df_feat, outdir=outdir, seq_len=lstm_seq,
                                                                  epochs=args.epochs, valid_frac=valid_frac,
                                                                  gamma=float(args.focal_gamma),
                                                                  sample_weights_full=w_series)

    # HRM base (vol.)
    pva_hrm_raw = None; yva_hrm = None
    if args.use_hrm:
        _, (_, yva_hrm, pva_hrm_raw) = train_hrm_timesplit(
            df_feat, outdir=outdir, seq_len=lstm_seq, epochs=args.epochs,
            high_period=max(5, min(30, args.hrm_high_period)),
            hidden_low=args.hrm_hidden_low, hidden_high=args.hrm_hidden_high,
            valid_frac=valid_frac, gamma=float(args.focal_gamma),
            sample_weights_full=w_series
        )

    # ---------- OOF (volitelně) ----------
    p_oof_xgb = p_oof_lstm = p_oof_hrm = None
    y_oof_xgb = y_oof_lstm = y_oof_hrm = None

    if args.oof_splits and args.oof_splits > 1:
        print(f"[OOF] Generuji OOF s n_splits={args.oof_splits} …")
        p_oof_xgb, y_oof_xgb = oof_preds_xgb(df_feat, n_splits=int(args.oof_splits), sample_weights=w_series)
        p_oof_lstm, y_oof_lstm = oof_preds_seq(df_feat, n_splits=int(args.oof_splits), seq_len=lstm_seq,
                                               model_kind="LSTM", gamma=float(args.focal_gamma),
                                               sample_weights_full=w_series)
        if args.use_hrm:
            p_oof_hrm, y_oof_hrm = oof_preds_seq(df_feat, n_splits=int(args.oof_splits), seq_len=lstm_seq,
                                                 model_kind="HRM", gamma=float(args.focal_gamma),
                                                 sample_weights_full=w_series,
                                                 hrm_cfg={"high_period": max(5, min(30, args.hrm_high_period)),
                                                          "hidden_low": args.hrm_hidden_low,
                                                          "hidden_high": args.hrm_hidden_high})

    # ---------- KALIBRACE + REPORT ----------
    # XGB
    cal_xgb = calibrate_platt_and_save(outdir, "xgb",
                                       p_valid=(p_oof_xgb[~np.isnan(p_oof_xgb)] if p_oof_xgb is not None else pva_xgb_raw),
                                       y_valid=(y_oof_xgb[~np.isnan(p_oof_xgb)] if p_oof_xgb is not None else yva_tab))
    pva_xgb_cal = cal_xgb(pva_xgb_raw) if cal_xgb is not None else pva_xgb_raw
    _quant_report("XGB", pva_xgb_raw, pva_xgb_cal)

    # LSTM
    cal_lstm = None; pva_lstm_cal = None
    if pva_lstm_raw is not None:
        if p_oof_lstm is not None and y_oof_lstm is not None:
            mask = ~np.isnan(p_oof_lstm)
            cal_lstm = calibrate_platt_and_save(outdir, "lstm", p_oof_lstm[mask], y_oof_lstm[mask])
        elif yva_seq is not None and len(yva_seq)==len(pva_lstm_raw):
            cal_lstm = calibrate_platt_and_save(outdir, "lstm", pva_lstm_raw, yva_seq)
        pva_lstm_cal = cal_lstm(pva_lstm_raw) if cal_lstm is not None else pva_lstm_raw
        _quant_report("LSTM", pva_lstm_raw, pva_lstm_cal)

    # HRM
    if pva_hrm_raw is not None:
        if p_oof_hrm is not None and y_oof_hrm is not None:
            mask = ~np.isnan(p_oof_hrm)
            _quant_report("HRM (OOF)", p_oof_hrm[mask], None)
        _quant_report("HRM", pva_hrm_raw, None)

    # ---------- AUTO-THR (z XGB po kalibraci) ----------
    up_thr, dn_thr, margin, best_mcc, cov = auto_thresholds(
        pva_xgb_cal, yva_tab, max_margin=0.2,
        min_action_coverage=float(args.thr_min_act),
        min_abstain_coverage=float(args.thr_min_abstain),
        min_margin=float(args.thr_min_margin)
    )
    print(f"[AUTO-THR] up={up_thr:.3f} down={dn_thr:.3f} margin={margin:.3f} MCC={best_mcc:.4f} coverage={cov:.2f}")

    # ---------- ULOŽ OOF (NOVÉ) ----------
    # Preferuj OOF; pokud není, spadne to na validační split (zarovnání viz funkce)
    save_oof_npz(
        outdir=outdir,
        y_base=y_oof_xgb if (p_oof_xgb is not None and y_oof_xgb is not None) else yva_tab,
        p_xgb_base=p_oof_xgb if (p_oof_xgb is not None) else pva_xgb_raw,
        seq_len=lstm_seq,
        p_lstm_oof=p_oof_lstm,
        p_hrm_oof=p_oof_hrm,
        y_seq_valid=yva_seq,
        p_lstm_valid=pva_lstm_raw,
        p_hrm_valid=pva_hrm_raw,
        y_tab_valid=yva_tab,
        p_xgb_valid=pva_xgb_raw
    )

    # ---------- META: zarovnat délky a naučit ----------
    if args.train_meta:
        shift = max(0, lstm_seq-1)
        y_tab_al = yva_tab[shift:]
        px_al = pva_xgb_raw[shift:]
        pl_al = pva_lstm_raw if pva_lstm_raw is not None else None
        ph_al = pva_hrm_raw  if pva_hrm_raw  is not None else None

        lengths = [len(y_tab_al), len(px_al)]
        if pl_al is not None: lengths.append(len(pl_al))
        if ph_al is not None: lengths.append(len(ph_al))
        L = min(lengths)

        y_meta = y_tab_al[-L:]
        px_meta = px_al[-L:]
        pl_meta = pl_al[-L:] if pl_al is not None else None
        ph_meta = ph_al[-L:] if ph_al is not None else None

        _, meta_auc = train_meta_hrm(
            outdir=outdir,
            p_xgb=px_meta,
            p_lstm=pl_meta,
            p_hrm=ph_meta,
            y_valid=y_meta,
            seq_len=args.meta_seq_len,
            epochs=args.meta_epochs,
            high_period=max(5, min(30, args.meta_high_period)),
            hidden_low=args.meta_hidden_low,
            hidden_high=args.meta_hidden_high
        )

    meta = {
        "horizon_sec": int(args.horizon),
        "trained_at": int(time.time()),
        "recommended": {"CONF_ENTER_UP": float(up_thr), "CONF_ENTER_DOWN": float(dn_thr), "ABSTAIN_MARGIN": float(margin)},
        "calibration": {"xgb": bool(cal_xgb), "lstm": bool(cal_lstm)},
        "seq_len": int(lstm_seq),
        # volitelný alias pro kompatibilitu
        "auto_thr": {"up": float(up_thr), "down": float(dn_thr), "margin": float(margin)}
    }
    with open(os.path.join(outdir, "meta.json"), "w") as f:
        json.dump(meta, f)
    print(f"[META] Saved → {os.path.join(outdir, 'meta.json')} {meta}")

if __name__ == '__main__':
    main()