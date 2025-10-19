# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

# Seznam featur (udržuj krátký a čistý; můžeš později rozšířit)
FEATURE_COLS = [
    # returns (více měřítek)
    'ret_1', 'ret_3', 'ret_5', 'ret_15', 'ret_30',
    # EMA poměry (trend vs mean reversion)
    'ema_5_over_15', 'ema_15_over_60',
    # volatilita
    'vol_ret_10', 'vol_ret_30',
    # orderflow / imbalance (pokud k dispozici)
    'imb_5', 'imb_15',
    # akcelerace ceny
    'dclose_1', 'dclose_3',
    # čas dne (cyklické)
    'tod_sin', 'tod_cos',
    # objem (log)
    'log_vol_5',
]

def _safe_div(a, b, eps=1e-9):
    return a / np.maximum(np.abs(b), eps)

def _ema(s, span):
    return s.ewm(span=span, adjust=False).mean()

def _rsi(close, n=14):
    # zde RSI nakonec nepoužíváme do FEATURE_COLS, ale můžeš přidat
    delta = close.diff()
    up = delta.clip(lower=0.0).rolling(n).mean()
    dn = (-delta.clip(upper=0.0)).rolling(n).mean()
    rs = _safe_div(up, dn)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi

def compute_features(df: pd.DataFrame, horizon: int = 30, deadzone_eps: float = 0.0005) -> pd.DataFrame:
    """
    df: index=ts (sekundy), sloupce: open,high,low,close,volume,(buy_vol),(sell_vol)
    horizon: posun v sekundách dopředu
    deadzone_eps: mrtvá zóna pro |r_h| (relat. změna). Hodnoty ~0.0003–0.0007 se osvědčují.
    """
    d = df.copy().sort_index()
    # Základní returns v různých oknech
    close = d['close'].astype(float)
    ret_1  = close.pct_change(1)
    ret_3  = close.pct_change(3)
    ret_5  = close.pct_change(5)
    ret_15 = close.pct_change(15)
    ret_30 = close.pct_change(30)

    # EMA poměry
    ema5  = _ema(close, 5)
    ema15 = _ema(close, 15)
    ema60 = _ema(close, 60)
    ema_5_over_15  = _safe_div(ema5, ema15) - 1.0
    ema_15_over_60 = _safe_div(ema15, ema60) - 1.0

    # Volatilita returns
    vol_ret_10 = ret_1.rolling(10).std()
    vol_ret_30 = ret_1.rolling(30).std()

    # Imbalance – pokud chybí buy_vol/sell_vol, udělej 0
    if 'buy_vol' in d.columns and 'sell_vol' in d.columns:
        imb = _safe_div(d['buy_vol'] - d['sell_vol'], d['volume'] + 1e-9)
    else:
        imb = pd.Series(0.0, index=d.index)
    imb_5  = imb.rolling(5).mean()
    imb_15 = imb.rolling(15).mean()

    # Akcelerace ceny
    dclose_1 = close.diff(1)
    dclose_3 = close.diff(3)

    # Čas dne (cyklické)
    tod = (d.index.values % 86400).astype(np.float64)  # sekund v dni
    tod_sin = np.sin(2*np.pi * tod / 86400.0)
    tod_cos = np.cos(2*np.pi * tod / 86400.0)

    # Objem – log průměr
    log_vol_5 = np.log1p(d['volume'].rolling(5).mean())

    out = pd.DataFrame({
        'ret_1': ret_1, 'ret_3': ret_3, 'ret_5': ret_5, 'ret_15': ret_15, 'ret_30': ret_30,
        'ema_5_over_15': ema_5_over_15, 'ema_15_over_60': ema_15_over_60,
        'vol_ret_10': vol_ret_10, 'vol_ret_30': vol_ret_30,
        'imb_5': imb_5, 'imb_15': imb_15,
        'dclose_1': dclose_1, 'dclose_3': dclose_3,
        'tod_sin': tod_sin, 'tod_cos': tod_cos,
        'log_vol_5': log_vol_5,
    }, index=d.index)

    # Label a váha (dead-zone)
    future = close.shift(-horizon)
    r_h = (future - close) / close
    y = (r_h > 0.0).astype(np.int64)
    w = np.where(np.abs(r_h) < float(deadzone_eps), 0.2, 1.0)  # „tiché“ případy méně vážené

    out['y'] = y
    out['w'] = w
    # Vyhoď NaN na zač/konci
    out = out.dropna()
    out['close']  = df['close'].reindex(out.index).astype(float)
    out['volume'] = df['volume'].reindex(out.index).astype(float)
    out = out.reset_index().rename(columns={'ts':'ts'}).set_index('ts').sort_index()
    return out

def make_lstm_sequences(
    df_feat,
    seq_len: int = 60,
    feature_cols: list | None = None,
    target_col: str = 'y',
    return_index: bool = False,
):
    """
    Vytvoří sekvence pro sekvenční modely.

    Parametry:
      - df_feat: DataFrame se sloupci feature_cols a target_col, index = ts (sekundy)
      - seq_len: délka sekvence (počet kroků)
      - feature_cols: seznam featur; když None, použije FEATURE_COLS
      - target_col: název cílového sloupce (binární 0/1)
      - return_index: True => vrátí navíc i zarovnané indexy posledních kroků

    Návrat:
      - když return_index=False: (X, y)
      - když return_index=True:  (X, y, idx)
        X  shape = (Nseq, seq_len, F)
        y  shape = (Nseq,)
        idx shape = (Nseq,) – index (ts) odpovídající y[t] (tj. poslední krok sekvence)
    """
    import numpy as np
    import pandas as pd

    if feature_cols is None:
        from .features import FEATURE_COLS as _DEF_COLS  # bezpečně z lokálního modulu
        feature_cols = _DEF_COLS

    if df_feat is None or len(df_feat) == 0:
        if return_index:
            return (np.empty((0, seq_len, len(feature_cols)), dtype=np.float32),
                    np.empty((0,), dtype=np.int64),
                    np.empty((0,), dtype=np.int64))
        else:
            return (np.empty((0, seq_len, len(feature_cols)), dtype=np.float32),
                    np.empty((0,), dtype=np.int64))

    Xall = df_feat[feature_cols].values.astype(np.float32)
    yall = df_feat[target_col].values.astype(np.int64)
    idx_all = df_feat.index.values
    N = len(df_feat)

    if N < seq_len:
        if return_index:
            return (np.empty((0, seq_len, Xall.shape[1]), dtype=np.float32),
                    np.empty((0,), dtype=np.int64),
                    np.empty((0,), dtype=idx_all.dtype))
        else:
            return (np.empty((0, seq_len, Xall.shape[1]), dtype=np.float32),
                    np.empty((0,), dtype=np.int64))

    # poskládej sekvence přes "klouzavé okno"
    Xs = np.stack([Xall[t-seq_len+1:t+1, :] for t in range(seq_len-1, N)], axis=0)
    ys = yall[seq_len-1:]
    idx_out = idx_all[seq_len-1:]

    if return_index:
        return Xs, ys, idx_out
    return Xs, ys