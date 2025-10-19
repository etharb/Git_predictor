# app/runtime/backtest.py
# -*- coding: utf-8 -*-
import argparse, json, numpy as np, pandas as pd

def backtest_from_probs(df: pd.DataFrame, thr_up: float, thr_down: float, margin: float,
                        fee_bps: float = 0.0, slip_bps: float = 0.0, horizon: int = 10):
    """
    df: očekává sloupce ['ts', 'close', 'prob_up'] (časově seřazené, 1s krok)
    """
    df = df.sort_values('ts').reset_index(drop=True)
    fee = (fee_bps + slip_bps) / 1e4
    pnl = []
    for i in range(len(df)-horizon):
        p = float(df.loc[i, 'prob_up'])
        c0 = float(df.loc[i, 'close'])
        c1 = float(df.loc[i+horizon, 'close'])
        dec = 0
        if p >= (thr_up + margin): dec = +1
        elif p <= (thr_down - margin): dec = -1
        if dec == 0:
            pnl.append(0.0)
        else:
            r = (c1 - c0) / (c0 + 1e-12)
            g = dec * r - fee
            pnl.append(g)
    pnl = np.array(pnl, dtype=np.float64)
    ret = pnl.mean()
    vol = pnl.std() + 1e-12
    sharpe = ret / vol * np.sqrt(365*24*60*60 / horizon)  # annualizace z Hz
    dd = 0.0
    if len(pnl)>0:
        curve = np.cumsum(pnl)
        peak = np.maximum.accumulate(curve)
        dd = float(np.min(curve - peak))
    return {'mean': float(ret), 'vol': float(vol), 'sharpe': float(sharpe), 'max_dd': dd, 'trades': int((np.abs(pnl)>0).sum())}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True, help='CSV se sloupci ts,close,prob_up (např. z /download/signals)')
    ap.add_argument('--thr-up', type=float, required=True)
    ap.add_argument('--thr-down', type=float, required=True)
    ap.add_argument('--thr-margin', type=float, required=True)
    ap.add_argument('--fee-bps', type=float, default=0.0)
    ap.add_argument('--slip-bps', type=float, default=0.0)
    ap.add_argument('--horizon', type=int, default=10)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    res = backtest_from_probs(df, args.thr_up, args.thr_down, args.thr_margin,
                              fee_bps=args.fee_bps, slip_bps=args.slip_bps, horizon=args.horizon)
    print(json.dumps(res, indent=2))

if __name__ == '__main__':
    main()
