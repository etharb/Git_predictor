# app/runtime/tune_thresholds.py
# -*- coding: utf-8 -*-
import argparse, numpy as np, pandas as pd, json
from backtest import backtest_from_probs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--horizon', type=int, default=10)
    ap.add_argument('--fee-bps', type=float, default=0.0)
    ap.add_argument('--slip-bps', type=float, default=0.0)
    ap.add_argument('--grid', type=int, default=11, help='jemnost mřížky (liché číslo)')
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    best = None
    grid = int(args.grid)
    ups   = np.linspace(0.52, 0.70, grid)
    downs = np.linspace(0.30, 0.48, grid)
    margins = np.linspace(0.00, 0.05, 7)
    for u in ups:
        for d in downs:
            if d >= u: continue
            for m in margins:
                r = backtest_from_probs(df, u, d, m, fee_bps=args.fee_bps, slip_bps=args.slip_bps, horizon=args.horizon)
                key = {'up':float(u), 'down':float(d), 'margin':float(m)}
                if best is None or r['sharpe'] > best['res']['sharpe']:
                    best = {'thr': key, 'res': r}
    print(json.dumps(best, indent=2))

if __name__ == '__main__':
    main()