# app/runtime/drift_monitor.py
# -*- coding: utf-8 -*-
import argparse, json, numpy as np, pandas as pd

def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    e_hist, edges = np.histogram(expected, bins=bins)
    a_hist, _ = np.histogram(actual, bins=edges)
    e_perc = e_hist / (e_hist.sum() + 1e-12)
    a_perc = a_hist / (a_hist.sum() + 1e-12)
    e_perc = np.clip(e_perc, 1e-6, 1.0); a_perc = np.clip(a_perc, 1e-6, 1.0)
    return float(np.sum((a_perc - e_perc) * np.log(a_perc / e_perc)))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--baseline-parquet', required=True, help='Parquet s baseline featurami (train okno)')
    ap.add_argument('--live-parquet', required=True, help='Parquet s live featurami (stejné sloupce)')
    ap.add_argument('--features', nargs='*', default=None)
    args = ap.parse_args()

    df_b = pd.read_parquet(args.baseline_parquet)
    df_l = pd.read_parquet(args.live_parquet)

    cols = args.features or list(set(df_b.columns) & set(df_l.columns))
    out = {}
    for c in cols:
        try:
            b = df_b[c].dropna().values; a = df_l[c].dropna().values
            if len(b)>100 and len(a)>100:
                out[c] = psi(b, a, bins=10)
        except Exception:
            pass
    print(json.dumps({'psi': out, 'mean': float(np.mean(list(out.values())) if out else 0.0)}, indent=2))

if __name__ == '__main__':
    main()