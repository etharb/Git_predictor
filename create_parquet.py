import os, pandas as pd
from app.utils.features import compute_features

parq = "app/data/ETHUSDT_1s.parquet"
csv  = "app/data/ETHUSDT_1s.csv"

if os.path.exists(parq):
    df = pd.read_parquet(parq)
elif os.path.exists(csv):
    df = pd.read_csv(csv)
    if "ts" in df.columns: df = df.set_index("ts")
    df = df.sort_index()
else:
    raise SystemExit("Nenalezeno: app/data/ETHUSDT_1s.parquet ani .csv – viz bod (b).")

dff = compute_features(df, horizon=10)
dff.to_parquet("app/data/train_1s.parquet")
print("OK → app/data/train_1s.parquet, rows:", len(dff))
print(dff.head(3))