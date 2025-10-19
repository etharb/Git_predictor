import os
import pandas as pd

# vstupy – uprav, pokud máš jiné názvy
ohlcv_csv = 'app/data/ETHUSDT_1s.csv'          # musí mít ts, open, high, low, close, volume
feat_csv  = 'app/data/ETHUSDT_features.csv'    # má ts + feature sloupce (ret_*, ema_*, ...)

# načti OHLCV
df_ohlcv = pd.read_csv(ohlcv_csv)
if 'ts' in df_ohlcv.columns:
    df_ohlcv = df_ohlcv.set_index('ts')
# sanity pro názvy sloupců (případné alternativy)
rename_map = {}
for k in ['Open','High','Low','Close','Volume','price']:
    if k in df_ohlcv.columns:
        if k=='Open':   rename_map[k]='open'
        if k=='High':   rename_map[k]='high'
        if k=='Low':    rename_map[k]='low'
        if k=='Close':  rename_map[k]='close'
        if k=='Volume': rename_map[k]='volume'
        if k=='price':  rename_map[k]='close'
df_ohlcv = df_ohlcv.rename(columns=rename_map)

need = {'open','high','low','close','volume'}
missing = need - set(df_ohlcv.columns)
if missing:
    raise SystemExit(f"OHLCV soubor postrádá sloupce: {missing}. Uprav cesty/názvy.")

# načti featury
df_feat = pd.read_csv(feat_csv)
if 'ts' in df_feat.columns:
    df_feat = df_feat.set_index('ts')

# slouč podle indexu (ts), OHLCV + featury
df = df_ohlcv.join(df_feat, how='inner').sort_index()

out_path = 'app/data/train_1s.parquet'
df.to_parquet(out_path)
print(f"OK → {out_path}")
print("Sloupce:", list(df.columns)[:8], '...')