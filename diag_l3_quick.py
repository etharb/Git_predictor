#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
quick_l3_check.py
Rychlá diagnostika před L3 tréninkem (čistě v Pythonu, bez výpočtů).
Zkontroluje:
 - že train_1s.parquet existuje a není prázdný
 - že v každém setu (1s,5s,...) jsou modely (xgb/lstm/hrm)
 - že hrm_meta.pt má správný počet in_features
 - že L2 heads mají meta_gate.sets, l1_seq_len, l2_seq_len, use_raw_in_l2

Spusť:
    (.venv) python quick_l3_check.py
"""

import os
import json
import torch
import pandas as pd

# === Nastavení ===
DATA_FILE = "app/data/train_1s.parquet"
TARGET_SET = "app/models/weights/10s"
SETS = [
    "app/models/weights/1s",
    "app/models/weights/5s",
    "app/models/weights/10s",
    "app/models/weights/15s",
    "app/models/weights/30s",
    "app/models/weights/60s",
    "app/models/weights/180s",
]
L2_HEADS = [
    "app/models/weights/1s",
    "app/models/weights/10s",
    "app/models/weights/15s",
    "app/models/weights/30s",
    "app/models/weights/60s",
    "app/models/weights/180s",
]


def check_data_file(path: str) -> bool:
    if not os.path.exists(path):
        print(f"[DATA] ❌ Soubor {path} neexistuje.")
        return False
    try:
        df = pd.read_parquet(path)
    except Exception as e:
        print(f"[DATA] ❌ Nelze načíst {path}: {e}")
        return False
    if len(df) == 0:
        print(f"[DATA] ❌ {path} je prázdný (0 řádků).")
        return False
    print(f"[DATA] ✅ OK: {path} má {len(df)} řádků a {len(df.columns)} sloupců.")
    return True


def check_sets(sets) -> bool:
    ok = True
    print("\n=== KONTROLA SETŮ (L1) ===")
    for s in sets:
        has_xgb = os.path.exists(os.path.join(s, "xgb.model"))
        has_lstm = os.path.exists(os.path.join(s, "lstm.pt"))
        has_hrm = os.path.exists(os.path.join(s, "hrm.pt"))
        rawC = int(has_xgb) + int(has_lstm) + int(has_hrm)

        meta_path = os.path.join(s, "hrm_meta.pt")
        if not os.path.exists(meta_path):
            print(f"[SET] ⚠️  {s} nemá hrm_meta.pt (rawC={rawC})")
            ok = False
            continue

        try:
            meta = torch.load(meta_path, map_location="cpu")
            inf = int(meta.get("in_features") or -1)
            seq = int(meta.get("seq_len") or -1)
        except Exception as e:
            print(f"[SET] ❌ {s}/hrm_meta.pt nelze načíst: {e}")
            ok = False
            continue

        msg = f"[SET] {s:50s} rawC={rawC}  hrm_meta.in_features={inf}  seq_len={seq}"
        if inf != rawC:
            print(msg + "  → ❌ NESOULAD (špatný počet kanálů)")
            ok = False
        else:
            print(msg + "  → ✅ OK")
    return ok


def check_l2_heads(heads) -> bool:
    ok = True
    print("\n=== KONTROLA L2 HEADS ===")
    for h in heads:
        meta_path = os.path.join(h, "meta.json")
        if not os.path.exists(meta_path):
            print(f"[L2] ❌ {meta_path} neexistuje.")
            ok = False
            continue
        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
            mg = meta.get("meta_gate") or {}
            sets = mg.get("sets") or []
            l1 = mg.get("l1_seq_len")
            l2 = mg.get("l2_seq_len")
            use_raw = mg.get("use_raw_in_l2")
        except Exception as e:
            print(f"[L2] ❌ Nelze načíst {meta_path}: {e}")
            ok = False
            continue

        msg = (
            f"[L2] {meta_path:60s} sets_n={len(sets)}  "
            f"l1_seq_len={l1}  l2_seq_len={l2}  use_raw_in_l2={use_raw}"
        )
        if not sets or l1 is None or l2 is None:
            print(msg + "  → ❌ CHYBÍ HODNOTY")
            ok = False
        else:
            print(msg + "  → ✅ OK")
    return ok


def main():
    print("=== QUICK L3 CHECK (Python verze) ===\n")
    ok_data = check_data_file(DATA_FILE)
    ok_sets = check_sets(SETS)
    ok_heads = check_l2_heads(L2_HEADS)
    print("\n=== SHRNUTÍ ===")
    if all([ok_data, ok_sets, ok_heads]):
        print("✅ Všechno vypadá konzistentně, můžeš spustit L3 trénink.")
    else:
        print("❌ Něco nesedí – zkontroluj výpis výše.")


if __name__ == "__main__":
    main()