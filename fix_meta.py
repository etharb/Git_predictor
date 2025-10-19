import os, json
base = os.path.abspath("app/models/weights")
# L2 hlavy, které používáš v --l2-heads
heads = ["1s","10s","15s","30s","60s","180s"]
# L1/RAW sady (stejné jako v --sets)
sets_all = ["1s","5s","10s","15s","30s","60s","180s"]

abs_sets = [os.path.abspath(os.path.join(base, s)) for s in sets_all]

def rd(p):
    try:
        with open(p,"r") as f: return json.load(f)
    except Exception: return {}
def wr(p, obj):
    with open(p,"w") as f: json.dump(obj, f, indent=2)

for h in heads:
    d = os.path.join(base, h)
    meta_path = os.path.join(d, "meta.json")
    m = rd(meta_path)
    mg = m.get("meta_gate") or {}
    # doplníme/napravíme konfiguraci pro L2
    mg["sets"] = abs_sets
    mg["l1_seq_len"] = mg.get("l1_seq_len", 60)
    mg["l2_seq_len"] = mg.get("l2_seq_len", 60)
    mg["use_raw_in_l2"] = bool(mg.get("use_raw_in_l2", True))
    m["meta_gate"] = mg
    wr(meta_path, m)
    print(f"✓ fixed {meta_path}")