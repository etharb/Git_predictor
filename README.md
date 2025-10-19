# 🚀 Full Training & Runtime Guide — ETH/USDT HRM Model (All Enhancements)

## 📘 Overview

Tento projekt trénuje vícestupňový model pro **predikci 60s binárních pohybů ETH/USDT** z 1s dat.
Architektura:

- **L1:** HRM model (Harmonic Recurrent Model) — základní predikce pro různé horizonty (1s … 180s)  
- **L2:** Meta-gate — kombinuje L1 predikce napříč horizonty  
- **L3:** Supervisor — učení nad L2 + vol-scaled triple-barrier + cost-aware edge optimalizace  

Vše je kalibrováno (Platt + Isotonic), s adaptivními prahy, hysterézí, cooldownem a volitelným konsensem více sad v runtime.

---

## 🧩 Zahrnuté vylepšení

✅ Triple-Barrier s **volatilně škálovanými prahy (k·σ)**  
✅ Auto-optimalizace prahů z validace (max. očekávaný čistý edge po nákladech)  
✅ Cost-aware edge (fee/slippage v bps)  
✅ Platt **+** Isotonic kalibrace (fallback při selhání jedné)  
✅ Režimové featury (čas dne, σ, objem)  
✅ Fast sanity mód (–> `--fast-sanity`)  
✅ Hysteréze + cooldown v runtime (omezuje přepínání)  
✅ Consensus více sad (např. 1s/10s/15s = 2/3 majority vote)  
✅ Automatické uložení prahů do `meta.json`  
✅ Plně kompatibilní s UI `/health` a tlačítkem „Načíst prahy“

---

## ⚙️ Doporučené hodnoty po horizontech

| Horizon | σ-okno (sec) | kσ (up/dn) | edge_min_bps | fee_bps | slip_bps | Poznámka |
|----------|--------------:|------------:|---------------:|-----------:|-----------:|-----------|
| **1s**   | 90  | 1.0 | 0.6 | 1.0 | 1.0 | rychlý šum → menší kσ |
| **10s**  | 120 | 1.2 | 1.0 | 1.0 | 1.0 | výchozí standard |
| **15s**  | 150 | 1.2 | 1.2 | 1.0 | 1.0 | mírně vyšší hrana |
| **30s**  | 180 | 1.3 | 1.5 | 1.0 | 1.0 | méně signálů, větší jistota |
| **60s**  | 240 | 1.5 | 2.0 | 1.0 | 1.0 | stabilní delší trend |
| **180s** | 360 | 1.7 | 3.0 | 1.0 | 1.5 | dlouhé obchody, větší slip |

---

## 🧱 Příprava prostředí

```bash
cd /Users/martinoch/Desktop/mib_project/eth-predictor

source .venv/bin/activate

ulimit -n 4096
export OMP_NUM_THREADS=8
export VECLIB_MAXIMUM_THREADS=8
export MKL_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
export TORCH_NUM_THREADS=8

mkdir -p app/models/weights/{1s,5s,10s,15s,30s,60s,180s}
```

---

## 🧠 L1 — Základní modely

Trénink pro všechny horizonty (5s slouží jen pro L1/L2, ne pro L3):

```bash
for H in 1 5 10 15 30 60 180; do
  case $H in
    1)   SEQ=30 ;;
    5)   SEQ=45 ;;
    10)  SEQ=60 ;;
    15)  SEQ=90 ;;
    30)  SEQ=120 ;;
    60)  SEQ=180 ;;
    180) SEQ=300 ;;
  esac
  OUT=app/models/weights/${H}s
  python -m app.models.train_offline     --symbol ETHUSDT --days 14 --resume     --horizon $H --seq-len $SEQ     --epochs 12 --valid-frac 0.2 --oof-splits 5     --deadzone-eps 0.0005 --deadzone-weight 0.2     --focal-gamma 1.5     --use-hrm --hrm-high-period 10 --hrm-hidden-low 64 --hrm-hidden-high 64     --train-meta --meta-seq-len 60 --meta-epochs 12 --meta-high-period 10     --thr-min-margin 0.015 --thr-min-act 0.10 --thr-min-abstain 0.10     --save-features     --outdir $OUT
done
```

---

## 🧬 L2 — Meta-Gate

Dvouúrovňová meta nad L1 (bez 5s jako target, ale 5s se používá jako vstup):

```bash
SETS="app/models/weights/1s app/models/weights/5s app/models/weights/10s app/models/weights/15s app/models/weights/30s app/models/weights/60s app/models/weights/180s"

for TGT in 1s 10s 15s 30s 60s 180s; do
  python -m app.models.train_meta_gate     --target-set app/models/weights/$TGT     --sets $SETS     --l1-seq-len 60 --l2-seq-len 60     --epochs 12 --batch 256     --use-raw-in-l2     --hrm-high-period 10 --hrm-hidden-low 64 --hrm-hidden-high 64     --thr-min-margin 0.015 --thr-min-act 0.10 --thr-min-abstain 0.10     --platt-l2
done
```

---

## 🧠 L3 — Supervisor (kompletní režim)

Před spuštěním:
```bash
python create_parquet1.py
python create_parquet.py
python fix_meta.py
```

Poté trénuj všechny L3 (bez 5s jako target):

```bash
DATA=app/data/train_1s.parquet
SETS="app/models/weights/1s app/models/weights/5s app/models/weights/10s app/models/weights/15s app/models/weights/30s app/models/weights/60s app/models/weights/180s"
L2H="app/models/weights/1s app/models/weights/10s app/models/weights/15s app/models/weights/30s app/models/weights/60s app/models/weights/180s"

run_l3 () {
  TGT=$1; SIGW=$2; KS=$3; EDGE=$4; FEE=$5; SLIP=$6
  python -m app.models.train_supervisor     --data $DATA     --target-set app/models/weights/$TGT     --sets $SETS     --l2-heads $L2H     --use_raw_in_l3 --use-l1-in-l3 --use-l2-in-l3 --use-regime-in-l3     --l1-seq-len 60 --l3-seq-len 60     --epochs 12 --batch 256 --lr 8e-4     --hrm-hidden-low 96 --hrm-hidden-high 96 --hrm-high-period 12     --valid_frac 0.20     --triple-barrier --up-k-sigma $KS --dn-k-sigma $KS --sigma-window $SIGW     --edge-min-bps $EDGE     --fee-bps $FEE --slip-bps $SLIP     --platt-l3 --iso-l3
}

run_l3 1s   90  1.0 0.6  1.0 1.0
run_l3 10s  120 1.2 1.0  1.0 1.0
run_l3 15s  150 1.2 1.2  1.0 1.0
run_l3 30s  180 1.3 1.5  1.0 1.0
run_l3 60s  240 1.5 2.0  1.0 1.0
run_l3 180s 360 1.7 3.0  1.0 1.5
```

💡 *Chceš zrychlit druhý běh? Přidej `--fast-sanity`.*

---

## 🖥️ Runtime (Inference Server)

Spustí model s hysterézí, cooldownem a volitelným konsensem (majorita 2/3):

```bash
export MODEL_TAG="10s"
export HYSTERESIS=0.01
export COOLDOWN_SEC=10
export CONSENSUS_TAGS="1s,10s,15s"

uvicorn app.main:app --host 0.0.0.0 --port 8080
```

---

## 🧭 Shrnutí

| Vrstva | Účel | Klíčové volby |
|---------|------|----------------|
| **L1** | Základní HRM predikce (1s … 180s) | `--use-hrm --save-features` |
| **L2** | Meta-kombinace L1 výstupů | `--use-raw-in-l2 --platt-l2` |
| **L3** | Supervisor (volatility TB + kalibrace) | `--triple-barrier --platt-l3 --iso-l3 --auto-thr` |
| **Runtime** | Hysteréze + cooldown + consensus | ENV: `HYSTERESIS`, `COOLDOWN_SEC`, `CONSENSUS_TAGS` |

---

## ✅ Výstup po tréninku

```
app/models/weights/
 ├── 1s/
 │   ├── supervisor.json
 │   ├── supervisor_L3.calib.pkl
 │   ├── meta.json
 │   └── model_epoch_*.pt
 ├── 10s/
 │   ├── ...
 ├── ...
```

---

**Verze dokumentu:** 2025-10  
**Autor pipeline:** MOCH + ChatGPT (GPT-5)
