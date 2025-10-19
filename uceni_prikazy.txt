
  cd /Users/martinoch/Desktop/mib_project/eth-rt-predictor
  python3 -m venv .venv
 
  source .venv/bin/activate
  pip install -U pip
  pip install -r requirements.txt


  # volitelné zabezpečení + výkon
  export BASIC_AUTH_USER=admin
  export BASIC_AUTH_PASS=supersecret
  
  

//////////////////////////////////////////////////////
cd /Users/martinoch/Desktop/mib_project/eth-rt-predictor
source .venv/bin/activate
python -m uvicorn app.main:app --host 0.0.0.0 --port 8080

/////////////////////////////////////////////////////////////
cd /Users/martinoch/Desktop/mib_project/eth-rt-predictor
source .venv/bin/activate
  
ulimit -n 4096
export OMP_NUM_THREADS=8
export VECLIB_MAXIMUM_THREADS=8
export MKL_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
export TORCH_NUM_THREADS=8

mkdir -p app/models/weights/{1s,5s,10s,15s,30s,60s,180s}


> Pozn.: `--save-features` nepoužívej na víc horizonů (přepisuje stejný soubor). Když chceš uložit, dej ho jen k **jednomu** z příkazů.

### 1) tréninky per-horizont (s OOF, HRM, metou, kalibrací atd.)

#### 1s
python -m app.models.train_offline \
  --symbol ETHUSDT --days 14 --resume \
  --horizon 1 --seq-len 30 \
  --epochs 12 --valid-frac 0.2 --oof-splits 5 \
  --deadzone-eps 0.0005 --deadzone-weight 0.2 \
  --focal-gamma 1.5 \
  --use-hrm --hrm-high-period 10 --hrm-hidden-low 64 --hrm-hidden-high 64 \
  --train-meta --meta-seq-len 60 --meta-epochs 12 --meta-high-period 10 \
  --thr-min-margin 0.015 --thr-min-act 0.10 --thr-min-abstain 0.10 \
  --outdir app/models/weights/1s

#### 5s
python -m app.models.train_offline \
  --symbol ETHUSDT --days 14 --resume \
  --horizon 5 --seq-len 45 \
  --epochs 12 --valid-frac 0.2 --oof-splits 5 \
  --deadzone-eps 0.0005 --deadzone-weight 0.2 \
  --focal-gamma 1.5 \
  --use-hrm --hrm-high-period 10 --hrm-hidden-low 64 --hrm-hidden-high 64 \
  --train-meta --meta-seq-len 60 --meta-epochs 12 --meta-high-period 10 \
  --thr-min-margin 0.015 --thr-min-act 0.10 --thr-min-abstain 0.10 \
  --outdir app/models/weights/5s

#### 10s  *(sem klidně přidej `--save-features`, pokud chceš mít uložené featury; ale jen sem)*
python -m app.models.train_offline \
  --symbol ETHUSDT --days 14 --resume \
  --horizon 10 --seq-len 60 \
  --epochs 12 --valid-frac 0.2 --oof-splits 5 \
  --deadzone-eps 0.0005 --deadzone-weight 0.2 \
  --focal-gamma 1.5 \
  --use-hrm --hrm-high-period 10 --hrm-hidden-low 64 --hrm-hidden-high 64 \
  --train-meta --meta-seq-len 60 --meta-epochs 12 --meta-high-period 10 \
  --thr-min-margin 0.015 --thr-min-act 0.10 --thr-min-abstain 0.10 \
  --outdir app/models/weights/10s \
  --save-features

#### 15s
python -m app.models.train_offline \
  --symbol ETHUSDT --days 14 --resume \
  --horizon 15 --seq-len 90 \
  --epochs 12 --valid-frac 0.2 --oof-splits 5 \
  --deadzone-eps 0.0005 --deadzone-weight 0.2 \
  --focal-gamma 1.5 \
  --use-hrm --hrm-high-period 10 --hrm-hidden-low 64 --hrm-hidden-high 64 \
  --train-meta --meta-seq-len 60 --meta-epochs 12 --meta-high-period 10 \
  --thr-min-margin 0.015 --thr-min-act 0.10 --thr-min-abstain 0.10 \
  --outdir app/models/weights/15s

#### 30s
python -m app.models.train_offline \
  --symbol ETHUSDT --days 14 --resume \
  --horizon 30 --seq-len 120 \
  --epochs 12 --valid-frac 0.2 --oof-splits 5 \
  --deadzone-eps 0.0005 --deadzone-weight 0.2 \
  --focal-gamma 1.5 \
  --use-hrm --hrm-high-period 10 --hrm-hidden-low 64 --hrm-hidden-high 64 \
  --train-meta --meta-seq-len 60 --meta-epochs 12 --meta-high-period 10 \
  --thr-min-margin 0.015 --thr-min-act 0.10 --thr-min-abstain 0.10 \
  --outdir app/models/weights/30s

#### 60s
python -m app.models.train_offline \
  --symbol ETHUSDT --days 14 --resume \
  --horizon 60 --seq-len 180 \
  --epochs 12 --valid-frac 0.2 --oof-splits 5 \
  --deadzone-eps 0.0005 --deadzone-weight 0.2 \
  --focal-gamma 1.5 \
  --use-hrm --hrm-high-period 10 --hrm-hidden-low 64 --hrm-hidden-high 64 \
  --train-meta --meta-seq-len 60 --meta-epochs 12 --meta-high-period 10 \
  --thr-min-margin 0.015 --thr-min-act 0.10 --thr-min-abstain 0.10 \
  --outdir app/models/weights/60s


#### 180s
python -m app.models.train_offline \
  --symbol ETHUSDT --days 14 --resume \
  --horizon 180 --seq-len 300 \
  --epochs 12 --valid-frac 0.2 --oof-splits 5 \
  --deadzone-eps 0.0005 --deadzone-weight 0.2 \
  --focal-gamma 1.5 \
  --use-hrm --hrm-high-period 10 --hrm-hidden-low 64 --hrm-hidden-high 64 \
  --train-meta --meta-seq-len 60 --meta-epochs 12 --meta-high-period 10 \
  --thr-min-margin 0.015 --thr-min-act 0.10 --thr-min-abstain 0.10 \
  --outdir app/models/weights/180s


### 2) dvouvrstvá META-gate 


python -m app.models.train_meta_gate \
  --target-set app/models/weights/1s \
  --sets app/models/weights/1s app/models/weights/5s app/models/weights/10s app/models/weights/15s app/models/weights/30s app/models/weights/60s app/models/weights/180s \
  --l1-seq-len 60 --l2-seq-len 60 \
  --epochs 12 --batch 256 \
  --use-raw-in-l2 \
  --hrm-high-period 10 --hrm-hidden-low 64 --hrm-hidden-high 64 \
  --thr-min-margin 0.015 --thr-min-act 0.10 --thr-min-abstain 0.10 \
  --platt-l2
  
python -m app.models.train_meta_gate \
  --target-set app/models/weights/10s \
  --sets app/models/weights/1s app/models/weights/5s app/models/weights/10s app/models/weights/15s app/models/weights/30s app/models/weights/60s app/models/weights/180s \
  --l1-seq-len 60 --l2-seq-len 60 \
  --epochs 24 --batch 256 \
  --use-raw-in-l2 \
  --hrm-high-period 10 --hrm-hidden-low 64 --hrm-hidden-high 64 \
  --thr-min-margin 0.015 --thr-min-act 0.10 --thr-min-abstain 0.10 \
  --platt-l2
  
  
python -m app.models.train_meta_gate \
  --target-set app/models/weights/15s \
  --sets app/models/weights/1s app/models/weights/5s app/models/weights/10s app/models/weights/15s app/models/weights/30s app/models/weights/60s app/models/weights/180s \
  --l1-seq-len 60 --l2-seq-len 60 \
  --epochs 12 --batch 256 \
  --use-raw-in-l2 \
  --hrm-high-period 10 --hrm-hidden-low 64 --hrm-hidden-high 64 \
  --thr-min-margin 0.015 --thr-min-act 0.10 --thr-min-abstain 0.10 \
  --platt-l2
  
 python -m app.models.train_meta_gate \
   --target-set app/models/weights/30s \
   --sets app/models/weights/1s app/models/weights/5s app/models/weights/10s app/models/weights/15s app/models/weights/30s app/models/weights/60s app/models/weights/180s \
   --l1-seq-len 60 --l2-seq-len 60 \
   --epochs 12 --batch 256 \
   --use-raw-in-l2 \
   --hrm-high-period 10 --hrm-hidden-low 64 --hrm-hidden-high 64 \
   --thr-min-margin 0.015 --thr-min-act 0.10 --thr-min-abstain 0.10 \
   --platt-l2
   
python -m app.models.train_meta_gate \
  --target-set app/models/weights/60s \
  --sets app/models/weights/1s app/models/weights/5s app/models/weights/10s app/models/weights/15s app/models/weights/30s app/models/weights/60s app/models/weights/180s \
  --l1-seq-len 60 --l2-seq-len 60 \
  --epochs 12 --batch 256 \
  --use-raw-in-l2 \
  --hrm-high-period 10 --hrm-hidden-low 64 --hrm-hidden-high 64 \
  --thr-min-margin 0.015 --thr-min-act 0.10 --thr-min-abstain 0.10 \
  --platt-l2
  

python -m app.models.train_meta_gate \
   --target-set app/models/weights/180s \
   --sets app/models/weights/1s app/models/weights/5s app/models/weights/10s app/models/weights/15s app/models/weights/30s app/models/weights/60s app/models/weights/180s \
   --l1-seq-len 60 --l2-seq-len 60 \
   --epochs 12 --batch 256 \
   --use-raw-in-l2 \
   --hrm-high-period 10 --hrm-hidden-low 64 --hrm-hidden-high 64 \
   --thr-min-margin 0.015 --thr-min-act 0.10 --thr-min-abstain 0.10 \
   --platt-l2
   

# L3 trénink

python create_parquet1.py
python create_parquet.py
python fix_meta.py
///

python -m app.models.train_supervisor \
  --data app/data/train_1s.parquet \
  --target-set app/models/weights/10s \
  --sets app/models/weights/1s app/models/weights/5s app/models/weights/10s app/models/weights/15s app/models/weights/30s app/models/weights/60s app/models/weights/180s \
  --l2-heads app/models/weights/1s app/models/weights/10s app/models/weights/15s app/models/weights/30s app/models/weights/60s app/models/weights/180s \
  --use_raw_in_l3 --use-l1-in-l3 --use-l2-in-l3 \
  --l1-seq-len 60 --l3-seq-len 60 \
  --epochs 12 --batch 256 --lr 8e-4 \
  --hrm-hidden-low 96 --hrm-hidden-high 96 --hrm-high-period 12 \
  --valid_frac 0.20 \
  --platt-l3 \
  --thr-up 0.58 --thr-down 0.42 --thr-margin 0.02

Po doběhu máš v cílové složce supervisor_L3.pt + supervisor.json (+ volitelně supervisor_L3.calib.pkl).
V běžící appce pak stačí /action → reload_models (nebo přes UI tlačítko „Načíst modely“).


# diagnostic

(.venv) (base) martinoch@MacBook-Pro eth-rt-predictor %
python diag_l3_quick.py
python diag_l3.py
