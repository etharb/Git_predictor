# app/main.py
# -*- coding: utf-8 -*-
import asyncio
import os
import time
import hmac
import hashlib
import json
from collections import deque
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd
from fastapi import FastAPI, Response, Request, WebSocket, WebSocketDisconnect, Depends, HTTPException, status
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel
from dotenv import load_dotenv
import aiohttp

from .utils.stream_binance import ohlcv_1s_aggregator
from .utils.features import compute_features
from .models.ensemble import EnsemblePredictor

load_dotenv()

def env_float(k, dv):
    try:
        return float(os.getenv(k, str(dv)))
    except Exception:
        return dv

SYMBOL = os.getenv('SYMBOL', 'ETHUSDT')
LOOKBACK_SECONDS = int(os.getenv('LOOKBACK_SECONDS', '1200'))
MIN_BOOTSTRAP_SECONDS = int(os.getenv('MIN_BOOTSTRAP_SECONDS', '180'))

# Základní prahy (mohou být přepsány meta.json + dynamikou)
CONF_ENTER_UP = env_float('CONF_ENTER_UP', 0.58)
CONF_ENTER_DOWN = env_float('CONF_ENTER_DOWN', 0.42)
ABSTAIN_MARGIN = env_float('ABSTAIN_MARGIN', 0.02)

# Hystereze / cooldown
HYSTERESIS = env_float('HYSTERESIS', 0.01)
COOLDOWN_SEC = int(os.getenv('COOLDOWN_SEC', '10'))

# Konsensus (vážený)
CONSENSUS_TAGS = os.getenv('CONSENSUS_TAGS', '').strip()
CONSENSUS_DIRS = os.getenv('CONSENSUS_DIRS', '').strip()
CONSENSUS_WEIGHTS = os.getenv('CONSENSUS_WEIGHTS', '').strip()  # např. "1.0,0.8,0.6" nebo prázdné → auto=1

# Dynamické prahy podle volatility (zap/vyp a koeficienty)
DYN_THR_ENABLE = os.getenv('DYN_THR_ENABLE', '1').strip() == '1'
# baseline_sigma je odhad z 15min okna; koeficient říká jak moc roztahovat margin
DYN_THR_K_MARGIN = env_float('DYN_THR_K_MARGIN', 1.25)     # násobek (sigma / baseline_sigma)
DYN_THR_K_UPDOWN = env_float('DYN_THR_K_UPDOWN', 0.50)     # jak posouvat up/down od 0.5

# L3 abstain/uncert gating (volitelné)
USE_L3_ABSTAIN = os.getenv('USE_L3_ABSTAIN', '0').strip() == '1'
L3_ABSTAIN_THRESHOLD = env_float('L3_ABSTAIN_THRESHOLD', 0.60)
L3_UNCERT_THRESHOLD = env_float('L3_UNCERT_THRESHOLD', 0.60)

# Jednoduchý contextual bandit nad "execute/skip"
POLICY_BANDIT_ENABLE = os.getenv('POLICY_BANDIT_ENABLE', '1').strip() == '1'
BANDIT_MIN_OBS = int(os.getenv('BANDIT_MIN_OBS', '200'))   # od kdy spustit rozhodování banditem
BANDIT_ALPHA0 = env_float('BANDIT_ALPHA0', 1.0)            # Beta prior
BANDIT_BETA0  = env_float('BANDIT_BETA0', 1.0)

BASIC_USER = os.getenv('BASIC_AUTH_USER', '').strip()
BASIC_PASS = os.getenv('BASIC_AUTH_PASS', '').strip()

WEBHOOK_URL = os.getenv('WEBHOOK_URL', '').strip()
WEBHOOK_SECRET = os.getenv('WEBHOOK_SECRET', '').encode() if os.getenv('WEBHOOK_SECRET') else None

BASE_DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.join(BASE_DIR, "models")
WEIGHTS_BASE = os.path.join(MODELS_DIR, "weights")
DATA_DIR = os.path.join(BASE_DIR, 'data')
os.makedirs(DATA_DIR, exist_ok=True)

app = FastAPI(title='ETH Real-Time Predictor', version='1.6')
security = HTTPBasic()

def require_auth(creds: HTTPBasicCredentials = Depends(security)):
    if not BASIC_USER:
        return
    if not (creds.username == BASIC_USER and creds.password == BASIC_PASS):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized",
                            headers={"WWW-Authenticate": "Basic"})
    return True

class Signal(BaseModel):
    ts: int
    price: float
    prob_up: Optional[float] = None
    decision: Optional[str] = None
    # L1/L2/L3 detail
    p_xgb: Optional[float] = None
    p_lstm: Optional[float] = None
    p_hrm: Optional[float] = None
    p_meta: Optional[float] = None
    p_l2: Optional[float] = None
    p_l3: Optional[float] = None
    # L3 multi-heads
    p_abstain: Optional[float] = None
    p_uncert: Optional[float] = None
    p3: Optional[List[float]] = None
    hsel: Optional[List[float]] = None
    # ensemble
    p_ens: Optional[float] = None

class Outcome(BaseModel):
    ts_pred: int
    ts_out: int
    side: str
    price_pred: float
    price_out: float
    win: bool
    delta: float

# Stav
state = {
    'df': None,
    'df_feat': None,
    'last_signal': None,
    'predictor': None,
    'bootstrapped': False,
    'history': deque(maxlen=3600),
    'stream_task': None,
    'lock': asyncio.Lock(),
    'symbol': SYMBOL,
    'running': False,
    'subs': [],
    'last_logged_ts': None,
    'metrics': {
        'bars_total': 0,
        'signals_total': 0,
        'long_total': 0,
        'short_total': 0,
        'abstain_total': 0
    },
    # policy (hysterese/cooldown)
    'last_decision': None,
    'last_decision_ts': None,
    # outcomes
    'horizon_sec': None,
    'pending_preds': deque(maxlen=10000),
    'outcome_by_pred_ts': {},
    'recent_outcomes': deque(maxlen=2000),
    # model set
    'weights_dir': None,
    # consensus
    'consensus_predictors': [],      # list[(name, EnsemblePredictor)]
    'consensus_weights': [],         # list[float] stejná délka
    # bandit stav pro execute/skip
    'bandit': {
        # dvě "akce": 0=SKIP, 1=EXECUTE
        'alpha': [BANDIT_ALPHA0, BANDIT_ALPHA0],
        'beta':  [BANDIT_BETA0,  BANDIT_BETA0],
        'n_obs': 0
    },
    # baseline sigma pro dynamické prahy
    'baseline_sigma': None,
}

# --------- Pomocné ----------

def _list_modelsets() -> List[dict]:
    out = []
    if not os.path.isdir(WEIGHTS_BASE):
        return out
    root_files = set(os.listdir(WEIGHTS_BASE))
    if {'xgb.model', 'lstm.pt', 'hrm.pt', 'hrm_meta.pt'} & root_files:
        out.append({"name": "default", "path": WEIGHTS_BASE})
    for d in sorted(os.listdir(WEIGHTS_BASE)):
        p = os.path.join(WEIGHTS_BASE, d)
        if not os.path.isdir(p): continue
        files = set(os.listdir(p))
        if {'xgb.model','lstm.pt','hrm.pt','hrm_meta.pt'} & files:
            out.append({"name": d, "path": p})
    return out

def _load_predictor(weights_dir: Optional[str]):
    ep = EnsemblePredictor(weights_dir=weights_dir)
    state['predictor'] = ep
    state['horizon_sec'] = int(getattr(ep, 'horizon_sec', 1) or 1)

def _parse_weights(s: str, n: int) -> List[float]:
    if not s: return [1.0]*n
    try:
        vals = [float(x) for x in s.split(',') if x.strip()!='']
        if len(vals) != n: return [1.0]*n
        return vals
    except Exception:
        return [1.0]*n

def _load_consensus_predictors():
    state['consensus_predictors'].clear()
    tags = [t.strip() for t in CONSENSUS_TAGS.split(',') if t.strip()]
    dirs = [d.strip() for d in CONSENSUS_DIRS.split(',') if d.strip()]

    for tg in tags:
        p = os.path.join(WEIGHTS_BASE, tg)
        if os.path.isdir(p):
            try:
                ep = EnsemblePredictor(weights_dir=p)
                if ep.ready(): state['consensus_predictors'].append((f"tag:{tg}", ep))
            except Exception as e:
                print(f"[CONS] skip {p}: {e}")

    for d in dirs:
        if os.path.isdir(d):
            try:
                ep = EnsemblePredictor(weights_dir=d)
                if ep.ready(): state['consensus_predictors'].append((f"dir:{d}", ep))
            except Exception as e:
                print(f"[CONS] skip {d}: {e}")

    state['consensus_weights'] = _parse_weights(CONSENSUS_WEIGHTS, len(state['consensus_predictors']))
    if state['consensus_predictors']:
        print(f"[CONS] loaded: {[n for n,_ in state['consensus_predictors']]}  weights={state['consensus_weights']}")
    else:
        print("[CONS] none configured")

def _rolling_sigma(prices: pd.Series, win: int = 900) -> Optional[float]:
    """odhad σ z 15min okna (900s) – relative returns std"""
    if prices is None or len(prices) < win+1: return None
    r = prices.diff().iloc[-win:] / (prices.shift(1).iloc[-win:] + 1e-12)
    return float(np.nanstd(r.values))

def _dynamic_thresholds(prob_up: float) -> Tuple[float, float, float]:
    """
    Vrátí (thr_up, thr_down, margin) dynamicky upravené podle volatility.
    """
    if not DYN_THR_ENABLE or state['df'] is None:
        return CONF_ENTER_UP, CONF_ENTER_DOWN, ABSTAIN_MARGIN

    sigma = _rolling_sigma(state['df']['close'])
    if sigma is None:
        return CONF_ENTER_UP, CONF_ENTER_DOWN, ABSTAIN_MARGIN

    # baseline si nastavíme jednorázově při startu (nebo při prvním výpočtu)
    if state['baseline_sigma'] is None:
        state['baseline_sigma'] = sigma if sigma > 0 else 1e-6

    # koeficient = jak moc je aktuální vol > baseline
    k = max(0.5, min(2.0, sigma / (state['baseline_sigma'] + 1e-12)))

    # margin roztahujeme s volatilitou
    margin = max(0.0, min(0.20, ABSTAIN_MARGIN * (1.0 + DYN_THR_K_MARGIN*(k-1.0))))

    # up/down posuneme blíž/od 0.5 v závislosti na volatilitě
    up   = 0.5 + (CONF_ENTER_UP - 0.5) * (1.0 + DYN_THR_K_UPDOWN*(k-1.0))
    down = 0.5 - (0.5 - CONF_ENTER_DOWN) * (1.0 + DYN_THR_K_UPDOWN*(k-1.0))

    up   = max(0.0, min(1.0, up))
    down = max(0.0, min(1.0, down))
    margin = max(0.0, min(0.20, margin))
    return up, down, margin

# --------- Rozhodování ---------

def _bandit_choose(p_up: float) -> bool:
    """
    Jednoduchý Thompson Sampling nad akcí EXECUTE (1) vs SKIP (0).
    Kritérium "úspěchu" definujeme jako správný směr (validovaný outcome).
    Samotné update probíhá při resolve_outcomes (níže).
    """
    if not POLICY_BANDIT_ENABLE: return True
    b = state['bandit']
    if b['n_obs'] < BANDIT_MIN_OBS:
        return True
    # vzorek z Beta distribucí
    sample_exec = np.random.beta(b['alpha'][1], b['beta'][1])
    sample_skip = np.random.beta(b['alpha'][0], b['beta'][0])
    return (sample_exec >= sample_skip)

def _consensus_vote(base_decision: str, df_feat: Optional[pd.DataFrame], thr: Tuple[float,float,float]) -> str:
    """
    Vážený konsensus: každý prediktor dává hlas LONG/SHORT/ABSTAIN,
    hlasy (kromě ABSTAIN) agregujeme váhami → argmax.
    """
    if not state['consensus_predictors'] or df_feat is None or len(df_feat)==0:
        return base_decision

    thr_up, thr_down, margin = thr
    votes = {'LONG': 0.0, 'SHORT': 0.0}

    # základní rozhodnutí přidej s vahou 1.0, ale jen pokud není ABSTAIN
    if base_decision in ('LONG','SHORT'):
        votes[base_decision] += 1.0

    for (name, ep), w in zip(state['consensus_predictors'], state['consensus_weights']):
        try:
            d = ep.predict_detail(df_feat)
            p = float(d.get('p_ens', d.get('prob_up', 0.5)) or 0.5)
            v = 'ABSTAIN'
            if p >= (thr_up + margin): v='LONG'
            elif p <= (thr_down - margin): v='SHORT'
            if v in votes:
                votes[v] += float(w)
        except Exception as e:
            print(f"[CONS] {name} error: {e}")

    if votes['LONG'] > votes['SHORT']:
        return 'LONG'
    if votes['SHORT'] > votes['LONG']:
        return 'SHORT'
    return 'ABSTAIN'

def decide(prob_up: float, now_ts: int, df_feat: Optional[pd.DataFrame] = None) -> str:
    # Dynamické prahy
    thr_up, thr_down, margin = _dynamic_thresholds(prob_up)

    # Základní rozhodnutí
    base = 'ABSTAIN'
    if prob_up >= (thr_up + margin):
        base = 'LONG'
    elif prob_up <= (thr_down - margin):
        base = 'SHORT'

    # Hystereze
    last = state.get('last_decision')
    last_ts = state.get('last_decision_ts') or 0
    if last in ('LONG','SHORT'):
        if last == 'LONG' and prob_up >= max(0.0, thr_up - HYSTERESIS):
            base = 'LONG'
        elif last == 'SHORT' and prob_up <= min(1.0, thr_down + HYSTERESIS):
            base = 'SHORT'

    # Cooldown
    if last in ('LONG','SHORT') and now_ts - last_ts < COOLDOWN_SEC:
        if base != last:
            base = 'ABSTAIN'

    # L3 abstain/uncert gating
    if USE_L3_ABSTAIN and state['predictor'] and state['predictor'].ready() and df_feat is not None:
        try:
            d = state['predictor'].predict_detail(df_feat)
            p_abs = float(d.get('p_abstain', 0.0) or 0.0)
            p_unc = float(d.get('p_uncert', 0.0) or 0.0)
            if p_abs >= L3_ABSTAIN_THRESHOLD or p_unc >= L3_UNCERT_THRESHOLD:
                base = 'ABSTAIN'
        except Exception:
            pass

    # Vážený konsensus
    final_dec = _consensus_vote(base, df_feat, (thr_up, thr_down, margin))

    # Bandit execute/skip (jen pokud je signál)
    if final_dec in ('LONG','SHORT'):
        if not _bandit_choose(prob_up):
            final_dec = 'ABSTAIN'

    # stav
    if final_dec in ('LONG','SHORT') and final_dec != last:
        state['last_decision'] = final_dec
        state['last_decision_ts'] = now_ts
    return final_dec

# --------- Webhook / Broadcast / CSV ----------

def csv_path_for(ts: int) -> str:
    day = time.strftime('%Y%m%d', time.gmtime(ts))
    return os.path.join(DATA_DIR, f'signals_{day}.csv')

async def broadcast(event: dict):
    for q in list(state['subs']):
        try: q.put_nowait(event)
        except Exception: pass

async def post_webhook(sig: Signal):
    if not WEBHOOK_URL: return
    payload = sig.model_dump()
    body = (str(payload)).encode()
    headers = {}
    if WEBHOOK_SECRET:
        digest = hmac.new(WEBHOOK_SECRET, body, hashlib.sha256).hexdigest()
        headers['X-Signature'] = digest
    try:
        async with aiohttp.ClientSession() as sess:
            async with sess.post(WEBHOOK_URL, json=payload, headers=headers, timeout=5) as resp:
                await resp.text()
    except Exception as e:
        print(f"[WEBHOOK] failed: {e}")

# --------- Outcomes & Bandit update ----------

def _resolve_outcomes_if_possible(latest_ts: int):
    if state['df'] is None: return
    df = state['df']
    resolved = []
    for item in list(state['pending_preds']):
        ts_out = item['resolve_ts']
        if ts_out > latest_ts: break
        if ts_out not in df.index: continue
        price_out = float(df.loc[ts_out, 'close'])
        side = item['side']
        price_pred = float(item['price'])
        win = (price_out > price_pred) if side == 'LONG' else (price_out < price_pred)
        delta = price_out - price_pred
        oc = {
            'ts_pred': int(item['ts']),
            'ts_out': int(ts_out),
            'side': side,
            'price_pred': price_pred,
            'price_out': price_out,
            'win': bool(win),
            'delta': float(delta),
        }
        state['outcome_by_pred_ts'][item['ts']] = oc
        state['recent_outcomes'].append(oc)
        resolved.append(item)

        # Bandit update
        b = state['bandit']
        # reward = 1 pokud win, jinak 0. Update EXECUTE akce.
        if win:
            b['alpha'][1] += 1.0
        else:
            b['beta'][1]  += 1.0
        b['n_obs'] += 1

    for it in resolved:
        try: state['pending_preds'].remove(it)
        except ValueError: pass
        asyncio.create_task(broadcast({'type': 'outcome', 'data': it | state['outcome_by_pred_ts'][it['ts']]}))

# --------- Stream ----------

async def stream_worker(symbol: str):
    print(f"[STREAM] Connecting Binance for {symbol} (1s trades → OHLCV)...")
    from collections import deque as _deque
    bars = _deque(maxlen=LOOKBACK_SECONDS + 5)
    try:
        async for bar in ohlcv_1s_aggregator(symbol, lookback_seconds=LOOKBACK_SECONDS):
            bars.append(bar)
            state['metrics']['bars_total'] += 1

            df = pd.DataFrame(bars).set_index('ts').sort_index()
            state['df'] = df

            if len(df) >= MIN_BOOTSTRAP_SECONDS:
                df_feat = compute_features(df)
                state['df_feat'] = df_feat
                state['bootstrapped'] = True

                detail = None
                p_ens = None
                if state['predictor'] and state['predictor'].ready():
                    detail = state['predictor'].predict_detail(df_feat)
                    p_ens = float(detail.get('p_ens', 0.5))

                last_ts = int(df.index[-1])
                last_price = float(df['close'].iloc[-1])

                # L3 multi-heady (pokud jsou)
                p_l2 = detail.get('p_l2') if detail else None
                p_l3 = detail.get('p_l3') if detail else None
                p_abstain = detail.get('p_abstain') if detail else None
                p_uncert = detail.get('p_uncert') if detail else None
                p3 = detail.get('p3') if detail else None
                hsel = detail.get('hsel') if detail else None

                decision = decide(p_ens, last_ts, df_feat=df_feat) if p_ens is not None else None

                sig = Signal(
                    ts=last_ts,
                    price=last_price,
                    prob_up=(p_ens if p_ens is not None else None),
                    decision=decision,
                    p_xgb=detail.get('p_xgb') if detail else None,
                    p_lstm=detail.get('p_lstm') if detail else None,
                    p_hrm=detail.get('p_hrm') if detail else None,
                    p_meta=detail.get('p_meta') if detail else None,
                    p_l2=p_l2, p_l3=p_l3,
                    p_abstain=p_abstain, p_uncert=p_uncert, p3=p3, hsel=hsel,
                    p_ens=p_ens,
                )
                state['last_signal'] = sig
                state['history'].append(sig.model_dump())

                state['metrics']['signals_total'] += 1
                if decision == 'LONG': state['metrics']['long_total'] += 1
                elif decision == 'SHORT': state['metrics']['short_total'] += 1
                else: state['metrics']['abstain_total'] += 1

                hz = state['horizon_sec'] or 1
                if decision in ('LONG', 'SHORT'):
                    state['pending_preds'].append({
                        'ts': last_ts, 'price': last_price, 'side': decision, 'resolve_ts': last_ts + hz,
                    })

                if state['last_logged_ts'] != last_ts:
                    csv_path = csv_path_for(last_ts)
                    new_file = not os.path.exists(csv_path)
                    line = f"{last_ts},{last_price},{(p_ens if p_ens is not None else '')},{decision or ''}\n"
                    with open(csv_path, 'a') as f:
                        if new_file: f.write('ts,price,prob_up,decision\n')
                        f.write(line)
                    state['last_logged_ts'] = last_ts

                await broadcast({'type':'signal', 'data': sig.model_dump()})
                if last_ts % 5 == 0:
                    print(f"[SIG] t={last_ts} price={last_price:.2f} p_up={p_ens if p_ens is not None else '—'} -> {decision}")

                if decision in ('LONG', 'SHORT'):
                    await post_webhook(sig)

                _resolve_outcomes_if_possible(last_ts)

    except asyncio.CancelledError:
        print("[STREAM] cancelled"); raise
    except Exception as e:
        print(f"[STREAM] error: {e}")
        await asyncio.sleep(1.0)

async def start_stream():
    async with state['lock']:
        if state['stream_task'] and not state['stream_task'].done(): return
        state['running'] = True
        state['stream_task'] = asyncio.create_task(stream_worker(state['symbol']))

async def stop_stream():
    async with state['lock']:
        if state['stream_task'] and not state['stream_task'].done():
            state['stream_task'].cancel()
            try: await state['stream_task']
            except asyncio.CancelledError: pass
        state['running'] = False
        state['stream_task'] = None

# --------- Meta prahy (meta.json) ----------

def _apply_meta_thresholds_from(weights_dir: str):
    global CONF_ENTER_UP, CONF_ENTER_DOWN, ABSTAIN_MARGIN
    meta_path = os.path.join(weights_dir or "", "meta.json")
    try:
        with open(meta_path, "r") as f:
            m = json.load(f)
        rec = m.get("recommended") or m.get("recommended_thresholds")
        if rec is None and "auto_thr" in m:
            at = m["auto_thr"]
            rec = {"CONF_ENTER_UP": at.get("up"), "CONF_ENTER_DOWN": at.get("down"), "ABSTAIN_MARGIN": at.get("margin")}
        if not rec:
            raise KeyError("recommended thresholds not found in meta.json")
        up = float(rec.get("CONF_ENTER_UP", CONF_ENTER_UP))
        dn = float(rec.get("CONF_ENTER_DOWN", CONF_ENTER_DOWN))
        mar = rec.get("ABSTAIN_MARGIN", ABSTAIN_MARGIN)
        try:
            mar = float(mar)
        except Exception:
            mar = abs(up-0.5)
        up = max(0.0, min(1.0, up))
        dn = max(0.0, min(1.0, dn))
        mar = max(0.0, min(0.5, mar))
        CONF_ENTER_UP, CONF_ENTER_DOWN, ABSTAIN_MARGIN = up, dn, mar
        print(f"[META] thresholds loaded from {meta_path}: up={up:.3f} down={dn:.3f} margin={mar:.3f}")
        return {"path": meta_path, "ok": True, "up": up, "down": dn, "margin": mar}
    except Exception as e:
        print(f"[META] thresholds not loaded from {meta_path}: {e}")
        return {"path": meta_path, "ok": False, "error": str(e),
                "up": CONF_ENTER_UP, "down": CONF_ENTER_DOWN, "margin": ABSTAIN_MARGIN}

# --------- FastAPI ----------

@app.on_event("startup")
async def startup_event():
    current = os.getenv("MODEL_DIR", "").strip()
    if not current:
        tag = os.getenv("MODEL_TAG", "").strip()
        if tag: current = os.path.join(WEIGHTS_BASE, tag)
    if not current: current = WEIGHTS_BASE
    state['weights_dir'] = os.path.abspath(current)
    _load_predictor(state['weights_dir'])
    _load_consensus_predictors()
    _apply_meta_thresholds_from(state['weights_dir'])
    if not state['predictor'] or not state['predictor'].ready():
        print("[MODEL] No saved models found in", state['weights_dir'])
    await start_stream()

# Static UI
app.mount('/static', StaticFiles(directory='app/frontend'), name='static')

@app.get('/')
def ui_root():
    return FileResponse('app/frontend/index.html')

@app.get('/health')
def health():
    gate = {}
    pred = state.get('predictor')
    if pred and pred.ready() and hasattr(pred, 'gate_get_state'):
        try: gate = pred.gate_get_state()
        except Exception: gate = {}
    thr_up, thr_down, margin = _dynamic_thresholds(0.5)  # snapshot
    return {
        'symbol': state['symbol'],
        'bootstrapped': state['bootstrapped'],
        'models_loaded': bool(pred and pred.ready()),
        'have_signal': state['last_signal'] is not None,
        'running': state['running'],
        'horizon_sec': int(state['horizon_sec'] or 1),
        'weights_dir': state['weights_dir'],
        # thresholds (aktuální – už po dynamice)
        'CONF_ENTER_UP': thr_up,
        'CONF_ENTER_DOWN': thr_down,
        'ABSTAIN_MARGIN': margin,
        # aliasy
        'thr_up': thr_up, 'thr_down': thr_down, 'margin': margin,
        # L3 gating info
        'USE_L3_ABSTAIN': USE_L3_ABSTAIN,
        'L3_ABSTAIN_THRESHOLD': L3_ABSTAIN_THRESHOLD,
        'L3_UNCERT_THRESHOLD': L3_UNCERT_THRESHOLD,
        # gate snapshot
        'gate': gate,
    }

@app.get('/signal', response_model=Signal, dependencies=[Depends(require_auth)])
def get_signal():
    if state['last_signal'] is None:
        return Response(status_code=503, content='Not enough data yet.')
    return state['last_signal']

@app.get('/history', dependencies=[Depends(require_auth)])
def history(limit: int = 600):
    return list(state['history'])[-limit:]

@app.get('/outcomes', dependencies=[Depends(require_auth)])
def outcomes(limit: int = 500):
    return list(state['recent_outcomes'])[-limit:]

@app.get('/config', dependencies=[Depends(require_auth)])
def get_config():
    return {
        'symbol': state['symbol'],
        'CONF_ENTER_UP': CONF_ENTER_UP,
        'CONF_ENTER_DOWN': CONF_ENTER_DOWN,
        'ABSTAIN_MARGIN': ABSTAIN_MARGIN,
        'DYN_THR_ENABLE': DYN_THR_ENABLE,
        'USE_L3_ABSTAIN': USE_L3_ABSTAIN,
        'L3_ABSTAIN_THRESHOLD': L3_ABSTAIN_THRESHOLD,
        'L3_UNCERT_THRESHOLD': L3_UNCERT_THRESHOLD,
        'POLICY_BANDIT_ENABLE': POLICY_BANDIT_ENABLE,
    }

@app.post('/config', dependencies=[Depends(require_auth)])
async def set_config(cfg: Dict):
    global CONF_ENTER_UP, CONF_ENTER_DOWN, ABSTAIN_MARGIN
    global USE_L3_ABSTAIN, L3_ABSTAIN_THRESHOLD, L3_UNCERT_THRESHOLD
    global DYN_THR_ENABLE, POLICY_BANDIT_ENABLE
    changed = []
    if 'SYMBOL' in cfg and isinstance(cfg['SYMBOL'], str):
        sym = cfg['SYMBOL'].upper()
        if sym != state['symbol']:
            state['symbol'] = sym; changed.append('SYMBOL')
            await stop_stream(); await start_stream()
    if 'CONF_ENTER_UP' in cfg:    CONF_ENTER_UP = float(cfg['CONF_ENTER_UP']); changed.append('CONF_ENTER_UP')
    if 'CONF_ENTER_DOWN' in cfg:  CONF_ENTER_DOWN = float(cfg['CONF_ENTER_DOWN']); changed.append('CONF_ENTER_DOWN')
    if 'ABSTAIN_MARGIN' in cfg:   ABSTAIN_MARGIN = float(cfg['ABSTAIN_MARGIN']); changed.append('ABSTAIN_MARGIN')
    if 'DYN_THR_ENABLE' in cfg:   DYN_THR_ENABLE = bool(cfg['DYN_THR_ENABLE']);  changed.append('DYN_THR_ENABLE')

    if 'USE_L3_ABSTAIN' in cfg:   USE_L3_ABSTAIN = bool(cfg['USE_L3_ABSTAIN']); changed.append('USE_L3_ABSTAIN')
    if 'L3_ABSTAIN_THRESHOLD' in cfg:
        L3_ABSTAIN_THRESHOLD = float(cfg['L3_ABSTAIN_THRESHOLD']); changed.append('L3_ABSTAIN_THRESHOLD')
    if 'L3_UNCERT_THRESHOLD' in cfg:
        L3_UNCERT_THRESHOLD = float(cfg['L3_UNCERT_THRESHOLD']); changed.append('L3_UNCERT_THRESHOLD')
    if 'POLICY_BANDIT_ENABLE' in cfg:
        POLICY_BANDIT_ENABLE = bool(cfg['POLICY_BANDIT_ENABLE']); changed.append('POLICY_BANDIT_ENABLE')

    try:
        with open('app/.env.runtime', 'w') as f:
            f.write(f"SYMBOL={state['symbol']}\n"
                    f"CONF_ENTER_UP={CONF_ENTER_UP}\n"
                    f"CONF_ENTER_DOWN={CONF_ENTER_DOWN}\n"
                    f"ABSTAIN_MARGIN={ABSTAIN_MARGIN}\n"
                    f"DYN_THR_ENABLE={int(DYN_THR_ENABLE)}\n"
                    f"USE_L3_ABSTAIN={int(USE_L3_ABSTAIN)}\n"
                    f"L3_ABSTAIN_THRESHOLD={L3_ABSTAIN_THRESHOLD}\n"
                    f"L3_UNCERT_THRESHOLD={L3_UNCERT_THRESHOLD}\n"
                    f"POLICY_BANDIT_ENABLE={int(POLICY_BANDIT_ENABLE)}\n"
                    f"MODEL_DIR={state['weights_dir']}\n")
    except Exception: pass
    return {'status': 'ok', 'changed': changed}

@app.post('/action', dependencies=[Depends(require_auth)])
async def action(body: Dict):
    t = body.get('type')
    if t == 'start': await start_stream(); return {'status':'started'}
    if t == 'stop':  await stop_stream();  return {'status':'stopped'}
    if t == 'reload_models':
        _load_predictor(state['weights_dir'])
        info = _apply_meta_thresholds_from(state['weights_dir'])
        return {'status':'reloaded',
                'models_loaded': bool(state['predictor'] and state['predictor'].ready()),
                'horizon_sec': state['horizon_sec'],
                'thresholds': {'CONF_ENTER_UP': CONF_ENTER_UP, 'CONF_ENTER_DOWN': CONF_ENTER_DOWN, 'ABSTAIN_MARGIN': ABSTAIN_MARGIN,
                               'HYSTERESIS': HYSTERESIS, 'COOLDOWN_SEC': COOLDOWN_SEC},
                'meta_debug': info}
    if t == 'reload_thresholds':
        info = _apply_meta_thresholds_from(state['weights_dir'])
        return {'status':'thresholds_reloaded',
                'thresholds': {'CONF_ENTER_UP': CONF_ENTER_UP, 'CONF_ENTER_DOWN': CONF_ENTER_DOWN, 'ABSTAIN_MARGIN': ABSTAIN_MARGIN,
                               'HYSTERESIS': HYSTERESIS, 'COOLDOWN_SEC': COOLDOWN_SEC,
                               'USE_L3_ABSTAIN': USE_L3_ABSTAIN, 'L3_ABSTAIN_THRESHOLD': L3_ABSTAIN_THRESHOLD,
                               'L3_UNCERT_THRESHOLD': L3_UNCERT_THRESHOLD},
                'meta_debug': info}
    return Response(status_code=400, content='Unknown action')

@app.get('/modelsets', dependencies=[Depends(require_auth)])
def list_modelsets():
    return {'current': state['weights_dir'], 'options': _list_modelsets()}

@app.post('/modelsets', dependencies=[Depends(require_auth)])
async def set_modelset(body: Dict):
    if 'path' in body and body['path']:
        path = os.path.abspath(body['path'])
    elif 'tag' in body and body['tag']:
        path = os.path.join(WEIGHTS_BASE, str(body['tag']))
    else:
        return Response(status_code=400, content="Provide 'path' or 'tag'.")
    if not os.path.isdir(path):
        return Response(status_code=404, content="Model set dir not found.")
    state['weights_dir'] = path
    _load_predictor(state['weights_dir'])
    _apply_meta_thresholds_from(state['weights_dir'])
    return {'status':'ok', 'current': state['weights_dir'], 'horizon_sec': state['horizon_sec'],
            'thresholds': {'CONF_ENTER_UP': CONF_ENTER_UP, 'CONF_ENTER_DOWN': CONF_ENTER_DOWN, 'ABSTAIN_MARGIN': ABSTAIN_MARGIN}}

@app.get('/download/signals', dependencies=[Depends(require_auth)])
def download_signals():
    ts = int(time.time()); path = csv_path_for(ts)
    if os.path.exists(path):
        return FileResponse(path, filename=os.path.basename(path), media_type='text/csv')
    files = sorted([f for f in os.listdir(DATA_DIR) if f.startswith('signals_')])
    if files:
        return FileResponse(os.path.join(DATA_DIR, files[-1]), filename=files[-1], media_type='text/csv')
    return Response(status_code=404, content='No CSV yet')

@app.get('/metrics')
def metrics():
    m = state['metrics']
    lines = [
        '# HELP bars_total Total 1s bars processed',
        '# TYPE bars_total counter',
        f"bars_total {m['bars_total']}",
        '# HELP signals_total Total signals produced',
        '# TYPE signals_total counter',
        f"signals_total {m['signals_total']}",
        '# HELP long_total Total LONG decisions',
        '# TYPE long_total counter',
        f"long_total {m['long_total']}",
        '# HELP short_total Total SHORT decisions',
        '# TYPE short_total counter',
        f"short_total {m['short_total']}",
        '# HELP abstain_total Total ABSTAIN decisions',
        '# TYPE abstain_total counter',
        f"abstain_total {m['abstain_total']}",
    ]
    return PlainTextResponse('\n'.join(lines))

@app.get('/meta', dependencies=[Depends(require_auth)])
def get_meta():
    p = os.path.join(state['weights_dir'] or "", "meta.json")
    out = {'path': p, 'exists': os.path.exists(p)}
    if os.path.exists(p):
        try:
            with open(p, 'r') as f: out['json'] = json.load(f)
        except Exception as e: out['error'] = str(e)
    return out

@app.get('/events', dependencies=[Depends(require_auth)])
async def sse(request: Request):
    from fastapi.responses import StreamingResponse
    q = asyncio.Queue(); state['subs'].append(q)
    async def gen():
        try:
            while True:
                if await request.is_disconnected(): break
                msg = await q.get(); yield "data: " + json.dumps(msg) + "\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            try: state['subs'].remove(q)
            except ValueError: pass
    return StreamingResponse(gen(), media_type='text/event-stream')

@app.websocket('/ws')
async def ws(ws: WebSocket):
    await ws.accept()
    if BASIC_USER:
        try:
            auth = ws.query_params.get('auth')
            if not auth or ':' not in auth: await ws.close(code=1008); return
            u, p = auth.split(':', 1)
            if not (u == BASIC_USER and p == BASIC_PASS): await ws.close(code=1008); return
        except Exception:
            await ws.close(code=1008); return
    q = asyncio.Queue(); state['subs'].append(q)
    try:
        while True:
            msg = await q.get(); await ws.send_json(msg)
    except WebSocketDisconnect:
        pass
    finally:
        try: state['subs'].remove(q)
        except ValueError: pass

# Meta-gate live ladění
@app.get('/gate', dependencies=[Depends(require_auth)])
def get_gate_state():
    pred = state.get('predictor')
    if not pred or not pred.ready():
        return {'ready': False, 'mode': 'auto', 'alpha': 0.3, 'components': [], 'gains': {}, 'last_probs': {}}
    st = pred.gate_get_state() if hasattr(pred, 'gate_get_state') else {}
    st['ready'] = True
    return st

@app.post('/gate', dependencies=[Depends(require_auth)])
def set_gate_state(cfg: Dict):
    pred = state.get('predictor')
    if not pred or not pred.ready():
        return Response(status_code=503, content='Predictor not ready')
    mode  = cfg.get('mode')
    alpha = cfg.get('alpha')
    gains = cfg.get('gains')
    reset = bool(cfg.get('reset', False))
    if hasattr(pred, 'gate_set_state'):
        pred.gate_set_state(mode=mode, alpha=alpha, gains=gains, reset=reset)
    return {'status':'ok', 'state': pred.gate_get_state()}