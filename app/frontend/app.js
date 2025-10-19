// app/frontend/app.js

const els = {
  // status & základ
  status: document.getElementById('status'),
  symbol: document.getElementById('symbol'),

  // prahy
  confUp: document.getElementById('confUp'),
  confDown: document.getElementById('confDown'),
  abstain: document.getElementById('abstain'),
  applyThrBtn: document.getElementById('applyThr'),
  loadThrBtn: (
    document.getElementById('loadThrSet') ||
    document.getElementById('loadThr') ||
    document.getElementById('loadThrFromSet')
  ),

  // řízení
  applySymbolBtn: document.getElementById('applySymbol'),
  startBtn: document.getElementById('start'),
  stopBtn: document.getElementById('stop'),
  reloadBtn: document.getElementById('reload'),
  dlCsv: document.getElementById('dlCsv'),

  // model set (select + apply)
  modelSet: document.getElementById('modelSet'),
  applyModelSet: document.getElementById('applyModelSet'),

  // KPI
  ts: document.getElementById('ts'),
  price: document.getElementById('price'),
  pup: document.getElementById('pup'),
  decision: document.getElementById('decision'),

  // per-model KPI
  pxgb: document.getElementById('pxgb'),
  plstm: document.getElementById('plstm'),
  phrm: document.getElementById('phrm'),
  pmeta: document.getElementById('pmeta'),
  pl2: document.getElementById('pl2'),
  pl3: document.getElementById('pl3'),
  pens: document.getElementById('pens'),

  // L3 heads
  pabstain: document.getElementById('pabstain'),
  puncert: document.getElementById('puncert'),
  p3down: document.getElementById('p3down'),
  p3abstain: document.getElementById('p3abstain'),
  p3up: document.getElementById('p3up'),
  hselTop: document.getElementById('hselTop'),
  hselVec: document.getElementById('hselVec'),

  // síla signálu
  meter: document.getElementById('meter'),
  meterLong: document.getElementById('meterLong'),
  meterShort: document.getElementById('meterShort'),
  markerUp: document.getElementById('markerUp'),
  markerDown: document.getElementById('markerDown'),
  markerNow: document.getElementById('markerNow'),
  strengthPct: document.getElementById('strengthPct'),

  // grafy
  priceChart: document.getElementById('priceChart'),
  probChart: document.getElementById('probChart'),

  // rolling mini-tabulka
  rollN: document.getElementById('rollN'),
  m_cnt: document.getElementById('m_cnt'),
  m_hit: document.getElementById('m_hit'),
  m_hit_long: document.getElementById('m_hit_long'),
  m_hit_short: document.getElementById('m_hit_short'),
  m_avgd: document.getElementById('m_avgd'),
  m_last10: document.getElementById('m_last10'),

  // --- LADĚNÍ GATE ---
  gatePanel: document.getElementById('tunePanel'),
  gateModeAuto: document.getElementById('gateModeAuto'),
  gateModeManual: document.getElementById('gateModeManual'),
  gateModeBlend: document.getElementById('gateModeBlend'),
  gateAlpha: document.getElementById('alphaSlider'),
  gateAlphaVal: document.getElementById('alphaVal'),
  compList: document.getElementById('compList'),
  applyTune: document.getElementById('applyTune'),
  resetTune: document.getElementById('resetTune'),
  gateLegend: document.getElementById('gateLegend'),
  compLegend: document.getElementById('compLegend'),

  // L2/L3 snapshot v panelu
  l2Ready: document.getElementById('l2Ready'),
  l2Sets: document.getElementById('l2Sets'),
  l2Lens: document.getElementById('l2Lens'),
  l3Ready: document.getElementById('l3Ready'),
  l3Sets: document.getElementById('l3Sets'),
  l3Lens: document.getElementById('l3Lens'),
};

// ---------------------- Basic Auth ----------------------
function authHeader(){
  const a = localStorage.getItem('auth');
  return a ? {'Authorization':'Basic ' + a} : {};
}
async function ensureAuth(r){
  if (r.status === 401){
    const u = prompt('Uživatel:');
    const p = prompt('Heslo:');
    if (u!=null && p!=null){
      localStorage.setItem('auth', btoa(u+':'+p));
      return true;
    }
    return false;
  }
  return true;
}

// ---------------------- HTTP helpers ----------------------
async function getJSON(url){
  const r = await fetch(url, {headers: authHeader()});
  if (!(await ensureAuth(r))) throw new Error('Unauthorized');
  if (!r.ok) throw new Error(await r.text());
  return await r.json();
}
async function postJSON(url, body){
  const r = await fetch(url, {
    method:'POST',
    headers:{'Content-Type':'application/json', ...authHeader()},
    body: JSON.stringify(body)
  });
  if (!(await ensureAuth(r))) throw new Error('Unauthorized');
  if (!r.ok) throw new Error(await r.text());
  return await r.json();
}

function fmtTs(t){ return new Date(t*1000).toLocaleTimeString(); }
function setText(el, val){ if (el) el.textContent = val; }
function pct(n){ return (n*100).toFixed(1) + '%'; }

// ---------------------- Konfigurace / prahy ----------------------
let CFG = { CONF_ENTER_UP: 0.58, CONF_ENTER_DOWN: 0.42, ABSTAIN_MARGIN: 0.02 };
let HORIZON_SEC = null;

async function refreshConfig(){
  const cfg = await getJSON('/config');
  CFG = cfg;
  if (els.symbol) els.symbol.value = cfg.symbol;
  if (els.confUp)    els.confUp.value    = (+cfg.CONF_ENTER_UP).toFixed(3);
  if (els.confDown)  els.confDown.value  = (+cfg.CONF_ENTER_DOWN).toFixed(3);
  if (els.abstain)   els.abstain.value   = (+cfg.ABSTAIN_MARGIN).toFixed(3);
  setMarkers(cfg.CONF_ENTER_DOWN, cfg.CONF_ENTER_UP);
}

function setMarkers(down, up){
  if (els.markerDown) els.markerDown.style.left = (down*100) + '%';
  if (els.markerUp)   els.markerUp.style.left   = (up*100) + '%';
}
function updatePointer(p){
  if (els.markerNow) els.markerNow.style.left = (Math.max(0, Math.min(1, p))*100) + '%';
}

// síla signálu (0..1) relativně k prahům
function computeStrength(p){
  const up = CFG.CONF_ENTER_UP, dn = CFG.CONF_ENTER_DOWN;
  if (p >= up){
    const s = (p - up) / Math.max(1e-9, 1 - up);
    return {side:'LONG', strength: Math.max(0, Math.min(1, s))};
  }
  if (p <= dn){
    const s = (dn - p) / Math.max(1e-9, dn);
    return {side:'SHORT', strength: Math.max(0, Math.min(1, s))};
  }
  return {side:'ABSTAIN', strength: 0};
}
function renderStrength(p){
  updatePointer(p);
  const {side, strength} = computeStrength(p);
  if (els.meterLong)  els.meterLong.style.width  = (side==='LONG'  ? strength*50 : 0) + '%';
  if (els.meterShort) els.meterShort.style.width = (side==='SHORT' ? strength*50 : 0) + '%';
  if (els.strengthPct) els.strengthPct.textContent = side === 'ABSTAIN' ? '—' : `${Math.round(strength*100)}%`;

  els.pup?.classList.remove('green','red');
  els.decision?.classList.remove('long','short','abstain');
  if (side === 'LONG'){  els.pup?.classList.add('green'); els.decision?.classList.add('long'); }
  else if (side === 'SHORT'){ els.pup?.classList.add('red');  els.decision?.classList.add('short'); }
  else { els.decision?.classList.add('abstain'); }
}

// Per-model obarvení
function colorByProb(el, p){
  if (!el) return;
  el.classList.remove('green','red');
  if (p == null || Number.isNaN(p)) return;
  if (p >= CFG.CONF_ENTER_UP) el.classList.add('green');
  else if (p <= CFG.CONF_ENTER_DOWN) el.classList.add('red');
}

// ---------------------- Model sets ----------------------
async function refreshModelSets(){
  const m = await getJSON('/modelsets');
  const opts = m.options || [];
  if (!els.modelSet) return;
  els.modelSet.innerHTML = '';
  opts.forEach(o=>{
    const opt = document.createElement('option');
    opt.value = o.path;
    opt.textContent = o.name + ' — ' + o.path.replace(/^.*weights\//,'weights/');
    if (o.path === m.current) opt.selected = true;
    els.modelSet.appendChild(opt);
  });
}

// ---------------------- Canvas helpers ----------------------
function drawLine(ctx, data, color, ymin, ymax){
  const W = ctx.canvas.width, H = ctx.canvas.height;
  if (!data || data.length < 2) return;
  ctx.beginPath();
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.5;
  for (let i=0;i<data.length;i++){
    const x = (i/(data.length-1)) * (W-20) + 10;
    const y = H - ((data[i]-ymin)/(ymax-ymin))*(H-20) - 10;
    if (i===0) ctx.moveTo(x,y); else ctx.lineTo(x,y);
  }
  ctx.stroke();
}
function drawBands(ctx){
  const W = ctx.canvas.width, H = ctx.canvas.height;
  // FIX: čistit plátno tady, ať grid zůstává i po vykreslení křivky
  ctx.clearRect(0,0,W,H);
  ctx.fillStyle = 'rgba(255,255,255,0.06)';
  for(let i=0;i<=4;i++){
    const y = (i/4)*(H-20)+10;
    ctx.fillRect(0, y-0.5, W, 1);
  }
}
function drawProbThresholds(ctx, up, down){
  const W = ctx.canvas.width, H = ctx.canvas.height;
  const ymin = 0, ymax = 1;
  const yFor = p => H - ((p - ymin)/(ymax - ymin))*(H-20) - 10;
  const yUp = yFor(up), yDn = yFor(down);
  ctx.save();
  ctx.fillStyle = 'rgba(255,255,255,0.05)';
  ctx.fillRect(0, Math.min(yUp,yDn), W, Math.abs(yUp - yDn));
  ctx.setLineDash([6,4]);
  ctx.beginPath(); ctx.strokeStyle = '#ff6b6b'; ctx.moveTo(0, yDn); ctx.lineTo(W, yDn); ctx.stroke();
  ctx.beginPath(); ctx.strokeStyle = '#41d392'; ctx.moveTo(0, yUp); ctx.lineTo(W, yUp); ctx.stroke();
  ctx.restore();
}
function drawDecisionDots(ctx, prices, decisions, ymin, ymax){
  if (!decisions || decisions.length !== prices.length) return;
  const W = ctx.canvas.width, H = ctx.canvas.height;
  for (let i=0;i<prices.length;i++){
    const d = (decisions[i] || '').toUpperCase();
    if (d !== 'LONG' && d !== 'SHORT') continue;
    const x = (i/(prices.length-1)) * (W-20) + 10;
    const y = H - ((prices[i]-ymin)/(ymax-ymin))*(H-20) - 10;
    ctx.beginPath();
    ctx.fillStyle = (d === 'LONG') ? '#41d392' : '#ff6b6b';
    ctx.arc(x, y, 3, 0, Math.PI*2);
    ctx.fill();
  }
}
function drawOutcomes(ctx, times, prices, outcomesMap, ymin, ymax){
  if (!times || !prices || !outcomesMap) return;
  const W = ctx.canvas.width, H = ctx.canvas.height;
  const idxOf = new Map(times.map((t, i)=>[t, i]));
  for (const [tsPredStr, oc] of Object.entries(outcomesMap)){
    const tsPred = parseInt(tsPredStr, 10);
    const i = idxOf.get(tsPred);
    const j = idxOf.get(oc.ts_out);
    if (i == null || j == null) continue;
    const x1 = (i/(prices.length-1)) * (W-20) + 10;
    const y1 = H - ((prices[i]-ymin)/(ymax-ymin))*(H-20) - 10;
    const x2 = (j/(prices.length-1)) * (W-20) + 10;
    const y2 = H - ((prices[j]-ymin)/(ymax-ymin))*(H-20) - 10;

    ctx.beginPath();
    ctx.lineWidth = 1;
    ctx.strokeStyle = oc.win ? '#41d392' : '#ff6b6b';
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.stroke();

    ctx.beginPath();
    ctx.lineWidth = 2;
    ctx.strokeStyle = oc.win ? '#41d392' : '#ff6b6b';
    ctx.fillStyle = 'transparent';
    ctx.arc(x2, y2, 4, 0, Math.PI*2);
    ctx.stroke();
  }
}
// minipruhy pod cenovým grafem
function drawDecisionMiniBars(ctx, times, prices, decisions, probsAligned, outcomesMap){
  if (!times || !prices || !decisions || !probsAligned) return;
  const N = prices.length;
  if (N < 2) return;
  const W = ctx.canvas.width, H = ctx.canvas.height;
  const plotH = (H - 20);
  const baseY = H - 10;
  const dx = (W - 20) / (N - 1);

  for (let i = 0; i < N; i++){
    const dec = (decisions[i] || '').toUpperCase();
    if (dec !== 'LONG' && dec !== 'SHORT') continue;

    const p = probsAligned[i];
    if (p == null || Number.isNaN(p)) continue;

    const { strength } = computeStrength(p);
    if (strength <= 0) continue;

    const ts = times[i];
    const oc = outcomesMap[ts];
    let fill = 'rgba(200,200,200,0.7)'; // pending
    if (oc){ fill = oc.win ? '#41d392' : '#ff6b6b'; }

    const xCenter = (i/(N-1)) * (W - 20) + 10;
    const width = Math.max(1, dx * 0.7);
    const x0 = Math.round(xCenter - width/2);

    const hMax = plotH * 0.25;
    const h = Math.max(1, Math.round(hMax * strength));

    ctx.fillStyle = fill;
    ctx.fillRect(x0, baseY - h, Math.ceil(width), h);
  }
}
// per-model minipruhy v PROB grafu
function drawProbComponentBars(ctx, probsMapSeries){
  const keys = ['xgb','lstm','hrm','meta'];
  const colors = { xgb: '#3388ff', lstm:'#ff9900', hrm:'#a066ff', meta:'#00b3a4' };
  const N = (probsMapSeries.xgb||probsMapSeries.lstm||probsMapSeries.hrm||probsMapSeries.meta||[]).length;
  if (!N) return;

  const W = ctx.canvas.width, H = ctx.canvas.height;
  const dx = (W - 20) / (N - 1);
  const baseY = H - 10;
  const bandH = Math.max(2, Math.floor((H - 20) * 0.10));

  const order = keys.filter(k => (probsMapSeries[k] && probsMapSeries[k].length === N));
  order.forEach((k, row)=>{
    const yTop = baseY - (row+1)* (bandH + 2);
    const arr = probsMapSeries[k];
    for (let i=0;i<N;i++){
      const p = arr[i];
      if (p==null || Number.isNaN(p)) continue;
      const xCenter = (i/(N-1)) * (W - 20) + 10;
      const width = Math.max(1, dx * 0.7);
      const x0 = Math.round(xCenter - width/2);
      const h = Math.max(1, Math.round(bandH * Math.max(0, Math.min(1, p))));
      ctx.fillStyle = colors[k];
      ctx.globalAlpha = 0.9;
      ctx.fillRect(x0, yTop + (bandH - h), Math.ceil(width), h);
      ctx.globalAlpha = 1.0;
    }
  });
 
  if (els.gateLegend){
    els.gateLegend.innerHTML = '';
	  order.forEach(k=>{
	    const li = document.createElement('div');
	    li.style.display = 'inline-flex';
	    li.style.alignItems = 'center';
	    li.style.marginRight = '10px';
	    const sw = document.createElement('span');
	    sw.style.display = 'inline-block';
	    sw.style.width = '10px';
	    sw.style.height = '10px';
	    sw.style.background = colors[k];
	    sw.style.marginRight = '6px';
	    li.appendChild(sw);
	    li.appendChild(document.createTextNode(k.toUpperCase()));
	    els.gateLegend.appendChild(li);
	  });
	}
}

// ---------------------- Rolling metriky ----------------------
const outcomesMap = {};
let outcomesList = [];

function renderRolling(){
  const N = Math.max(10, parseInt(els.rollN?.value || '200', 10));
  const arr = outcomesList.slice(-N);
  const cnt = arr.length;
  if (cnt === 0){
    setText(els.m_cnt,'0'); setText(els.m_hit,'—');
    setText(els.m_hit_long,'—'); setText(els.m_hit_short,'—');
    setText(els.m_avgd,'—'); setText(els.m_last10,'—');
    return;
  }
  let wins=0, wl=0, ws=0, nl=0, ns=0, sumd=0;
  for (const oc of arr){
    if (oc.win) wins++;
    sumd += oc.delta;
    if (oc.side === 'LONG'){ nl++; if (oc.win) wl++; }
    else if (oc.side === 'SHORT'){ ns++; if (oc.win) ws++; }
  }
  const last10 = arr.slice(-10).map(x => x.win ? '✓' : '×').join(' ');
  setText(els.m_cnt, String(cnt));
  setText(els.m_hit, pct(wins / cnt));
  setText(els.m_hit_long, nl? pct(wl/nl) : '—');
  setText(els.m_hit_short, ns? pct(ws/ns) : '—');
  setText(els.m_avgd, (sumd/cnt).toFixed(4));
  setText(els.m_last10, last10);
}

// ---------------------- Hlavní smyčka ----------------------
let times=[], prices=[], probs=[], decisions=[], probsAligned=[];
let probsX=[], probsL=[], probsH=[], probsM=[]; // per-model KPIs

async function refreshModelStatus(){
  try {
    const health = await getJSON('/health');
    if (typeof health.horizon_sec === 'number') HORIZON_SEC = health.horizon_sec;

    const hz = (typeof health.horizon_sec !== 'undefined') ? ` · horizon: <b>${health.horizon_sec}s</b>` : '';
    const wd = health.weights_dir ? ` · set: <b>${health.weights_dir.replace(/^.*weights\//,'weights/')}</b>` : '';
    const thr = (typeof health.thr_up === 'number' && typeof health.thr_down === 'number')
      ? ` · thr: <b>${(+health.thr_down).toFixed(3)} / ${(+health.thr_up).toFixed(3)}</b> · margin: <b>${(+health.margin || 0).toFixed(3)}</b>`
      : '';

    // volitelné doplnění hysteréze/cooldown (když to backend pošle v /health.gate apod.)
    const hy = (health.thresholds && typeof health.thresholds.HYSTERESIS === 'number')
      ? ` · hyst: <b>${(+health.thresholds.HYSTERESIS).toFixed(3)}</b>` : '';
    const cd = (health.thresholds && typeof health.thresholds.COOLDOWN_SEC === 'number')
      ? ` · cooldown: <b>${health.thresholds.COOLDOWN_SEC}s</b>` : '';

    els.status.innerHTML =
      `symbol: <b>${health.symbol}</b>${hz}${wd}${thr}${hy}${cd} · ` +
      `bootstrapped: <span class="badge ${health.bootstrapped?'ok':'fail'}">${health.bootstrapped?'OK':'WAIT'}</span> · ` +
      `modely: <span class="badge ${health.models_loaded?'ok':'fail'}">${health.models_loaded?'načteny':'nenalezeny'}</span> · ` +
      `běh: ${health.running?'▶':'⏸'}`;
  } catch(e){
    els.status.textContent = 'Chyba: ' + e.message;
  }
}

function _shortSetPath(p){
  try { return p.replace(/^.*weights\//,'weights/'); } catch(_){ return p; }
}
function _fmtSets(arr){
  if (!arr || !arr.length) return '—';
  return arr.map(_shortSetPath).join(', ');
}

async function refreshGatePanel(){
  if (!els.gatePanel) return;
  try{
    const st = await getJSON('/gate');
    if (!st.ready){
      els.gatePanel.style.display = 'none';
      return;
    }
    els.gatePanel.style.display = '';
    // L2/L3 snapshot
    setText(els.l2Ready, st.l2_ready ? 'OK' : '—');
    setText(els.l2Sets, _fmtSets(st.l2_sets || []));
    const l2lens = st.l2_lens ? `L1=${st.l2_lens.l1_seq_len}, L2=${st.l2_lens.l2_seq_len}${st.l2_use_raw? ' · raw:on' : ''}` : '—';
    setText(els.l2Lens, l2lens);

    setText(els.l3Ready, st.l3_ready ? 'OK' : '—');
    setText(els.l3Sets, _fmtSets(st.l3_sets || []));
    const l3lens = st.l3_lens ? `L3=${st.l3_lens.l3_seq_len}${st.l3_use_raw? ' · raw:on' : ''}` : '—';
    setText(els.l3Lens, l3lens);

    // mode
    if (els.gateModeAuto)   els.gateModeAuto.checked   = (st.mode === 'auto');
    if (els.gateModeManual) els.gateModeManual.checked = (st.mode === 'manual');
    if (els.gateModeBlend)  els.gateModeBlend.checked  = (st.mode === 'blend');

    // alpha
    if (els.gateAlpha){
      els.gateAlpha.value = st.alpha ?? 0.3;
      if (els.gateAlphaVal) els.gateAlphaVal.textContent = (+els.gateAlpha.value).toFixed(2);
    }

    // components list
    if (els.compList){
      els.compList.innerHTML = '';
      (st.components || []).forEach(k=>{
        const row = document.createElement('div');
        row.className = 'comp-row';
        const lab = document.createElement('div');
        lab.className = 'comp-lab';
        lab.textContent = k;
        const vals = document.createElement('div');
        vals.className = 'comp-vals';

        const prob = (st.last_probs && typeof st.last_probs[k]==='number') ? st.last_probs[k] : null;
        const probSpan = document.createElement('span');
        probSpan.className = 'comp-prob';
        probSpan.textContent = (prob==null? '—' : (+prob).toFixed(3));

        const inp = document.createElement('input');
        inp.type = 'range'; inp.min = '0.10'; inp.max = '3.00'; inp.step='0.01';
        inp.value = (st.gains && st.gains[k]!=null) ? st.gains[k] : 1.0;
        inp.dataset.key = k;
        const v = document.createElement('span');
        v.className = 'comp-gain';
        v.textContent = (+inp.value).toFixed(2);
        inp.oninput = ()=>{ v.textContent = (+inp.value).toFixed(2); };
        vals.appendChild(probSpan);
        vals.appendChild(inp);
        vals.appendChild(v);

        row.appendChild(lab);
        row.appendChild(vals);
        els.compList.appendChild(row);
      });
    }
  }catch(e){
    els.gatePanel && (els.gatePanel.style.display='none');
  }
}

async function tick(){
  try {
    await refreshModelStatus();
    await refreshGatePanel();
    await maybeAutoPreset();

    const hist = await getJSON('/history?limit=600');
    times = hist.map(h => h.ts);
    prices = hist.map(h => h.price);
    probsAligned = hist.map(h => (h.p_ens ?? h.prob_up ?? null));
    probs = probsAligned.filter(v => !Number.isNaN(v) && v != null);
    decisions = hist.map(h => (h.decision || '').toUpperCase());
    // per-model
    probsX = hist.map(h => (h.p_xgb ?? null));
    probsL = hist.map(h => (h.p_lstm ?? null));
    probsH = hist.map(h => (h.p_hrm ?? null));
    probsM = hist.map(h => (h.p_meta ?? null));

    if (hist.length>0){
      const last = hist[hist.length-1];
      const pupVal = (last.p_ens ?? last.prob_up ?? null);
      setText(els.ts, fmtTs(last.ts));
      setText(els.price, (last.price!=null)? (+last.price).toFixed(2) : '—');
      setText(els.pup, (pupVal!=null)? (+pupVal).toFixed(3):'—');
      setText(els.decision, last.decision ?? '—');

      // per-model KPI + barvy
      setText(els.pxgb,  last.p_xgb != null ? (+last.p_xgb ).toFixed(3) : '—');
      setText(els.plstm, last.p_lstm!= null ? (+last.p_lstm).toFixed(3) : '—');
      setText(els.phrm,  last.p_hrm != null ? (+last.p_hrm ).toFixed(3) : '—');
      setText(els.pmeta, last.p_meta!= null ? (+last.p_meta).toFixed(3) : '—');
      setText(els.pl2,   last.p_l2   != null ? (+last.p_l2  ).toFixed(3) : '—');
      setText(els.pl3,   last.p_l3   != null ? (+last.p_l3  ).toFixed(3) : '—');
      setText(els.pens,  last.p_ens  != null ? (+last.p_ens ).toFixed(3) : (pupVal!=null ? (+pupVal).toFixed(3) : '—'));

      colorByProb(els.pxgb,  last.p_xgb);
      colorByProb(els.plstm, last.p_lstm);
      colorByProb(els.phrm,  last.p_hrm);
      colorByProb(els.pmeta, last.p_meta);
      colorByProb(els.pl2,   last.p_l2);
      colorByProb(els.pl3,   last.p_l3);
      colorByProb(els.pens,  last.p_ens ?? pupVal);

      // L3 heads (pokud jsou)
      setText(els.pabstain, last.p_abstain!=null ? (+last.p_abstain).toFixed(3) : '—');
      setText(els.puncert,  last.p_uncert !=null ? (+last.p_uncert ).toFixed(3) : '—');
      if (Array.isArray(last.p3) && last.p3.length===3){
        setText(els.p3down,    (+last.p3[0]).toFixed(3));
        setText(els.p3abstain, (+last.p3[1]).toFixed(3));
        setText(els.p3up,      (+last.p3[2]).toFixed(3));
      } else {
        setText(els.p3down,'—'); setText(els.p3abstain,'—'); setText(els.p3up,'—');
      }
      if (Array.isArray(last.hsel) && last.hsel.length>0){
        const vals = last.hsel.map(x => (+x).toFixed(3));
        const argmax = last.hsel.indexOf(Math.max(...last.hsel));
        setText(els.hselTop, `#${argmax} / ${last.hsel.length}`);
        setText(els.hselVec, `[${vals.join(', ')}]`);
      } else {
        setText(els.hselTop,'—'); setText(els.hselVec,'—');
      }

      if (pupVal!=null) renderStrength(+pupVal);
    }

    // outcomes
    try {
      const outs = await getJSON('/outcomes?limit=500');
      outcomesList = outs;
      for (const oc of outs){ outcomesMap[oc.ts_pred] = oc; }
      renderRolling();
    } catch(_){}

    // PRICE chart
    const pctx = els.priceChart.getContext('2d');
    drawBands(pctx);
    if (prices.length>1){
      const minp = Math.min(...prices);
      const maxp = Math.max(...prices);
      drawLine(pctx, prices, '#4ea3ff', minp, maxp);
      drawDecisionDots(pctx, prices, decisions, minp, maxp);
      drawOutcomes(pctx, times, prices, outcomesMap, minp, maxp);
      drawDecisionMiniBars(pctx, times, prices, decisions, probsAligned, outcomesMap);
    }

    // PROB chart
    const prctx = els.probChart.getContext('2d');
    drawBands(prctx);
    if (probs.length>1){
      drawProbThresholds(prctx, CFG.CONF_ENTER_UP, CFG.CONF_ENTER_DOWN);
      drawLine(prctx, probs, '#41d392', 0, 1);
      drawProbComponentBars(prctx, { xgb: probsX, lstm: probsL, hrm: probsH, meta: probsM });
    }

  } catch (e) {
    els.status.textContent = 'Chyba: ' + e.message;
  } finally {
    setTimeout(tick, 1000);
  }
}

// ---------------------- Handlery ----------------------
async function applyThresholdsFromInputs(){
  await postJSON('/config', {
    CONF_ENTER_UP:   parseFloat(els.confUp.value),
    CONF_ENTER_DOWN: parseFloat(els.confDown.value),
    ABSTAIN_MARGIN:  parseFloat(els.abstain.value)
  });
  await refreshConfig();
}
if (els.applyThrBtn) els.applyThrBtn.onclick = applyThresholdsFromInputs;

if (els.loadThrBtn) els.loadThrBtn.onclick = async ()=>{
  try{
    const h = await getJSON('/health');
    const up = (typeof h.thr_up   === 'number') ? h.thr_up   : null;
    const dn = (typeof h.thr_down === 'number') ? h.thr_down : null;
    const mg = (typeof h.margin   === 'number') ? h.margin   :
               (typeof h.abstain_margin === 'number' ? h.abstain_margin : null);
    if (up==null || dn==null){
      alert('Aktivní sada neposlala doporučené prahy (thr_up/thr_down).');
      return;
    }
    if (els.confUp)   els.confUp.value   = (+up).toFixed(3);
    if (els.confDown) els.confDown.value = (+dn).toFixed(3);
    if (els.abstain && mg!=null) els.abstain.value = (+mg).toFixed(3);
    setMarkers(+dn, +up);
    await applyThresholdsFromInputs();
  }catch(e){
    alert('Načtení prahů ze sady selhalo: ' + e.message);
  }
};

if (els.applySymbolBtn) els.applySymbolBtn.onclick = async () => {
  await postJSON('/config', { SYMBOL: els.symbol.value });
  await refreshConfig();
};
if (els.startBtn)  els.startBtn.onclick  = async () => { await postJSON('/action', { type: 'start' }); };
if (els.stopBtn)   els.stopBtn.onclick   = async () => { await postJSON('/action', { type: 'stop' }); };
if (els.reloadBtn) els.reloadBtn.onclick = async () => { await postJSON('/action', { type: 'reload_models' }); await refreshModelStatus(); };
if (els.dlCsv)     els.dlCsv.onclick     = () => { window.location = '/download/signals'; };
if (els.rollN)     els.rollN.onchange    = renderRolling;

if (els.applyModelSet) els.applyModelSet.onclick = async ()=>{
  const val = els.modelSet.value;
  await postJSON('/modelsets', { path: val });
  await postJSON('/action', { type: 'reload_models' });
  try{
    const h = await getJSON('/health');
    if (typeof h.thr_up === 'number' && typeof h.thr_down === 'number'){
      if (els.confUp)   els.confUp.value   = (+h.thr_up).toFixed(3);
      if (els.confDown) els.confDown.value = (+h.thr_down).toFixed(3);
      if (els.abstain && typeof h.margin === 'number') els.abstain.value = (+h.margin).toFixed(3);
      setMarkers(+h.thr_down, +h.thr_up);
      await applyThresholdsFromInputs();
    }
  }catch(_){}
  await refreshConfig();
  await refreshModelSets();
  await refreshModelStatus();
  await refreshGatePanel();
};

// ---- Gate UI handlers ----
function readGateForm(){
  const mode = els.gateModeAuto?.checked ? 'auto' : (els.gateModeManual?.checked ? 'manual' : 'blend');
  const alpha = els.gateAlpha ? parseFloat(els.gateAlpha.value) : 0.3;
  const gains = {};
  if (els.compList){
    els.compList.querySelectorAll('input[type=range]').forEach(inp=>{
      const k = inp.dataset.key;
      gains[k] = parseFloat(inp.value);
    });
  }
  return { mode, alpha, gains };
}
if (els.gateAlpha){
  els.gateAlpha.oninput = ()=>{ if (els.gateAlphaVal) els.gateAlphaVal.textContent = (+els.gateAlpha.value).toFixed(2); };
}
if (els.applyTune){
  els.applyTune.onclick = async ()=>{
    const {mode, alpha, gains} = readGateForm();
    await postJSON('/gate', { mode, alpha, gains });
    await refreshGatePanel();
  };
}
if (els.resetTune){
  els.resetTune.onclick = async ()=>{
    await postJSON('/gate', { reset: true });
    await refreshGatePanel();
  };
}

// ---------- Presety (localStorage) ----------
function normPresetName(raw){
  if (!raw) return null;
  let n = String(raw).trim();
  if (n.startsWith('@')) n = n.slice(1);
  n = n.toLowerCase();
  return n || null;
}
function presetKey(n){ return 'gatePreset:' + n; }

async function saveCurrentPreset(){
  const nameRaw = document.getElementById('presetName')?.value || '';
  const name = normPresetName(nameRaw);
  if (!name){ setText(document.getElementById('presetStatus'),'Zadej @název'); return; }
  const {mode, alpha, gains} = readGateForm();
  const obj = { mode, alpha, gains };
  localStorage.setItem(presetKey(name), JSON.stringify(obj));
  localStorage.setItem('gateLastPreset', name);
  setText(document.getElementById('presetStatus'), `Uloženo jako @${name}`);
}
async function applyPresetByName(){
  const nameRaw = document.getElementById('presetName')?.value || '';
  const name = normPresetName(nameRaw);
  if (!name){ setText(document.getElementById('presetStatus'),'Zadej @název'); return; }
  const raw = localStorage.getItem(presetKey(name));
  if (!raw){ setText(document.getElementById('presetStatus'), `Preset @${name} nenalezen`); return; }
  try{
    const obj = JSON.parse(raw);
    await postJSON('/gate', obj);
    await refreshGatePanel();
    localStorage.setItem('gateLastPreset', name);
    setText(document.getElementById('presetStatus'), `Aplikováno: @${name}`);
  }catch(e){
    setText(document.getElementById('presetStatus'), 'Chyba při aplikaci');
  }
}
function deletePreset(){
  const nameRaw = document.getElementById('presetName')?.value || '';
  const name = normPresetName(nameRaw);
  if (!name){ setText(document.getElementById('presetStatus'),'Zadej @název'); return; }
  localStorage.removeItem(presetKey(name));
  setText(document.getElementById('presetStatus'), `Smazáno: @${name}`);
}
function autoApplyPresetFromURLorStorage(){
  const params = new URLSearchParams(location.search);
  let n = params.get('preset') || location.hash;
  if (n && n.startsWith('#')) n = n.slice(1);
  if (!n){
    const last = localStorage.getItem('gateLastPreset');
    if (last) n = '@' + last;
  }
  const name = normPresetName(n||'');
  if (!name) return;
  const inp = document.getElementById('presetName');
  if (inp) inp.value = '@' + name;
  applyPresetByName();
}
(function bindPresetButtons(){
  const saveBtn = document.getElementById('savePreset');
  const applyBtn = document.getElementById('applyPreset');
  const delBtn = document.getElementById('deletePreset');
  if (saveBtn) saveBtn.onclick = saveCurrentPreset;
  if (applyBtn) applyBtn.onclick = applyPresetByName;
  if (delBtn) delBtn.onclick = deletePreset;
})();
let __autoPresetApplied = false;
async function maybeAutoPreset(){
  if (!__autoPresetApplied){
    __autoPresetApplied = true;
    autoApplyPresetFromURLorStorage();
  }
}

// ---------------------- SSE (pokud není Basic Auth) ----------------------
(function connectSSE(){
  if (localStorage.getItem('auth')) return;
  let es;
  function open(){
    es = new EventSource('/events', { withCredentials: false });
    es.onmessage = (ev)=>{
      try {
        const msg = JSON.parse(ev.data);
        if (msg.type==='signal' && msg.data){
          const d = msg.data;
          const pupVal = (d.p_ens ?? d.prob_up ?? null);

          setText(els.ts, fmtTs(d.ts));
          setText(els.price, (+d.price).toFixed(2));
          setText(els.pup, (pupVal!=null)? (+pupVal).toFixed(3):'—');
          setText(els.decision, d.decision ?? '—');

          setText(els.pxgb,  d.p_xgb != null ? (+d.p_xgb ).toFixed(3) : '—');
          setText(els.plstm, d.p_lstm!= null ? (+d.p_lstm).toFixed(3) : '—');
          setText(els.phrm,  d.p_hrm != null ? (+d.p_hrm ).toFixed(3) : '—');
          setText(els.pmeta, d.p_meta!= null ? (+d.p_meta).toFixed(3) : '—');
          setText(els.pl2,   d.p_l2  != null ? (+d.p_l2  ).toFixed(3) : '—');
          setText(els.pl3,   d.p_l3  != null ? (+d.p_l3  ).toFixed(3) : '—');
          setText(els.pens,  d.p_ens != null ? (+d.p_ens ).toFixed(3) : (pupVal!=null ? (+pupVal).toFixed(3) : '—'));

          colorByProb(els.pxgb,  d.p_xgb);
          colorByProb(els.plstm, d.p_lstm);
          colorByProb(els.phrm,  d.p_hrm);
          colorByProb(els.pmeta, d.p_meta);
          colorByProb(els.pl2,   d.p_l2);
          colorByProb(els.pl3,   d.p_l3);
          colorByProb(els.pens,  d.p_ens ?? pupVal);

          // L3 heads
          setText(els.pabstain, d.p_abstain!=null ? (+d.p_abstain).toFixed(3) : '—');
          setText(els.puncert,  d.p_uncert !=null ? (+d.p_uncert ).toFixed(3) : '—');
          if (Array.isArray(d.p3) && d.p3.length===3){
            setText(els.p3down,    (+d.p3[0]).toFixed(3));
            setText(els.p3abstain, (+d.p3[1]).toFixed(3));
            setText(els.p3up,      (+d.p3[2]).toFixed(3));
          } else {
            setText(els.p3down,'—'); setText(els.p3abstain,'—'); setText(els.p3up,'—');
          }
          if (Array.isArray(d.hsel) && d.hsel.length>0){
            const vals = d.hsel.map(x => (+x).toFixed(3));
            const argmax = d.hsel.indexOf(Math.max(...d.hsel));
            setText(els.hselTop, `#${argmax} / ${d.hsel.length}`);
            setText(els.hselVec, `[${vals.join(', ')}]`);
          } else {
            setText(els.hselTop,'—'); setText(els.hselVec,'—');
          }

          // posuň řady
          times.push(d.ts);      if (times.length>600)    times.shift();
          prices.push(d.price);  if (prices.length>600)   prices.shift();
          const dec = (d.decision||'').toUpperCase();
          decisions.push(dec);   if (decisions.length>600) decisions.shift();
          if (pupVal!=null){
            probsAligned.push(+pupVal); if (probsAligned.length>600) probsAligned.shift();
            probsX.push(d.p_xgb ?? null); if (probsX.length>600) probsX.shift();
            probsL.push(d.p_lstm ?? null); if (probsL.length>600) probsL.shift();
            probsH.push(d.p_hrm ?? null); if (probsH.length>600) probsH.shift();
            probsM.push(d.p_meta ?? null); if (probsM.length>600) probsM.shift();
            renderStrength(+pupVal);
          }
        } else if (msg.type === 'outcome' && msg.data){
          const oc = msg.data;
          outcomesMap[oc.ts_pred] = oc;
          outcomesList.push(oc);
          renderRolling();
        }
      } catch(_){}
    };
    es.onerror = ()=>{ try{ es.close(); }catch(_){ } setTimeout(open, 3000); };
  }
  open();
})();

// ---------------------- Init ----------------------
Promise.all([refreshConfig(), refreshModelSets(), refreshModelStatus()]).then(()=>tick());