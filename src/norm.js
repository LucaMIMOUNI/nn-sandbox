/* ==================================================================
   MODULE 2 — BatchNorm / LayerNorm / InstanceNorm / GroupNorm
   ================================================================== */
const NORM = (function(){


/* ================================ state ================================ */
const S = {
  N:3, C:4, H:3, Wd:3,
  norm:"batch", G:2,
  epsExp:-5, aff:"init", mode:"train", mom:0.1,
  dead:false,
  batch:1,                       // which sample of the data distribution we are showing
  steps:0,                       // training batches fed so far
  focus:null,                    // {t,n,c,h,w}
  calc:{N:null, C:null, H:null, W:null},
};

const DIMS = [
  {id:"N",  name:"batch N",   cls:"N",  hint:"how many samples are in the batch"},
  {id:"C",  name:"channels C", cls:"C",  hint:"num_features — one γ, β per channel"},
  {id:"H",  name:"height H",  cls:"HW", hint:"spatial size"},
  {id:"Wd", name:"width W",   cls:"HW", hint:"spatial size"},
];

const el = id => document.getElementById("bn_" + id);
const eps = () => Math.pow(10, S.epsExp);

let X = null, Y = null;          // Float32Array N*C*H*W
let MU = null, VA = null;        // per (n,c): the statistic actually used
let GRP = [], GID = null;        // pooling groups, and (n,c) -> group index
let RM = null, RV = null;        // running_mean, running_var
let GAM = null, BET = null;      // gamma, beta per channel
let L = {};                      // pixel layout

const idx  = (n,c,h,w) => ((n*S.C + c)*S.H + h)*S.Wd + w;
const nc   = (n,c) => n*S.C + c;
const calc = k => S.calc[k] === null ? S[k === "W" ? "Wd" : k] : S.calc[k];
const calcLinked = () => Object.values(S.calc).every(v => v === null);

/* ============================== the data ==============================
   Each channel has its own fixed distribution (mean, spread). A "batch" is
   a fresh sample from it — so running statistics converge to something real
   as you feed more batches.
   ====================================================================== */
function prng(s){ let x = s>>>0; return () => (x = (x*1664525 + 1013904223)>>>0) / 4294967296; }
function gauss(r){ const u = Math.max(r(), 1e-9); return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*r()); }
const r1 = v => Math.round(v*10)/10;

function channelDist(c){
  const r = prng(9001 + c*7919);
  r(); r();
  return {mu: r1(r()*18 - 8), sd: r1(0.4 + r()*2.6)};
}

function buildData(){
  const {N,C,H,Wd} = S;
  X = new Float32Array(N*C*H*Wd);
  const r = prng(1234567 + S.batch*104729);
  for(let c=0;c<C;c++){
    const {mu, sd} = channelDist(c);
    const s = (S.dead && c === 0) ? 0 : sd;
    for(let n=0;n<N;n++) for(let h=0;h<H;h++) for(let w=0;w<Wd;w++)
      X[idx(n,c,h,w)] = r1(mu + s*gauss(r));
  }
}

function buildAffine(){
  GAM = new Float32Array(S.C).fill(1);
  BET = new Float32Array(S.C);
  if(S.aff !== "learned") return;
  const r = prng(555);
  for(let c=0;c<S.C;c++){ GAM[c] = r1(0.4 + r()*1.6); BET[c] = r1(r()*4 - 2); }
}

function resetRunning(){
  RM = new Float32Array(S.C);          // PyTorch initialises these to 0 and 1
  RV = new Float32Array(S.C).fill(1);
  S.steps = 0;
}

/* ============================ pooling groups ============================
   The one thing that separates the four normalisation layers.
   ====================================================================== */
function groupKey(n,c){
  switch(S.norm){
    case "batch":    return "c"+c;
    case "layer":    return "n"+n;
    case "instance": return "n"+n+"|c"+c;
    default:         return "n"+n+"|g"+Math.floor(c/(S.C/S.G));
  }
}
function groupLabel(g){
  const cs = [...new Set(g.members.map(m => m.c))].sort((a,b)=>a-b);
  const ns = [...new Set(g.members.map(m => m.n))].sort((a,b)=>a-b);
  const cpart = cs.length === S.C ? "all c" : cs.length === 1 ? "c"+cs[0] : `c${cs[0]}–${cs[cs.length-1]}`;
  const npart = ns.length === S.N ? "all n" : "n"+ns[0];
  return `${npart} · ${cpart}`;
}

function buildGroups(){
  const {N,C,H,Wd} = S;
  const map = new Map();
  GRP = []; GID = new Int32Array(N*C);
  for(let n=0;n<N;n++) for(let c=0;c<C;c++){
    const key = groupKey(n,c);
    if(!map.has(key)){ map.set(key, GRP.length); GRP.push({key, members:[]}); }
    const gi = map.get(key);
    GRP[gi].members.push({n,c});
    GID[nc(n,c)] = gi;
  }
  for(const g of GRP){
    let sum = 0, n = 0;
    for(const m of g.members) for(let h=0;h<H;h++) for(let w=0;w<Wd;w++){ sum += X[idx(m.n,m.c,h,w)]; n++; }
    const mean = sum/n;
    let ss = 0;
    for(const m of g.members) for(let h=0;h<H;h++) for(let w=0;w<Wd;w++){
      const d = X[idx(m.n,m.c,h,w)] - mean; ss += d*d;
    }
    g.count = n;
    g.mean  = mean;
    g.var   = ss/n;                       // biased — this is what normalises
    g.varU  = n > 1 ? ss/(n-1) : 0;       // unbiased — this is what running_var stores
    g.label = groupLabel(g);
  }
}

/* ============================== the maths ============================== */
function normalise(){
  const {N,C,H,Wd} = S;
  Y = new Float32Array(N*C*H*Wd);
  MU = new Float32Array(N*C); VA = new Float32Array(N*C);
  const frozen = S.norm === "batch" && S.mode === "eval";
  for(let n=0;n<N;n++) for(let c=0;c<C;c++){
    const g = GRP[GID[nc(n,c)]];
    const mu = frozen ? RM[c] : g.mean;
    const va = frozen ? RV[c] : g.var;
    MU[nc(n,c)] = mu; VA[nc(n,c)] = va;
    const inv = 1/Math.sqrt(va + eps());
    for(let h=0;h<H;h++) for(let w=0;w<Wd;w++){
      const i = idx(n,c,h,w);
      Y[i] = GAM[c]*(X[i] - mu)*inv + BET[c];
    }
  }
}

// one training step: a fresh batch moves the running estimates towards the truth
function feedBatch(){
  S.batch++;
  buildData(); buildGroups();
  if(S.mode === "train" && S.norm === "batch"){
    const m = S.mom;
    for(let c=0;c<S.C;c++){
      const g = GRP[GID[nc(0,c)]];
      RM[c] = (1-m)*RM[c] + m*g.mean;
      RV[c] = (1-m)*RV[c] + m*g.varU;      // note: unbiased here, biased above
    }
    S.steps++;
  }
  refresh();
}

/* ============================== rendering ============================== */
const dark = () => matchMedia("(prefers-color-scheme: dark)").matches;
function palette(){
  const d = dark();
  return {
    line:  d ? "#2a323d" : "#d6dde5",
    text:  d ? "#e6edf3" : "#1c2128",
    dim:   d ? "#8b949e" : "#57606a",
    dim2:  d ? "#6e7681" : "#8c959f",
    panel: d ? "#161b22" : "#ffffff",
    veil:  d ? "rgba(22,27,34,.80)" : "rgba(255,255,255,.80)",
    pos:   d ? [240,161,50]  : [188,108,0],
    neg:   d ? [74,163,255]  : [9,105,218],
    accent:d ? "#4aa3ff" : "#0969da",
    amber: d ? "#f0a132" : "#bc6c00",
  };
}
function tint(v, max, P){
  const t = Math.min(1, Math.abs(v)/(max || 1));
  const rgb = v >= 0 ? P.pos : P.neg;
  return `rgba(${rgb[0]},${rgb[1]},${rgb[2]},${0.10 + 0.80*t})`;
}
const fmt = (v, d=2) => {
  if(!isFinite(v)) return "∞";
  if(Math.abs(v) < 5e-3) v = 0;
  const s = Math.abs(v - Math.round(v)) < 1e-9 ? String(Math.round(v)) : v.toFixed(d);
  return s === "-0" ? "0" : s;
};

function fitCanvas(cv, w, h){
  const dpr = devicePixelRatio || 1;
  cv.width = Math.round(w*dpr); cv.height = Math.round(h*dpr);
  cv.style.width = w+"px"; cv.style.height = h+"px";
  const g = cv.getContext("2d");
  g.setTransform(dpr,0,0,dpr,0,0);
  return g;
}

const HEAD = 17, TOPLBL = 13, GUT = 24;

function layout(){
  const avail = Math.max(360, (el("vizwrap").clientWidth || 900) - 2);
  const {N,C,H,Wd} = S;
  let best = null;
  for(const side of [true,false]){
    for(let cs=34; cs>=3; cs--){
      const g  = Math.max(3, Math.round(cs*0.4));
      const tw = GUT + C*Wd*cs + (C-1)*g;
      const th = HEAD + TOPLBL + N*H*cs + (N-1)*g;
      const w  = side ? tw*2 + 54 : tw;
      if(w <= avail){ best = {cs, g, tw, th, side}; break; }
    }
    if(best) break;
  }
  if(!best){
    const cs = 3, g = 3;
    best = {cs, g, tw: GUT + C*Wd*cs + (C-1)*g, th: HEAD + TOPLBL + N*H*cs + (N-1)*g, side:false};
  }
  best.ax = 0; best.ay = 0;
  best.bx = best.side ? best.tw + 54 : 0;
  best.by = best.side ? 0 : best.th + 26;
  L = best;
  L.w = best.side ? best.tw*2 + 54 : best.tw;
  L.h = best.side ? best.th : best.th*2 + 26;
}

function blockXY(ox, oy, n, c){
  return {x: ox + GUT + c*(S.Wd*L.cs + L.g), y: oy + HEAD + TOPLBL + n*(S.H*L.cs + L.g)};
}

function drawTensor(g, T, ox, oy, title, max, P, live){
  const {N,C,H,Wd} = S, cs = L.cs;
  g.fillStyle = P.dim; g.font = "600 11px ui-sans-serif,system-ui";
  g.textAlign = "left"; g.textBaseline = "alphabetic";
  g.fillText(title, ox, oy + 11);

  g.font = "10px ui-monospace,Menlo,monospace"; g.fillStyle = P.dim2;
  for(let c=0;c<C;c++){
    const b = blockXY(ox, oy, 0, c);
    if(Wd*cs > 14) g.fillText("c"+c, b.x, oy + HEAD + TOPLBL - 3);
  }
  g.textAlign = "right";
  for(let n=0;n<N;n++){
    const b = blockXY(ox, oy, n, 0);
    if(H*cs > 11) g.fillText("n"+n, ox + GUT - 5, b.y + H*cs/2 + 3);
  }
  g.textAlign = "center"; g.textBaseline = "middle";

  const showNums = cs >= 17;
  for(let n=0;n<N;n++) for(let c=0;c<C;c++){
    const b = blockXY(ox, oy, n, c);
    const on = !live || live.has(nc(n,c));
    for(let h=0;h<H;h++) for(let w=0;w<Wd;w++){
      const v = T[idx(n,c,h,w)];
      const x = b.x + w*cs, y = b.y + h*cs;
      g.fillStyle = P.panel; g.fillRect(x, y, cs, cs);
      g.fillStyle = tint(v, max, P);
      g.globalAlpha = on ? 1 : 0.16;
      g.fillRect(x, y, cs, cs);
      g.globalAlpha = 1;
      if(showNums){
        const txt = fmt(v, 1);
        g.fillStyle = on ? P.text : P.dim2;
        g.globalAlpha = on ? 1 : 0.45;
        g.font = `600 ${Math.min(11, Math.max(6, Math.floor(cs*0.95/Math.max(txt.length, 1.7))))}px ui-monospace,Menlo,monospace`;
        g.fillText(txt, x + cs/2, y + cs/2 + 0.5);
        g.globalAlpha = 1;
      }
    }
    g.strokeStyle = on ? P.accent : P.line;
    g.lineWidth = on ? 1.6 : 1;
    g.globalAlpha = on ? 1 : 0.5;
    g.strokeRect(b.x - .5, b.y - .5, Wd*cs + 1, H*cs + 1);
    g.globalAlpha = 1;
  }
}

function paint(){
  const P = palette();
  layout();
  const g = fitCanvas(el("viz"), L.w, L.h);
  g.clearRect(0,0,L.w,L.h);

  let mx = 0, my = 0;
  for(let i=0;i<X.length;i++){ mx = Math.max(mx, Math.abs(X[i])); my = Math.max(my, Math.abs(Y[i])); }

  const f = S.focus;
  const live = f ? new Set(GRP[GID[nc(f.n,f.c)]].members.map(m => nc(m.n,m.c))) : null;

  drawTensor(g, X, L.ax, L.ay, `x — input (${S.N}, ${S.C}, ${S.H}, ${S.Wd})`, mx, P, live);
  drawTensor(g, Y, L.bx, L.by, `y — normalised`, my, P, live);

  // the arrow between the two tensors
  g.fillStyle = P.dim; g.font = "11px ui-sans-serif,system-ui";
  g.textAlign = "center"; g.textBaseline = "middle";
  if(L.side){
    const x = L.tw + 27, y = L.ay + L.th/2;
    g.fillText("→", x, y - 8);
    g.font = "10px ui-monospace,Menlo,monospace";
    g.fillText("γ·x̂+β", x, y + 8);
  } else {
    g.fillText("↓  γ·x̂ + β", L.tw/2, L.th + 13);
  }

  // ring the hovered cell
  if(f){
    const ox = f.t === "x" ? L.ax : L.bx, oy = f.t === "x" ? L.ay : L.by;
    const b = blockXY(ox, oy, f.n, f.c);
    g.strokeStyle = P.amber; g.lineWidth = 2;
    g.strokeRect(b.x + f.w*L.cs - 1, b.y + f.h*L.cs - 1, L.cs + 2, L.cs + 2);
  }
  renderMath();
  paintDist(P, mx, my);
}

/* --------- every value of every group on one axis, before and after --------- */
const DROWS = 12;
function paintDist(P, mx, my){
  const shown = Math.min(GRP.length, DROWS);
  const avail = Math.max(320, (el("distwrap").clientWidth || 700) - 2);
  const gut = 62, gap = 22;
  const pw = Math.floor((avail - gut - gap)/2);
  const rh = 15, top = 20;
  const h = top + shown*rh + 10;
  const g = fitCanvas(el("dist"), avail, h);
  g.clearRect(0,0,avail,h);

  const spanX = Math.max(mx, 1e-6), spanY = Math.max(my, 1e-6);
  const px = (v, x0, span) => x0 + pw/2 + (v/span)*(pw/2 - 6);

  g.font = "10px ui-sans-serif,system-ui"; g.fillStyle = P.dim2;
  g.textAlign = "center"; g.textBaseline = "middle";
  g.fillText(`before  (±${fmt(spanX,1)})`, gut + pw/2, 9);
  g.fillText(`after  (±${fmt(spanY,1)})`, gut + pw + gap + pw/2, 9);

  const fi = S.focus ? GID[nc(S.focus.n, S.focus.c)] : -1;
  for(let i=0;i<shown;i++){
    const grp = GRP[i], y = top + i*rh + rh/2;
    const on = i === fi || fi < 0;
    g.textAlign = "right"; g.fillStyle = on ? P.text : P.dim2;
    g.globalAlpha = on ? 1 : .45;
    g.font = "10px ui-monospace,Menlo,monospace";
    g.fillText(grp.label, gut - 6, y);

    [[0, X, spanX, P.pos], [1, Y, spanY, P.neg]].forEach(([side, T, span, rgb]) => {
      const x0 = gut + side*(pw + gap);
      g.strokeStyle = P.line; g.lineWidth = 1;
      g.beginPath(); g.moveTo(x0, y + 5.5); g.lineTo(x0 + pw, y + 5.5); g.stroke();
      g.strokeStyle = P.dim2;                              // the zero mark
      g.beginPath(); g.moveTo(px(0, x0, span), y - 5); g.lineTo(px(0, x0, span), y + 5.5); g.stroke();
      g.fillStyle = `rgba(${rgb[0]},${rgb[1]},${rgb[2]},${on ? .7 : .2})`;
      for(const m of grp.members) for(let hh=0;hh<S.H;hh++) for(let ww=0;ww<S.Wd;ww++){
        const v = T[idx(m.n, m.c, hh, ww)];
        g.beginPath(); g.arc(px(v, x0, span), y, 2.1, 0, 6.2832); g.fill();
      }
    });
    g.globalAlpha = 1;
  }

  const b = S.aff === "learned";
  el("distnote").innerHTML =
    `Each dot is one number of the tensor. Left: the raw values, one row per pooling group — the groups sit at
     different centres and have different spreads. Right: the same numbers after the layer${
       b ? `, rescaled by the learned <span class="pv">γ</span>, <span class="pv">β</span>` : ""}.
     ${b ? "Turn affine off" : "With affine off or at init"} every row is centred on 0 with variance 1 —
     that <em>is</em> the whole operation.
     ${GRP.length > DROWS ? `<br>Showing the first ${DROWS} of ${GRP.length} groups.` : ""}`;
}

/* ======================= the formula, written out ======================= */
const xv = v => `<span class="xv">${fmt(v)}</span>`;
const sv = v => `<span class="sv">${fmt(v)}</span>`;
const pv = v => `<span class="pv">${fmt(v)}</span>`;
function mrow(label, html){ return `<div class="mrow"><span class="mlbl">${label}</span><span class="mval">${html}</span></div>`; }

function groupWhy(g){
  const {N,C,H,Wd} = S;
  switch(S.norm){
    case "batch":
      return `channel <b>${g.members[0].c}</b> — one mean for the whole channel, pooling
              <b>N·H·W = ${N}·${H}·${Wd} = ${g.count}</b> values across every sample in the batch.`;
    case "layer":
      return `sample <b>${g.members[0].n}</b> — one mean for that sample's entire feature stack, pooling
              <b>C·H·W = ${C}·${H}·${Wd} = ${g.count}</b> values. No other sample is involved.`;
    case "instance":
      return `sample <b>${g.members[0].n}</b>, channel <b>${g.members[0].c}</b> — one mean per feature map,
              pooling <b>H·W = ${H}·${Wd} = ${g.count}</b> values.`;
    default: {
      const cs = g.members.map(m => m.c);
      return `sample <b>${g.members[0].n}</b>, channels <b>${Math.min(...cs)}…${Math.max(...cs)}</b> — one mean per
              group of ${S.C/S.G} channels, pooling <b>(C/G)·H·W = ${g.count}</b> values.`;
    }
  }
}

function renderMath(){
  const f = S.focus;
  if(!f){ el("readout").innerHTML = `<span style="color:var(--dim2)">Hover a cell to see the arithmetic that produced it.</span>`; return; }
  const {n,c,h,w} = f;
  const g  = GRP[GID[nc(n,c)]];
  const mu = MU[nc(n,c)], va = VA[nc(n,c)];
  const x  = X[idx(n,c,h,w)], y = Y[idx(n,c,h,w)];
  const sd = Math.sqrt(va + eps());
  const frozen = S.norm === "batch" && S.mode === "eval";
  const src = frozen ? "running_" : "";
  const E = S.epsExp === 0 ? "1" : `1e${S.epsExp}`;

  el("readout").innerHTML =
    mrow("this cell", `<b>y[${n}][${c}][${h}][${w}]</b>`) +
    mrow("formula",
      `<span class="pv">γ[${c}]</span> · ( <span class="xv">x[${n}][${c}][${h}][${w}]</span> −
       <span class="sv">${src}μ[${c}]</span> ) / √( <span class="sv">${src}σ²[${c}]</span> + ε )
       + <span class="pv">β[${c}]</span>`) +
    mrow("values",
      `${pv(GAM[c])} · ( ${xv(x)} − ${sv(mu)} ) / √( ${sv(va)} + ${E} ) + ${pv(BET[c])}`) +
    mrow("maths",
      `${pv(GAM[c])} · ${fmt(x-mu)} / ${fmt(sd, 3)} + ${pv(BET[c])}
       &nbsp;=&nbsp; <span class="res">${fmt(y)}</span>`) +
    mrow(frozen ? "eval" : "why these",
      frozen
        ? `<span style="color:var(--dim)">eval() ignores this batch entirely — <span class="sv">running_μ</span> and
           <span class="sv">running_σ²</span> come from the ${S.steps} batch${S.steps===1?"":"es"} fed so far.</span>`
        : `<span style="color:var(--dim)">${groupWhy(g)}</span>`);
}

/* ============================== side panels ============================== */
function counts(N,C,H,W){
  switch(S.norm){
    case "batch":    return {stats:C,   pool:N*H*W,       expr:"N·H·W"};
    case "layer":    return {stats:N,   pool:C*H*W,       expr:"C·H·W"};
    case "instance": return {stats:N*C, pool:H*W,         expr:"H·W"};
    default:         return {stats:N*S.G, pool:(C/S.G)*H*W, expr:"(C/G)·H·W"};
  }
}
const th = v => v.toLocaleString("en-US");

function renderFormula(){
  const N = calc("N"), C = calc("C"), H = calc("H"), W = calc("W");
  const q = counts(N,C,H,W);
  const learn = S.aff === "off" ? 0 : (S.norm === "layer" ? 2*C*H*W : 2*C);
  const buf   = S.norm === "batch" ? 2*C : 0;
  const name  = ({batch:`BatchNorm2d(${C})`, layer:`LayerNorm([${C}, ${H}, ${W}])`,
                  instance:`InstanceNorm2d(${C})`, group:`GroupNorm(${S.G}, ${C})`})[S.norm];

  el("formula").innerHTML =
    `<span class="lbl">pytorch</span>${name}
     <span class="lbl">shape</span>(<b>${N}</b>, <b>${C}</b>, <b>${H}</b>, <b>${W}</b>) →
       (<b>${N}</b>, <b>${C}</b>, <b>${H}</b>, <b>${W}</b>) <span style="color:var(--dim2)">unchanged</span>
     <span class="lbl">statistics</span><span class="big">${th(q.stats)}</span> pairs of (μ, σ²)
     <span class="lbl">each pools</span>${q.expr} = <b>${th(q.pool)}</b> values
     <span class="lbl">learnable</span>${learn ? `γ, β = ${th(learn)} floats` : "none (affine off)"}
     <span class="lbl">buffers</span>${buf ? `running_mean, running_var = ${th(buf)} floats` : "none"}`;

  const notes = [];
  if(S.norm === "batch")
    notes.push(`Normalisation never changes the shape — the interesting number is <b>${th(q.pool)}</b>,
                how many values each mean is estimated from. Halve the batch and that estimate gets noisier;
                that is the whole reason BatchNorm struggles with small batches.`);
  else
    notes.push(`Every statistic is computed inside one sample, so the batch size does not enter at all —
                each mean still pools <b>${th(q.pool)}</b> values whether N is 1 or 1000.`);
  if(S.norm === "layer")
    notes.push(`PyTorch's <code>LayerNorm</code> gives γ and β one entry per element of
                <code>normalized_shape</code>, hence ${th(learn)} and not ${th(2*C)}. This page applies
                per-channel γ, β for all four layers so the only thing that changes is the pooling.`);
  if(!calcLinked())
    notes.push(`<span style="color:var(--dim2)">The tensor drawn on the left is
                ${S.N}×${S.C}×${S.H}×${S.Wd} — these numbers are for the shape you typed.
                <em>↺ drawn</em> puts them back in sync.</span>`);
  el("shapenote").innerHTML = notes.join("<br><br>");
}

function renderStatTable(){
  const fi = S.focus ? GID[nc(S.focus.n, S.focus.c)] : -1;
  const showRun = S.norm === "batch";
  const rows = GRP.slice(0, 16).map((g,i) => `
    <tr class="${i===fi?'on':''}">
      <td>${g.label}</td><td>${g.count}</td><td>${fmt(g.mean)}</td><td>${fmt(g.var)}</td>
      ${showRun ? `<td>${fmt(RM[g.members[0].c])}</td><td>${fmt(RV[g.members[0].c])}</td>` : ""}
    </tr>`).join("");
  el("stattable").innerHTML = `
    <table class="stats">
      <tr><th>group</th><th>n</th><th>μ</th><th>σ²</th>
        ${showRun ? `<th>run μ</th><th>run σ²</th>` : ""}</tr>
      ${rows}
    </table>
    ${GRP.length > 16 ? `<div class="note">…and ${GRP.length-16} more.</div>` : ""}
    <div class="note">${showRun
      ? `<b>μ, σ²</b> are this batch's; <b>run μ, run σ²</b> are the running estimates after
         ${S.steps} training batch${S.steps===1?"":"es"}. <code>train()</code> uses the first pair,
         <code>eval()</code> the second.`
      : `There are no running statistics — this layer recomputes μ and σ² from whatever tensor it is given,
         so <code>train()</code> and <code>eval()</code> behave identically.`}</div>`;
}

function renderDiag(){
  const {N,C,H,Wd} = S;
  const q = counts(N,C,H,Wd);
  const batchDep = S.norm === "batch";
  const out = [];

  out.push(`<div style="margin-bottom:7px">
    <span class="flag ${batchDep?'bad':'ok'}">${batchDep?'batch-dependent':'per-sample'}</span>
    <span style="color:var(--dim);font-size:11.5px;margin-left:7px">${q.stats} statistic${q.stats===1?"":"s"},
      ${q.pool} value${q.pool===1?"":"s"} each</span></div>`);

  const notes = [];
  if(batchDep){
    notes.push(`Each output depends on the <em>other samples in the batch</em>. That is why the layer has to
                keep running statistics for inference, and why <code>train()</code> and <code>eval()</code>
                give different numbers for the same input.`);
    if(N*H*Wd === 1)
      notes.push(`<span class="err">N·H·W = 1 — PyTorch raises "Expected more than 1 value per channel when
                  training".</span> With a single value per channel the variance is 0 and the output is
                  entirely β.`);
    else if(q.pool < 8)
      notes.push(`Only ${q.pool} values per statistic: the estimates are noisy, and the layer effectively
                  injects that noise into training. This is the small-batch failure mode that GroupNorm was
                  designed for.`);
  } else {
    notes.push(`Nothing crosses the batch dimension, so inference on a single sample gives exactly the same
                answer as inference in a batch of 64. No running statistics, no train/eval divergence.`);
  }

  if(S.norm === "group"){
    if(S.G === 1)      notes.push(`<b>G = 1</b> pools all channels together — that is LayerNorm over (C, H, W).`);
    else if(S.G === C) notes.push(`<b>G = C</b> gives every channel its own group — that is InstanceNorm.`);
    else               notes.push(`GroupNorm interpolates: <b>G = 1</b> is LayerNorm, <b>G = C</b> is InstanceNorm.`);
  }

  const zero = GRP.find(g => g.var < 1e-9);
  if(zero)
    notes.push(`Group <b>${zero.label}</b> has σ² = 0, so x − μ = 0 for every one of its values and the output is
                exactly β no matter what ε is. ε only keeps the division finite —
                <b>1/√(0 + ${S.epsExp===0?"1":"1e"+S.epsExp})</b> = ${fmt(1/Math.sqrt(eps()),1)}.`);
  else if(S.epsExp >= -2)
    notes.push(`ε = ${S.epsExp===0?"1":"1e"+S.epsExp} is large enough to bite: it inflates the denominator, so
                the output variance lands below 1. At the default 1e-5 it is invisible.`);

  if(S.aff === "learned")
    notes.push(`γ and β are free parameters. They can undo the normalisation completely (γ = σ, β = μ), which is
                exactly why adding the layer cannot make the network less expressive.`);

  out.push(`<div class="note" style="margin-top:0">${notes.join("<br><br>")}</div>`);
  el("diag").innerHTML = out.join("");
}

/* ============================== controls ============================== */
function buildControls(){
  el("ctrls").innerHTML = DIMS.map(P => `
    <div class="ctrl">
      <label for="bn_r_${P.id}"><span class="${P.cls}">${P.name}</span><b id="bn_v_${P.id}"></b></label>
      <input type="range" id="bn_r_${P.id}" min="1" max="10" step="1">
      <div class="hint">${P.hint}</div>
    </div>`).join("");

  DIMS.forEach(P => el("r_"+P.id).addEventListener("input", e => {
    S[P.id] = +e.target.value;
    relink(); fixGroups(); S.focus = null;
    if(P.id === "C") resetRunning();
    refresh(true);
  }));

  const seg = (id, attr, fn) => document.querySelectorAll(`#bn_${id} button`).forEach(b =>
    b.addEventListener("click", () => { fn(b.dataset[attr]); }));

  seg("normseg", "norm", v => { S.norm = v; fixGroups(); S.focus = null; refresh(); });
  seg("epsseg",  "eps",  v => { S.epsExp = +v; refresh(); });
  seg("affseg",  "aff",  v => { S.aff = v; refresh(true); });
  seg("modeseg", "mode", v => { S.mode = v; refresh(); });
  seg("momseg",  "mom",  v => { S.mom = +v; refresh(); });

  el("feed").addEventListener("click", feedBatch);
  el("reset").addEventListener("click", () => { resetRunning(); refresh(); });
  el("dead").addEventListener("change", e => { S.dead = e.target.checked; refresh(true); });

  document.querySelectorAll("#view-norm [data-preset]").forEach(b => b.addEventListener("click", () => {
    Object.assign(S, JSON.parse(b.dataset.preset));
    relink(); fixGroups(); S.focus = null; resetRunning(); refresh(true);
  }));

  ["dN","dC","dH","dW"].forEach(id => el(id).addEventListener("change", () =>
    setCalc(+el("dN").value, +el("dC").value, +el("dH").value, +el("dW").value)));
  el("dsync").addEventListener("click", () => { relink(); refresh(); });
}

function relink(){ S.calc = {N:null, C:null, H:null, W:null}; }

// dims small enough to draw are applied to the tensor; anything else lives in
// the calculator alone.
function setCalc(N, C, H, W){
  const cl = v => Math.max(1, Math.min(8192, Math.round(v) || 1));
  [N,C,H,W] = [cl(N), cl(C), cl(H), cl(W)];
  if([N,C,H,W].every(v => v <= 10)){
    relink();
    Object.assign(S, {N, C, H, Wd:W});
    fixGroups(); S.focus = null; resetRunning(); refresh(true);
  } else {
    S.calc = {N, C, H, W};
    fixGroups(); refresh();
  }
}

const divisors = n => Array.from({length:n}, (_,i) => i+1).filter(d => n % d === 0);
function fixGroups(){
  const ds = divisors(calc("C"));
  if(!ds.includes(S.G)) S.G = ds.reduce((a,b) => Math.abs(b-S.G) < Math.abs(a-S.G) ? b : a, ds[0]);
}

function syncControls(){
  DIMS.forEach(P => { el("r_"+P.id).value = S[P.id]; el("v_"+P.id).textContent = S[P.id]; });
  const on = (id, attr, val) => document.querySelectorAll(`#bn_${id} button`).forEach(b =>
    b.classList.toggle("on", b.dataset[attr] === String(val)));
  on("normseg","norm",S.norm); on("epsseg","eps",S.epsExp);
  on("affseg","aff",S.aff);    on("modeseg","mode",S.mode); on("momseg","mom",S.mom);

  el("groupwrap").style.display = S.norm === "group" ? "" : "none";
  el("groupseg").innerHTML = divisors(calc("C")).map(d =>
    `<button data-g="${d}" class="${d===S.G?'on':''}">${d}</button>`).join("");
  document.querySelectorAll("#bn_groupseg button").forEach(b =>
    b.addEventListener("click", () => { S.G = +b.dataset.g; S.focus = null; refresh(); }));

  el("normhint").innerHTML = ({
    batch:    `<code>nn.BatchNorm2d</code> — one μ, σ² per <b>channel</b>, pooled over the batch and all pixels.`,
    layer:    `<code>nn.LayerNorm</code> — one μ, σ² per <b>sample</b>, pooled over channels and pixels.`,
    instance: `<code>nn.InstanceNorm2d</code> — one μ, σ² per <b>feature map</b>, pooled over pixels only.`,
    group:    `<code>nn.GroupNorm</code> — one μ, σ² per <b>group of channels</b>, inside one sample.`,
  })[S.norm];

  el("affhint").innerHTML = ({
    off:     `y is exactly the normalised x̂. PyTorch calls this <code>affine=False</code>.`,
    init:    `γ = 1, β = 0 — how PyTorch initialises them, so the layer starts as pure normalisation.`,
    learned: `Arbitrary trained values: the layer can now shift and stretch each channel back out again.`,
  })[S.aff];

  const isBN = S.norm === "batch";
  el("runwrap").style.display = isBN ? "" : "none";
  document.querySelectorAll("#bn_modeseg button").forEach(b => b.disabled = !isBN && b.dataset.mode === "eval");
  el("runhint").innerHTML = !isBN ? "" : S.mode === "train"
    ? `Each batch nudges the estimates: <span class="mono">running = (1−m)·running + m·batch</span>.
       ${S.steps} fed so far.`
    : `<b>eval() freezes them.</b> Feeding batches no longer changes running_mean or running_var — that is the
       point, inference must not depend on whatever else is in the batch.`;
  el("dead").checked = S.dead;

  const c = S.calc;
  el("dN").value = calc("N"); el("dC").value = calc("C");
  el("dH").value = calc("H"); el("dW").value = calc("W");
  el("dsync").style.display = calcLinked() ? "none" : "";
}

/* ============================== interaction ============================== */
function hit(ev){
  const rect = el("viz").getBoundingClientRect();
  const x = ev.clientX - rect.left, y = ev.clientY - rect.top;
  const cs = L.cs, bw = S.Wd*cs, bh = S.H*cs;
  for(const [t, ox, oy] of [["x", L.ax, L.ay], ["y", L.bx, L.by]]){
    const lx = x - (ox + GUT), ly = y - (oy + HEAD + TOPLBL);
    if(lx < 0 || ly < 0) continue;
    const c = Math.floor(lx/(bw + L.g)), n = Math.floor(ly/(bh + L.g));
    if(c >= S.C || n >= S.N) continue;
    const ix = lx - c*(bw + L.g), iy = ly - n*(bh + L.g);
    if(ix >= bw || iy >= bh) continue;
    return {t, n, c, h: Math.floor(iy/cs), w: Math.floor(ix/cs)};
  }
  return null;
}

function wire(){
  el("viz").addEventListener("mousemove", e => {
    const h = hit(e);
    if(!h) return;                                  // between blocks: keep the last reading
    const f = S.focus;
    if(f && f.t===h.t && f.n===h.n && f.c===h.c && f.h===h.h && f.w===h.w) return;
    S.focus = h; paint(); renderStatTable();
  });
  const hidden = () => document.getElementById("view-norm").classList.contains("hidden");
  let t; addEventListener("resize", () => { clearTimeout(t); t = setTimeout(() => { if(!hidden()) paint(); }, 120); });
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => { if(!hidden()) refresh(); });
}

/* ================================ main ================================ */
// open on the channel whose mean is furthest from zero — the one where
// subtracting μ visibly does something
function bestExample(){
  let best = 0, score = -1;
  for(let i=0;i<GRP.length;i++){
    const s = Math.abs(GRP[i].mean);
    if(s > score){ score = s; best = i; }
  }
  const m = GRP[best].members[0];
  return {t:"x", n:m.n, c:m.c, h:0, w:0};
}

function refresh(rebuild){
  if(!RM || RM.length !== S.C) resetRunning();
  if(rebuild || !X || X.length !== S.N*S.C*S.H*S.Wd) buildData();
  buildAffine();
  buildGroups();
  normalise();
  if(S.focus && (S.focus.n >= S.N || S.focus.c >= S.C || S.focus.h >= S.H || S.focus.w >= S.Wd)) S.focus = null;
  if(!S.focus) S.focus = bestExample();
  syncControls();
  renderFormula();
  renderStatTable();
  renderDiag();
  paint();
}

let started = false;
return {
  show(){
    if(!started){ started = true; buildControls(); wire(); refresh(true); }
    else refresh();
  }
};
})();
