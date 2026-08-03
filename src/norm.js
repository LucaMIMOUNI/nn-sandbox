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
  cs:0,                          // training step shown in the covariate-shift panel
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

let X = null, XH = null, Y = null;   // Float32Array N*C*H*W — raw, normalised, after affine
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

// gamma and beta are state, not something recomputed every frame — otherwise the
// sliders would spring back the moment anything else changed
function buildAffine(reset){
  if(!reset && GAM && GAM.length === S.C) return;
  GAM = new Float32Array(S.C).fill(1);
  BET = new Float32Array(S.C);
  if(S.aff !== "learned") return;
  const r = prng(555);
  for(let c=0;c<S.C;c++){ GAM[c] = r1(0.4 + r()*1.6); BET[c] = r1(r()*4 - 2); }
}

// dragging one channel's gamma or beta puts the control into "custom"
function setAffine(which, v){
  if(S.aff === "off") return;
  const c = S.focus ? S.focus.c : 0;
  if(which === "gam") GAM[c] = v; else BET[c] = v;
  S.aff = "custom";
  refresh();
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
  Y  = new Float32Array(N*C*H*Wd);
  XH = new Float32Array(N*C*H*Wd);
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
      XH[i] = (X[i] - mu)*inv;                // normalised, before the affine step
      Y[i]  = GAM[c]*XH[i] + BET[c];
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
    frame: d ? "#12171e" : "#f7f9fb",
    frameLine: d ? "#303a47" : "#cbd5df",
    vio:   d ? [210,168,255] : [130, 80,223],
    vioCss:d ? "#d2a8ff" : "#8250df",
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

const HEAD = 17, TOPLBL = 14, GUT = 40, FPAD = 5;

function roundRect(g, x, y, w, h, r){
  r = Math.min(r, w/2, h/2);
  g.beginPath();
  g.moveTo(x+r, y);
  g.arcTo(x+w, y, x+w, y+h, r);
  g.arcTo(x+w, y+h, x, y+h, r);
  g.arcTo(x, y+h, x, y, r);
  g.arcTo(x, y, x+w, y, r);
  g.closePath();
}

/* ---------- layout ----------
   One tensor, not two. Each **sample** gets a box of its own; the **channels**
   inside it only get a gap. That asymmetry is the whole picture: which numbers
   live in the same sample, versus which merely live in the same channel across
   different samples. Getting those two confused is what makes BatchNorm hard.
   ---------------------------------------------------------------------- */
function layout(){
  const avail = Math.max(360, (el("vizwrap").clientWidth || 900) - 2);
  const {N,C,H,Wd} = S;
  let best = null;
  for(let cs=34; cs>=3; cs--){
    const gc = Math.max(2, Math.round(cs*0.34));          // between channels — small
    const gn = Math.max(10, Math.round(cs*0.8));          // between samples  — large
    const fw = 2*FPAD + C*Wd*cs + (C-1)*gc;
    if(GUT + fw <= avail){ best = {cs, gc, gn, fw, fh: 2*FPAD + H*cs}; break; }
  }
  if(!best){
    const cs = 3, gc = 2;
    best = {cs, gc, gn:10, fw: 2*FPAD + C*Wd*cs + (C-1)*gc, fh: 2*FPAD + H*cs};
  }
  best.w = Math.max(GUT + best.fw, 200);   // never clip the title on a tiny tensor
  best.h = HEAD + TOPLBL + N*best.fh + (N-1)*best.gn;
  L = best;
}
const frameXY = n => ({x: GUT, y: HEAD + TOPLBL + n*(L.fh + L.gn)});
function blockXY(n, c){
  const f = frameXY(n);
  return {x: f.x + FPAD + c*(S.Wd*L.cs + L.gc), y: f.y + FPAD};
}

function drawX(g, P, live){
  const {N,C,H,Wd} = S, cs = L.cs;
  let max = 0;
  for(let i=0;i<X.length;i++) max = Math.max(max, Math.abs(X[i]));

  g.textAlign = "left"; g.textBaseline = "alphabetic";
  g.fillStyle = P.dim; g.font = "600 11px ui-sans-serif,system-ui";
  g.fillText(`x — input (${N}, ${C}, ${H}, ${Wd})`, 0, 11);

  g.font = "600 10px ui-monospace,Menlo,monospace";           // the channel ruler
  for(let c=0;c<C;c++){
    const b = blockXY(0, c);
    if(Wd*cs > 14){
      const on = !live || Array.from({length:N}, (_,n) => nc(n,c)).some(k => live.has(k));
      g.fillStyle = on ? P.accent : P.dim2;
      g.globalAlpha = on ? 1 : .5;
      g.fillText("c"+c, b.x, HEAD + TOPLBL - 5);
      g.globalAlpha = 1;
    }
  }

  const showNums = cs >= 17;
  for(let n=0;n<N;n++){
    const f = frameXY(n);
    const anyLive = !live || Array.from({length:C}, (_,c) => nc(n,c)).some(k => live.has(k));

    // the sample container — the batch dimension, made a physical object
    g.globalAlpha = anyLive ? 1 : .5;
    g.fillStyle = P.frame;
    roundRect(g, f.x, f.y, L.fw, L.fh, 7); g.fill();
    g.strokeStyle = anyLive ? P.frameLine : P.line; g.lineWidth = 1; g.stroke();
    g.fillStyle = P.amber;                                    // batch-coloured spine
    roundRect(g, f.x, f.y + 2, 3, L.fh - 4, 1.5); g.fill();
    g.globalAlpha = 1;

    g.textAlign = "right"; g.textBaseline = "middle";
    g.font = "600 11px ui-monospace,Menlo,monospace";
    g.fillStyle = anyLive ? P.amber : P.dim2;
    g.fillText("n"+n, f.x - 8, f.y + L.fh/2);
    g.textAlign = "center";

    for(let c=0;c<C;c++){
      const b = blockXY(n, c);
      const on = !live || live.has(nc(n,c));
      for(let h=0;h<H;h++) for(let w=0;w<Wd;w++){
        const v = X[idx(n,c,h,w)];
        const x = b.x + w*cs, y = b.y + h*cs;
        g.fillStyle = P.panel; g.fillRect(x, y, cs, cs);
        g.fillStyle = tint(v, max, P);
        g.globalAlpha = on ? 1 : 0.14;
        g.fillRect(x, y, cs, cs);
        g.globalAlpha = 1;
        if(showNums){
          const txt = fmt(v, 1);
          g.fillStyle = on ? P.text : P.dim2;
          g.globalAlpha = on ? 1 : 0.4;
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
  g.textAlign = "left"; g.textBaseline = "alphabetic";
}

function paint(){
  const P = palette();
  layout();
  const g = fitCanvas(el("viz"), L.w, L.h);
  g.clearRect(0,0,L.w,L.h);

  const f = S.focus;
  const live = f ? new Set(GRP[GID[nc(f.n,f.c)]].members.map(m => nc(m.n,m.c))) : null;
  drawX(g, P, live);

  if(f){                                                    // ring the hovered cell
    const b = blockXY(f.n, f.c);
    g.strokeStyle = P.text; g.lineWidth = 2;
    g.strokeRect(b.x + f.w*L.cs - 1.5, b.y + f.h*L.cs - 1.5, L.cs + 3, L.cs + 3);
  }
  renderMath();
  renderStepFormula();
  paintFocus(P);
  paintDist(P);
  paintCS(P);
}

/* ================= covariate shift — the moving target =================
   The problem BatchNorm was introduced to solve. Training changes the weights
   of every layer at once, so the distribution a layer *receives* keeps moving
   underneath it: it spends its updates chasing that instead of learning. The
   original paper called this internal covariate shift.

   Two things make it expensive, and both are on screen. The distribution
   drifts off the range where the next non-linearity has any gradient left, and
   it keeps moving, so nothing the layer learned about the old range still
   holds. Normalising pins both: whatever arrives, what leaves is (β, γ).

   (Honesty: later work — Santurkar et al., 2018 — showed BatchNorm helps even
   when shift is injected deliberately, and credits a smoother loss surface.
   The pinning below is real either way; the causal story is not settled.)
   ====================================================================== */
const CS_T = 16;              // training steps drawn
const CS_BAND = 2.5;          // |z| where a sigmoid still keeps ~28% of its best gradient
const CS_ROW = 13, CS_HEAD = 30, CS_FOOT = 24, CS_GUT = 54, CS_GAP = 26;

// Abramowitz & Stegun 7.1.26 — good to 1.5e-7, plenty for a percentage
function erf(x){
  const s = x < 0 ? -1 : 1;
  x = Math.abs(x);
  const t = 1/(1 + 0.3275911*x);
  const y = 1 - ((((1.061405429*t - 1.453152027)*t + 1.421413741)*t
                  - 0.284496736)*t + 0.254829592)*t*Math.exp(-x*x);
  return s*y;
}
const ncdf   = (v, m, sd) => 0.5*(1 + erf((v - m)/(sd*Math.SQRT2)));
const inBand = (m, sd) => ncdf(CS_BAND, m, sd) - ncdf(-CS_BAND, m, sd);

// what one channel hands the next layer at training step t. Not a model of a
// real network — a seeded walk with a slow push outward, which is what a
// layer's input does while the weights below it are still moving.
function drift(t){
  const r = prng(424242);
  let mu = 0.15, sd = 1;
  for(let i=0;i<t;i++){
    mu += 0.45*gauss(r) + 0.10;
    sd  = Math.max(0.15, sd*(1 + 0.09*gauss(r) + 0.09));
  }
  return {mu, sd};
}

function paintCS(P){
  const W = Math.max(660, (el("cswrap").clientWidth || 900) - 2);
  const pw = Math.floor((W - CS_GUT - CS_GAP)/2);
  const H = CS_HEAD + CS_T*CS_ROW + CS_FOOT;
  const g = fitCanvas(el("cs"), W, H);
  g.clearRect(0, 0, W, H);

  const c = S.focus ? S.focus.c : 0;
  const gam = S.aff === "off" ? 1 : GAM[c], bet = S.aff === "off" ? 0 : BET[c];
  const after = {mu: bet, sd: Math.max(Math.abs(gam), 0.05)};

  // one range for both panels — comparing them is the whole point
  let R = CS_BAND + 0.6;
  for(let t=0;t<CS_T;t++){
    const d = drift(t);
    R = Math.max(R, Math.abs(d.mu) + 2.6*d.sd);
  }
  R = Math.max(R, Math.abs(after.mu) + 2.6*after.sd);

  const sides = [
    {x0: CS_GUT, head: "without BatchNorm", rgb: P.pos, at: t => drift(t)},
    {x0: CS_GUT + pw + CS_GAP, head: `with BatchNorm  (γ ${fmt(gam)}, β ${fmt(bet)})`,
     rgb: P.vio, at: () => after},
  ];

  g.textAlign = "center"; g.textBaseline = "alphabetic";
  for(const s of sides){
    const px = v => s.x0 + pw/2 + (v/R)*(pw/2 - 6);

    g.font = "600 10px ui-sans-serif,system-ui"; g.fillStyle = P.dim;
    g.fillText(s.head, s.x0 + pw/2, 11);

    // the band where a sigmoid still has a gradient worth passing back
    g.fillStyle = `rgba(${P.neg[0]},${P.neg[1]},${P.neg[2]},.10)`;
    g.fillRect(px(-CS_BAND), CS_HEAD - 8, px(CS_BAND) - px(-CS_BAND), CS_T*CS_ROW + 10);
    g.font = "9px ui-sans-serif,system-ui"; g.fillStyle = P.dim2;
    g.fillText("gradient still flows here", s.x0 + pw/2, CS_HEAD - 12);

    for(let t=0;t<CS_T;t++){
      const {mu, sd} = s.at(t), y = CS_HEAD + t*CS_ROW + CS_ROW/2;
      const on = t === S.cs;
      const a = on ? .85 : .26;
      g.fillStyle = `rgba(${s.rgb[0]},${s.rgb[1]},${s.rgb[2]},${a})`;
      const lo = px(mu - 2*sd), hi = px(mu + 2*sd);
      roundRect(g, lo, y - 3.5, Math.max(2, hi - lo), 7, 3.5); g.fill();
      g.fillStyle = `rgba(${s.rgb[0]},${s.rgb[1]},${s.rgb[2]},${on ? 1 : .5})`;
      g.fillRect(px(mu) - .5, y - 6, 1, 12);                  // where the centre sits
    }

    g.strokeStyle = P.line; g.lineWidth = 1;                  // the axis, and zero
    const ay = CS_HEAD + CS_T*CS_ROW + 5;
    g.beginPath(); g.moveTo(s.x0, ay); g.lineTo(s.x0 + pw, ay); g.stroke();
    g.strokeStyle = P.dim2;
    g.beginPath(); g.moveTo(px(0), ay - 3); g.lineTo(px(0), ay + 3); g.stroke();
    g.font = "9px ui-monospace,Menlo,monospace"; g.fillStyle = P.dim2;
    g.fillText("0", px(0), ay + 14);
    g.fillText(`−${CS_BAND}`, px(-CS_BAND), ay + 14);
    g.fillText(`+${CS_BAND}`, px(CS_BAND), ay + 14);
  }

  g.textAlign = "right"; g.font = "9.5px ui-monospace,Menlo,monospace";
  for(let t=0;t<CS_T;t++){
    const on = t === S.cs;
    g.fillStyle = on ? P.text : P.dim2;
    g.fillText(on ? `step ${t}` : String(t), CS_GUT - 8, CS_HEAD + t*CS_ROW + CS_ROW/2 + 3.5);
  }
  g.textAlign = "left";

  const d = drift(S.cs), pb = inBand(d.mu, d.sd), pa = inBand(after.mu, after.sd);
  el("csnote").innerHTML =
    `Each row is one training step, and each bar is where <b>one channel's values</b> sit as they arrive
     at the next layer — the bar spans μ ± 2σ, the tick is μ. Nothing below has been trained here; the
     drift is a stand-in for what the layers underneath do to their output while <em>they</em> are being
     trained.
     <br><b style="color:var(--fg)">Left, the problem.</b> The bars walk off and widen. At step
     <b>${S.cs}</b> this channel arrives at μ ${fmt(d.mu)}, σ ${fmt(d.sd)}, which leaves
     <b class="${pb < 0.4 ? "err" : ""}">${fmt(100*pb, 1)}%</b> of its values in the shaded band — the
     part of a sigmoid with a gradient worth passing back. It was ${fmt(100*inBand(0.15, 1), 1)}% at
     step 0. The next layer is being asked to learn from a distribution that will not be there next
     step, through a non-linearity that has gone flat.
     <br><b style="color:var(--fg)">Right, the fix.</b> Every row is identical, because it has to be:
     the output of the normalising step is <b>(β, γ)</b> = (${fmt(bet)}, ${fmt(gam)}) whatever came in,
     so <b>${fmt(100*pa, 1)}%</b> stay in the band at <em>every</em> step. The layer above now sees the
     same kind of input on step 15 as on step 0, which is what lets it use a larger learning rate and
     stop re-adapting.
     ${Math.abs(gam) > 2.2
       ? `<br>γ = ${fmt(gam)} is wide enough to push values back out of the band by itself — the
          protection is only as good as what training does with γ and β.`
       : ""}`;
}

/* --------- the selected number, and only that one, on its way through ---------
   The whole output tensor told you nothing you could read. One value, in its
   three states, tells you the entire operation.
   ------------------------------------------------------------------------- */
function groupStats(T, grp){
  let s = 0, k = 0;
  for(const m of grp.members) for(let h=0;h<S.H;h++) for(let w=0;w<S.Wd;w++){ s += T[idx(m.n,m.c,h,w)]; k++; }
  const mean = s/k;
  let ss = 0;
  for(const m of grp.members) for(let h=0;h<S.H;h++) for(let w=0;w<S.Wd;w++){
    const d = T[idx(m.n,m.c,h,w)] - mean; ss += d*d;
  }
  return {mean, sd: Math.sqrt(ss/k)};
}

const COL_W = 150, ARROW_W = 138, TILE_W = 92, TILE_H = 46;
const TITLE_Y = 11, TILE_Y = 17, NOTE_Y = TILE_Y + TILE_H + 14;
const BELL_TOP = 92, AXIS_Y = 154, STATS_Y = 172, FOCUS_H = 186;

/* ---------- what the numbers actually do, and what a normal would say ----------
   The solid curve is a kernel density estimate: every value gets a small normal
   dropped on it and they are summed. So it is bell-like, but it follows the real
   numbers — a shoulder where three of them cluster, a thinner tail on the side
   with fewer. Twenty-seven samples of a genuine normal never make a clean bell,
   and pretending otherwise is the thing this replaced.

   The bandwidth is the dial between the two pictures: narrow and every value
   shows as its own spike, wide and it converges to the analytic bell. 0.85 of
   Silverman's rule keeps one clear peak while leaving the asymmetry visible.

   Smoothing adds width — the estimate carries variance s² + h², so the solid
   curve sits a little lower and wider than the dashed one even when the data is
   perfectly normal. The asymmetry is the data; that much of the height gap is
   the estimator, which is why the note points at the shape and not the peak.
   ------------------------------------------------------------------------- */
const DSTEP = 2;                                  // pixels between density samples
function kde(vals, sd, aw, ctr, hw){
  const half = aw/2 - 4, n = vals.length;
  const h = Math.max(0.85*sd*Math.pow(n, -0.2), hw/12);
  const out = [];
  for(let x=0; x<=aw; x+=DSTEP){
    const v = ctr + (x - aw/2)/half*hw;
    let d = 0;
    for(let i=0;i<n;i++) d += Math.exp(-0.5*((v - vals[i])/h)**2);
    out.push(d/(n*h*Math.sqrt(2*Math.PI)));
  }
  return out;
}
// drawn as a real density, so squeezing the spread makes the curve taller: the
// area under it is always 1
function curve(g, dens, ax, scale, fill, stroke){
  const maxH = AXIS_Y - BELL_TOP;
  g.beginPath();
  for(let i=0;i<dens.length;i++){
    const x = ax + i*DSTEP, y = AXIS_Y - Math.min(maxH, dens[i]*scale);
    i ? g.lineTo(x, y) : g.moveTo(x, y);
  }
  g.strokeStyle = stroke; g.lineWidth = 1.4; g.stroke();
  g.lineTo(ax + (dens.length - 1)*DSTEP, AXIS_Y); g.lineTo(ax, AXIS_Y); g.closePath();
  g.fillStyle = fill; g.fill();
}
// the normal with the same μ and σ — the summary the layer actually works from,
// dashed behind the data it is a summary of
function bell(g, m, sd, ax, aw, ctr, hw, scale, stroke){
  const half = aw/2 - 4, cx = ax + aw/2, maxH = AXIS_Y - BELL_TOP;
  g.beginPath();
  for(let x=ax, first=true; x<=ax+aw; x+=2, first=false){
    const v = ctr + (x - cx)/half*hw;
    const d = Math.exp(-0.5*((v-m)/sd)**2) / (sd*Math.sqrt(2*Math.PI));
    const y = AXIS_Y - Math.min(maxH, d*scale);
    first ? g.moveTo(x, y) : g.lineTo(x, y);
  }
  g.setLineDash([3,3]); g.strokeStyle = stroke; g.lineWidth = 1; g.stroke();
  g.setLineDash([]);
}

function paintFocus(P){
  const wpx = 3*COL_W + 2*ARROW_W;
  const g = fitCanvas(el("focus"), wpx, FOCUS_H);
  g.clearRect(0, 0, wpx, FOCUS_H);
  const f = S.focus;
  if(!f) return;

  const {n,c,h,w} = f, i = idx(n,c,h,w);
  const grp = GRP[GID[nc(n,c)]];
  const mu = MU[nc(n,c)], va = VA[nc(n,c)], sd = Math.sqrt(va + eps());
  const identity = GAM[c] === 1 && BET[c] === 0;

  // ONE scale shared by all three axes, so the collapse onto zero is visible
  // rather than normalised away
  let span = 1e-6;
  for(const T of [X, XH, Y]) for(const m of grp.members)
    for(let hh=0;hh<S.H;hh++) for(let ww=0;ww<S.Wd;ww++)
      span = Math.max(span, Math.abs(T[idx(m.n,m.c,hh,ww)]));

  const stages = [
    {T:X,  v:X[i],  head:"x — raw",        note:`x[${n}][${c}][${h}][${w}]`},
    {T:XH, v:XH[i], head:"x̂ — normalised", note:"(x − μ) / √(σ² + ε)"},
    {T:Y,  v:Y[i],  head:"y — after γ, β", note: identity ? "γ·x̂ + β  (identity)" : "γ·x̂ + β"},
  ];
  const st = stages.map(s => Object.assign(s, groupStats(s.T, grp)));

  // Each axis is centred on its own μ, and all three cover the same number of
  // values per pixel. That is what makes the squeeze visible: same units either
  // side, so a smaller σ is a narrower curve, and taller, since the area is 1.
  //
  // Centring every axis on zero instead would shove the raw curve into a corner
  // whenever μ is far from zero, while x̂ sat comfortably in the middle. The
  // shift is shown by where the zero mark lands, which is the honest place.
  const aw = COL_W - 16;
  let reach = 0;
  for(const s of st){
    s.vals = [];
    for(const m of grp.members) for(let hh=0;hh<S.H;hh++) for(let ww=0;ww<S.Wd;ww++)
      s.vals.push(s.T[idx(m.n,m.c,hh,ww)]);
    for(const v of s.vals) reach = Math.max(reach, Math.abs(v - s.mean));
  }
  // sized from the furthest value any stage reaches, so nothing is ever drawn
  // off the end of its axis
  const HW = Math.max(1.12*reach, span/40);                 // values either side of μ
  const floor = HW/60;                                      // keeps a spike of 0 width drawable

  // one vertical scale for all three columns, tall enough for whichever of the
  // six curves peaks highest — so nothing is silently flattened against the top
  let peak = 0;
  for(const s of st){
    s.dens = kde(s.vals, Math.max(s.sd, floor), aw, s.mean, HW);
    for(const d of s.dens) peak = Math.max(peak, d);
    peak = Math.max(peak, 1/(Math.max(s.sd, floor)*Math.sqrt(2*Math.PI)));
  }
  const scale = (AXIS_Y - BELL_TOP) / (peak || 1);

  st.forEach((s, k) => {
    const x0 = k*(COL_W + ARROW_W), mid = x0 + COL_W/2;
    const ax = mid - aw/2, px = v => ax + aw/2 + ((v - s.mean)/HW)*(aw/2 - 4);
    const last = k === 2;

    g.textAlign = "center"; g.textBaseline = "alphabetic";
    g.font = "600 10px ui-sans-serif,system-ui";
    g.fillStyle = last && identity ? P.dim2 : P.dim;
    g.fillText(s.head, mid, TITLE_Y);

    const tx = mid - TILE_W/2;                              // the value itself
    g.fillStyle = P.panel; roundRect(g, tx, TILE_Y, TILE_W, TILE_H, 7); g.fill();
    g.fillStyle = tint(s.v, span, P); roundRect(g, tx, TILE_Y, TILE_W, TILE_H, 7); g.fill();
    if(last && identity){ g.setLineDash([4,3]); g.strokeStyle = P.dim2; }
    else g.strokeStyle = P.frameLine;
    g.lineWidth = 1.2; g.stroke(); g.setLineDash([]);
    g.fillStyle = P.text; g.font = "700 17px ui-monospace,Menlo,monospace";
    g.textBaseline = "middle";
    g.fillText(fmt(s.v), mid, TILE_Y + TILE_H/2 + 1);

    g.font = "9.5px ui-monospace,Menlo,monospace"; g.fillStyle = P.dim2;
    g.textBaseline = "alphabetic";
    g.fillText(s.note, mid, NOTE_Y);

    // what the values do, the normal that summarises them, then the values
    // themselves on the axis beneath
    const rgb = k === 0 ? P.pos : k === 1 ? P.neg : P.vio;
    curve(g, s.dens, ax, scale,
          `rgba(${rgb[0]},${rgb[1]},${rgb[2]},.16)`, `rgba(${rgb[0]},${rgb[1]},${rgb[2]},.85)`);
    bell(g, s.mean, Math.max(s.sd, floor), ax, aw, s.mean, HW, scale, P.dim2);

    g.strokeStyle = P.line; g.lineWidth = 1;
    g.beginPath(); g.moveTo(ax, AXIS_Y); g.lineTo(ax + aw, AXIS_Y); g.stroke();

    // where zero falls. On the raw axis it is usually off the end — which is
    // precisely the distance − μ is about to close. On x̂ it lands dead centre.
    g.strokeStyle = P.dim2; g.fillStyle = P.dim2;
    const z = px(0);
    if(z >= ax && z <= ax + aw){
      g.beginPath(); g.moveTo(z, AXIS_Y - 4); g.lineTo(z, AXIS_Y + 4); g.stroke();
    } else {
      const dir = z < ax ? -1 : 1, edge = dir < 0 ? ax + 1 : ax + aw - 1;
      g.beginPath();
      g.moveTo(edge + dir*5, AXIS_Y); g.lineTo(edge - dir*2, AXIS_Y - 4);
      g.lineTo(edge - dir*2, AXIS_Y + 4); g.closePath(); g.fill();
    }
    g.fillStyle = `rgba(${rgb[0]},${rgb[1]},${rgb[2]},.65)`;
    for(const m of grp.members) for(let hh=0;hh<S.H;hh++) for(let ww=0;ww<S.Wd;ww++){
      g.beginPath(); g.arc(px(s.T[idx(m.n,m.c,hh,ww)]), AXIS_Y + 4, 1.9, 0, 6.2832); g.fill();
    }
    g.fillStyle = P.amber;                                  // this one, marked
    g.beginPath();
    g.moveTo(px(s.v), AXIS_Y - 3); g.lineTo(px(s.v) - 4, AXIS_Y - 10);
    g.lineTo(px(s.v) + 4, AXIS_Y - 10); g.closePath(); g.fill();

    g.font = "9.5px ui-monospace,Menlo,monospace"; g.fillStyle = P.dim;
    g.fillText(`μ ${fmt(s.mean)}   σ ${fmt(s.sd)}`, mid, STATS_Y);
  });

  // the two operations, written on the arrows
  const E = S.epsExp === 0 ? "1" : `1e${S.epsExp}`;
  const ops = [
    [`− μ  (${fmt(mu)})`, `÷ √(σ² + ε)  (${fmt(sd,3)})`, P.vioCss],
    [`× γ  (${fmt(GAM[c])})`, `+ β  (${fmt(BET[c])})`, P.accent],
  ];
  ops.forEach(([top, bot, col], k) => {
    const x0 = COL_W + k*(COL_W + ARROW_W), x1 = x0 + ARROW_W, y = TILE_Y + TILE_H/2;
    const dim = k === 1 && identity;
    g.strokeStyle = dim ? P.line : P.dim2; g.lineWidth = 1.4;
    g.beginPath(); g.moveTo(x0 + 12, y); g.lineTo(x1 - 16, y); g.stroke();
    g.beginPath(); g.moveTo(x1 - 16, y); g.lineTo(x1 - 23, y - 4.5);
    g.lineTo(x1 - 23, y + 4.5); g.closePath(); g.fillStyle = dim ? P.line : P.dim2; g.fill();
    g.textAlign = "center"; g.textBaseline = "alphabetic";
    g.font = "10px ui-monospace,Menlo,monospace";
    g.fillStyle = dim ? P.dim2 : col;
    g.fillText(top, (x0 + x1)/2, y - 11);
    g.fillText(bot, (x0 + x1)/2, y + 20);
  });

  el("focusnote").innerHTML =
    `<b style="color:var(--fg)">Reading it.</b> The solid curve is the shape
     <b>every number pooled into the same μ</b> actually makes — each value contributes a small bump and
     they are added up, so it comes out bell-<em>like</em> but lopsided, with a real shoulder and tail.
     The dashed line is the perfect normal with that same μ and σ: the <em>summary</em> the layer works
     from. Twenty-seven values never make a clean bell — one side runs out further, the peak sits off to
     one side — and everything μ and σ know about that is nothing.
     The dots are those values and the triangle is the one you are pointing at.
     The axes are centred on their own μ but cover the same
     <b>values per pixel</b>, so the middle curve being narrower <em>and</em> taller is real and not a
     change of zoom — squeezing the spread to 1 has to push the peak up, since the area under a density
     is always 1. The small mark on each axis is <b>zero</b>: off the end on the raw axis, which is the
     distance − μ closes, and dead centre once it has.
     <br><b style="color:var(--fg)">Why − μ and ÷ σ.</b> They put every channel on the same footing.
     Whatever this one happened to be doing — sitting at ${fmt(st[0].mean)}, ${fmt(st[0].sd)} wide — comes
     out at 0 and 1, like all the others, and stays there however the layers below it change. The next
     layer then gets an input it can count on instead of a moving one.
     ${identity
       ? `<br><b style="color:var(--fg)">Why γ and β.</b> Forcing 0 and 1 is a restriction: some channels
          want to be wider, or off-centre, and a sigmoid handed only 0-and-1 inputs never leaves its
          straight middle. γ and β buy the freedom back — <b>and only they are learned</b>, so the network
          chooses the spread and centre while the normalising keeps them steady.
          <b>Right now γ = 1 and β = 0, the identity</b>, so the third curve is the second one exactly.
          That is what <em>at init</em> means. Drag γ or β, or press <em>trained</em>, to separate them.`
       : `<br><b style="color:var(--fg)">Why γ and β.</b> Forcing 0 and 1 is a restriction: some channels
          want to be wider, or off-centre, and a sigmoid handed only 0-and-1 inputs never leaves its
          straight middle. γ and β buy the freedom back — <b>and only they are learned</b>, so the network
          chooses the spread and centre while the normalising keeps them steady. Here it has stretched
          the curve by <b>γ = ${fmt(GAM[c])}</b> and shifted it by <b>β = ${fmt(BET[c])}</b>.`}`;
}

/* ---------- the operation itself, spelled out once, always on screen ---------- */
function renderStepFormula(){
  const c = S.focus ? S.focus.c : 0;
  const identity = GAM[c] === 1 && BET[c] === 0;
  el("stepformula").innerHTML =
    `<span class="lbl">1 · normalise — no parameters</span>
     <b>x̂</b> = ( x − <span class="sv">μ</span> ) / √( <span class="sv">σ²</span> + ε )
     <span style="color:var(--dim2)">→ always mean 0, variance 1</span>
     <span class="lbl">2 · rescale — the learnable part</span>
     <b>y</b> = <span class="pv">γ</span> · x̂ + <span class="pv">β</span>
     <span style="color:var(--dim2)">→ one (<span class="pv">γ</span>, <span class="pv">β</span>) pair per channel</span>
     <span class="lbl">for channel ${c}</span>
     y = <span class="pv">${fmt(GAM[c])}</span> · x̂ ${BET[c] < 0 ? "−" : "+"} <span class="pv">${fmt(Math.abs(BET[c]))}</span>
     ${identity ? `<span class="flag ok" style="margin-left:6px">identity</span>` : ""}`;
}

/* --------- every value of every group on one axis, before and after --------- */
const DROWS = 12;
function paintDist(P){
  const shown = Math.min(GRP.length, DROWS);
  const avail = Math.max(320, (el("distwrap").clientWidth || 700) - 2);
  const gut = 62, gap = 22;
  const pw = Math.floor((avail - gut - gap)/2);
  const rh = 15, top = 20;
  const h = top + shown*rh + 10;
  const g = fitCanvas(el("dist"), avail, h);
  g.clearRect(0,0,avail,h);

  let mx = 0, my = 0;
  for(let i=0;i<X.length;i++){ mx = Math.max(mx, Math.abs(X[i])); my = Math.max(my, Math.abs(Y[i])); }
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

  el("distnote").innerHTML =
    `Each dot is one number of the tensor, one row per pooling group. Left: the raw values — the
     groups sit at different centres with different spreads. Right: the same numbers after the layer,
     every row lined up on one centre and one spread — which is the point.
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
  const i  = idx(n,c,h,w);
  const x  = X[i], xh = XH[i], y = Y[i];
  const sd = Math.sqrt(va + eps());
  const frozen = S.norm === "batch" && S.mode === "eval";
  const src = frozen ? "running_" : "";
  const E = S.epsExp === 0 ? "1" : `1e${S.epsExp}`;
  const flat = GAM[c] === 1 && BET[c] === 0;

  el("readout").innerHTML =
    mrow("this cell", `<b>x[${n}][${c}][${h}][${w}] = ${xv(x)}</b> &nbsp;→&nbsp; <b>y = ${fmt(y)}</b>`) +
    mrow("pooled with", frozen
      ? `<span style="color:var(--dim)">nothing — eval() ignores this batch. <span class="sv">running_μ</span> and
         <span class="sv">running_σ²</span> come from the ${S.steps} batch${S.steps===1?"":"es"} fed so far.</span>`
      : `<span style="color:var(--dim)">${groupWhy(g)}</span>`) +
    mrow("step 1", `x̂ = ( ${xv(x)} − <span class="sv">${src}μ</span> ${sv(mu)} ) / √( <span class="sv">${src}σ²</span> ${sv(va)} + ${E} )
                    &nbsp;=&nbsp; ${fmt(x-mu)} / ${fmt(sd, 3)} &nbsp;=&nbsp; <span class="res">${fmt(xh)}</span>`) +
    mrow("step 2", `y = ${pv(GAM[c])} · ${fmt(xh)} + ${pv(BET[c])} &nbsp;=&nbsp; <span class="res">${fmt(y)}</span>`
                   + (flat ? ` <span style="color:var(--dim2)">— γ = 1, β = 0, so this step does nothing</span>` : ""));
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
    if(P.id === "C"){ resetRunning(); buildAffine(true); }
    refresh(true);
  }));

  const seg = (id, attr, fn) => document.querySelectorAll(`#bn_${id} button`).forEach(b =>
    b.addEventListener("click", () => { fn(b.dataset[attr]); }));

  seg("normseg", "norm", v => { S.norm = v; fixGroups(); S.focus = null; refresh(); });
  seg("epsseg",  "eps",  v => { S.epsExp = +v; refresh(); });
  seg("affseg",  "aff",  v => { S.aff = v; buildAffine(true); refresh(); });
  el("rgam").addEventListener("input", e => setAffine("gam", +e.target.value));
  el("rbet").addEventListener("input", e => setAffine("bet", +e.target.value));
  seg("modeseg", "mode", v => { S.mode = v; refresh(); });
  seg("momseg",  "mom",  v => { S.mom = +v; refresh(); });

  el("csprev").addEventListener("click", () => { csStop(); csGo(S.cs - 1); });
  el("csnext").addEventListener("click", () => { csStop(); csGo(S.cs + 1); });
  el("csplay").addEventListener("click", csPlay);

  el("feed").addEventListener("click", feedBatch);
  el("reset").addEventListener("click", () => { resetRunning(); refresh(); });
  el("dead").addEventListener("change", e => { S.dead = e.target.checked; refresh(true); });

  document.querySelectorAll("#view-norm [data-preset]").forEach(b => b.addEventListener("click", () => {
    Object.assign(S, JSON.parse(b.dataset.preset));
    relink(); fixGroups(); S.focus = null; resetRunning(); buildAffine(true); refresh(true);
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
    fixGroups(); S.focus = null; resetRunning(); buildAffine(true); refresh(true);
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
    off:     `<code>affine=False</code> — there is no γ or β at all, so y <em>is</em> x̂.`,
    init:    `<code>affine=True</code>, at PyTorch's initialisation. γ=1 and β=0 are the <b>identity</b>,
              so the numbers are <b>exactly the same as affine=False</b> — nothing on screen will move.
              The difference is that they are now parameters with gradients. Press <em>trained</em>, or
              drag them below, to see where training takes them.`,
    learned: `Where γ and β tend to land after training — each channel gets its own pair.`,
    custom:  `Your own γ, β for this channel. Watch the third curve stretch by γ and slide by β.`,
  })[S.aff];

  const affOff = S.aff === "off";
  const fc = S.focus ? S.focus.c : 0;
  el("affctrl").style.opacity = affOff ? .4 : 1;
  el("rgam").disabled = el("rbet").disabled = affOff;
  el("gamlbl").textContent = `γ · channel ${fc}`;
  el("betlbl").textContent = `β · channel ${fc}`;
  el("rgam").value = GAM[fc]; el("vgam").textContent = fmt(GAM[fc]);
  el("rbet").value = BET[fc]; el("vbet").textContent = fmt(BET[fc]);
  el("afftable").innerHTML = affOff ? "" :
    `<b style="color:var(--fg)">every channel</b><br>` +
    Array.from({length:S.C}, (_,c) =>
      `<span class="mono" style="${c===fc?"color:var(--fg)":""}">c${c} &nbsp;γ ${fmt(GAM[c])} &nbsp;β ${fmt(BET[c])}</span>`
    ).join("<br>");

  const isBN = S.norm === "batch";
  el("runwrap").style.display = isBN ? "" : "none";
  document.querySelectorAll("#bn_modeseg button").forEach(b => b.disabled = !isBN && b.dataset.mode === "eval");
  el("runhint").innerHTML = !isBN ? "" : S.mode === "train"
    ? `Each batch nudges the estimates: <span class="mono">running = (1−m)·running + m·batch</span>.
       ${S.steps} fed so far.`
    : `<b>eval() freezes them.</b> Feeding batches no longer changes running_mean or running_var — that is the
       point, inference must not depend on whatever else is in the batch.`;
  el("dead").checked = S.dead;

  el("cslabel").textContent = `step ${S.cs} of ${CS_T - 1}`;
  el("csprev").disabled = S.cs === 0;
  el("csnext").disabled = S.cs === CS_T - 1;
  el("csplay").textContent = csTimer ? "❚❚ pause" : "▶ train";

  const c = S.calc;
  el("dN").value = calc("N"); el("dC").value = calc("C");
  el("dH").value = calc("H"); el("dW").value = calc("W");
  el("dsync").style.display = calcLinked() ? "none" : "";
}

/* ---------- walking the training steps ---------- */
let csTimer = 0;
function csGo(t){
  S.cs = Math.max(0, Math.min(CS_T - 1, t));
  paint(); syncControls();
}
function csStop(){
  clearInterval(csTimer); csTimer = 0;
  if(started) syncControls();               // the tab bar can stop this before wire() ever ran
}
function csPlay(){
  if(csTimer) return csStop();
  if(S.cs >= CS_T - 1) S.cs = 0;
  csTimer = setInterval(() => {
    if(S.cs >= CS_T - 1) return csStop();
    csGo(S.cs + 1);
  }, 420);
  syncControls();
}

/* ============================== interaction ============================== */
function hit(ev){
  const rect = el("viz").getBoundingClientRect();
  const x = ev.clientX - rect.left, y = ev.clientY - rect.top;
  const cs = L.cs, bw = S.Wd*cs, bh = S.H*cs;
  for(let n=0;n<S.N;n++) for(let c=0;c<S.C;c++){
    const b = blockXY(n, c);
    const ix = x - b.x, iy = y - b.y;
    if(ix >= 0 && iy >= 0 && ix < bw && iy < bh)
      return {t:"x", n, c, h: Math.floor(iy/cs), w: Math.floor(ix/cs)};
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
  },
  hide(){ csStop(); }
};
})();
