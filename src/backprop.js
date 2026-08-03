/* ==================================================================
   MODULE 3 — Backpropagation

   The page is one neuron drawn twice: a forward band along the top,
   and the backward band directly underneath it, station by station,
   so every gradient sits below the value it came from. Each station
   carries its own equation with this point's numbers substituted in,
   and is filled with the colour of the number it holds — so the whole
   pass can be read as a picture first and as arithmetic second.
   ================================================================== */
const BP = (function(){

/* =============================== engine ===============================
   A scalar autograd graph, micrograd style. Every number this page shows —
   every chip, every edge width, every colour — is read off one of these
   nodes, and nothing is recomputed in the drawing code. That is the only way
   the picture and the arithmetic cannot drift apart.
   ====================================================================== */
function NOOP(){}

function V(d, prev, op){
  return {d, g:0, prev: prev || [], op: op || "", local:0, back:NOOP};
}
function add(a, b){
  const o = V(a.d + b.d, [a,b], "+");
  o.back = () => { a.g += o.g; b.g += o.g; };
  return o;
}
function sub(a, b){
  const o = V(a.d - b.d, [a,b], "−");
  o.back = () => { a.g += o.g; b.g -= o.g; };
  return o;
}
function mul(a, b){
  const o = V(a.d * b.d, [a,b], "·");
  o.back = () => { a.g += b.d*o.g; b.g += a.d*o.g; };   // a === b is fine, both lines fire
  return o;
}
// the local derivative is stored, not re-derived later, so the chain-rule chip
// prints the exact number the backward pass used
function unary(z, val, der, op){
  const o = V(val, [z], op);
  o.local = der;
  o.back = () => { z.g += der*o.g; };
  return o;
}

function backward(root){
  const topo = [], seen = new Set();
  (function visit(n){
    if(seen.has(n)) return;
    seen.add(n);
    for(const p of n.prev) visit(p);
    topo.push(n);
  })(root);
  for(const n of topo) n.g = 0;
  root.g = 1;                                  // dL/dL
  for(let i = topo.length - 1; i >= 0; i--) topo[i].back();
}

/* ============================= activations ============================= */
const ACT = {
  linear: {
    lbl:"linear", call:"z", der:"1",
    raw: z => z,
    f: z => unary(z, z.d, 1, "id"),
    hint:`No nonlinearity at all — a bare <code>nn.Linear</code>. ∂a/∂z is 1, so the gradient reaches the
          weights undamped, and the fit is a straight line.`,
  },
  sigmoid: {
    lbl:"sigmoid", call:"σ(z)", der:"a·(1 − a)",
    raw: z => 1/(1 + Math.exp(-z)),
    f: z => { const s = 1/(1 + Math.exp(-z.d)); return unary(z, s, s*(1 - s), "σ"); },
    hint:`Squashes into (0, 1). ∂a/∂z = a(1−a) peaks at <b>0.25</b> and collapses towards zero at both
          ends — push z past ±6 and almost no gradient survives the trip back.`,
  },
  tanh: {
    lbl:"tanh", call:"tanh(z)", der:"1 − a²",
    raw: z => Math.tanh(z),
    f: z => { const t = Math.tanh(z.d); return unary(z, t, 1 - t*t, "tanh"); },
    hint:`Squashes into (−1, 1), centred on zero. ∂a/∂z = 1−a² peaks at <b>1</b>, so it saturates like
          sigmoid but passes four times as much gradient through the middle.`,
  },
  relu: {
    lbl:"relu", call:"relu(z)", der:"1 if z > 0, else 0",
    raw: z => Math.max(0, z),
    f: z => unary(z, Math.max(0, z.d), z.d > 0 ? 1 : 0, "relu"),
    hint:`Either passes the gradient through untouched or kills it outright. If z stays negative for every
          point, every gradient is exactly 0 and the neuron can never recover — a <b>dead ReLU</b>.`,
  },
};

/* ============================== the data ==============================
   A handful of points, fixed and reproducible. The neuron is fitted to all of
   them at once, which is what makes the curve on the right worth watching.
   ====================================================================== */
function prng(s){ let x = s>>>0; return () => (x = (x*1664525 + 1013904223)>>>0) / 4294967296; }
function gauss(r){ const u = Math.max(r(), 1e-9); return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*r()); }
const r2 = v => Math.round(v*100)/100;

const SETS = {
  line: {
    name:"a line to fit", note:"linear — least squares, the classic",
    nin:1, act:"linear", w:[-0.6], b:-0.9, lr:0.2,
    make(){
      const r = prng(20250801), out = [];
      for(let i=0;i<13;i++){
        const x = r2(-2.1 + i*0.35);
        out.push({x:[x, 0], t: r2(0.7*x + 0.3 + 0.22*gauss(r))});
      }
      return out;
    },
    why:`The neuron is <code>a = w·x + b</code> — a straight line. Gradient descent rotates and slides it
         until the residuals are as short as they can get. That is linear regression, and backprop is how
         it finds the answer.`,
  },
  step: {
    name:"a step to fit", note:"sigmoid — logistic regression",
    nin:1, act:"sigmoid", w:[0.4], b:0.1, lr:1.5,
    make(){
      const out = [];
      for(let i=0;i<14;i++){
        const x = r2(-2.1 + i*0.32);
        out.push({x:[x, 0], t: x > 0.35 ? 1 : 0});
      }
      return out;
    },
    why:`The targets are 0 on one side and 1 on the other. A sigmoid can only fit that by growing steep —
         watch <b>w</b> climb, which is exactly the curve sharpening. It never gets all the way there,
         because a steeper sigmoid means a smaller ∂a/∂z and so a smaller gradient.`,
  },
  plane: {
    name:"two clusters", note:"sigmoid on a plane — a boundary",
    nin:2, act:"sigmoid", w:[0.3, -0.4], b:0, lr:1.0,
    make(){
      const r = prng(77003), out = [];
      for(let i=0;i<9;i++)  out.push({x:[r2(-1.1 + 0.5*gauss(r)), r2(-0.7 + 0.5*gauss(r))], t:0});
      for(let i=0;i<9;i++)  out.push({x:[r2( 1.1 + 0.5*gauss(r)), r2( 0.8 + 0.5*gauss(r))], t:1});
      return out;
    },
    why:`Two inputs, so the neuron is a plane pushed through a sigmoid. The solid line is where it says
         exactly 0.5 — its decision boundary. Training slides and rotates that line until the two clusters
         land on opposite sides of it.`,
  },
};

/* ================================ state ================================ */
const PHASES = [
  {id:"z", lbl:"weighted sum", eq:() => "z = Σ wᵢ·xᵢ + b"},
  {id:"a", lbl:"activation",   eq:() => "a = " + ACT[S.act].call},
  {id:"L", lbl:"loss",         eq:() => "L = (a − t)²"},
  {id:"g", lbl:"backward",     eq:() => "∂L/∂w = ∂L/∂a · ∂a/∂z · x"},
  {id:"u", lbl:"update",       eq:() => "w ← w − η · ∂L/∂w"},
];

const S = {
  set:"line", nin:1, data:[],
  w:[-0.6, 0], b:-0.9,
  w0:[-0.6, 0], b0:-0.9,          // what "reset" goes back to
  act:"linear", lr:0.2,
  sample: 6,                      // which point fills the stations
  phase: 0,
  sel: {k:"w", i:0},              // which parameter the chain rule is written for
  hover: null,                    // which station is under the cursor
  chip: "",                       // which chain-rule factor is hovered in the ledger
  drag: null,                     // a weight being dragged on its own wire
  trails: true,
  steps: 0,
  hist: [],                       // batch loss, one per update
  track: [],                      // (w, b) at each update — where every prediction has been
  playing: false,
};

const SUB = ["₁","₂","₃","₄"];
const el = id => document.getElementById("bp_" + id);

/* ============================== the maths ============================== */
let GS = [];                      // one graph per data point
let G = null;                     // the selected one — what every station reads
let GRAD = {w:[0,0], b:0};        // the batch gradient: the mean of the per-point ones
let LOSS = 0;                     // the batch loss: the mean of the per-point ones

function graph(d){
  const x = [], w = [];
  for(let i=0;i<S.nin;i++){ x.push(V(d.x[i])); w.push(V(S.w[i])); }
  const b = V(S.b);
  const prod = x.map((xi, i) => mul(w[i], xi));
  let z = prod[0];
  for(let i=1;i<S.nin;i++) z = add(z, prod[i]);
  z = add(z, b);
  const a = ACT[S.act].f(z);
  const t = V(d.t);
  const e = sub(a, t);
  const L = mul(e, e);            // this point's squared error
  backward(L);
  return {x, w, b, prod, z, a, t, e, L, d};
}

function compute(){
  GS = S.data.map(graph);
  const n = GS.length || 1;
  LOSS = GS.reduce((s, g) => s + g.L.d, 0)/n;
  // L = mean of the per-point losses, so dL/dw is the mean of the per-point gradients
  GRAD = {w: Array.from({length:S.nin}, (_, i) => GS.reduce((s, g) => s + g.w[i].g, 0)/n),
          b: GS.reduce((s, g) => s + g.b.g, 0)/n};
  S.sample = Math.min(S.sample, GS.length - 1);
  G = GS[S.sample];
}

// the same forward pass with no graph at all — only for differencing
function lossAt(w, b){
  let s = 0;
  for(const d of S.data){
    let z = b;
    for(let i=0;i<S.nin;i++) z += w[i]*d.x[i];
    const e = ACT[S.act].raw(z) - d.t;
    s += e*e;
  }
  return s/(S.data.length || 1);
}

function gradCheck(){
  const h = 1e-4, rows = [];
  for(let i=0;i<S.nin;i++){
    const up = S.w.slice(); up[i] += h;
    const dn = S.w.slice(); dn[i] -= h;
    rows.push({name:`∂L/∂w${SUB[i]}`, engine:GRAD.w[i], num:(lossAt(up, S.b) - lossAt(dn, S.b))/(2*h)});
  }
  rows.push({name:"∂L/∂b", engine:GRAD.b, num:(lossAt(S.w, S.b + h) - lossAt(S.w, S.b - h))/(2*h)});
  let worst = 0;
  for(const r of rows) worst = Math.max(worst, Math.abs(r.engine - r.num)/Math.max(1, Math.abs(r.num)));
  return {rows, worst};
}

const diverged = () => !isFinite(LOSS) || LOSS > 1e8;

function applyUpdate(){
  if(diverged()) return;
  S.hist.push(LOSS);                          // the loss this step was taken at
  S.track.push({w:S.w.slice(), b:S.b});
  if(S.hist.length > 600){ S.hist.shift(); S.track.shift(); }
  for(let i=0;i<S.nin;i++) S.w[i] -= S.lr*GRAD.w[i];
  S.b -= S.lr*GRAD.b;
  S.steps++;
  compute();
}

function train(n){
  for(let k=0;k<n && !diverged();k++) applyUpdate();
  S.phase = Math.max(S.phase, 2);
  refresh();
}

function loadSet(key){
  const D = SETS[key];
  S.set = key; S.nin = D.nin; S.act = D.act; S.lr = D.lr;
  S.data = D.make();
  S.w = [D.w[0], D.w[1] || 0]; S.b = D.b;
  S.w0 = S.w.slice(); S.b0 = S.b;
  // open on the point furthest from the origin: at x = 0 every product is 0 and
  // the whole diagram fades to nothing, which is a poor first impression
  S.sample = S.data.reduce((best, d, i, a) =>
    Math.hypot(...d.x.slice(0, S.nin)) > Math.hypot(...a[best].x.slice(0, S.nin)) ? i : best, 0);
  S.sel = {k:"w", i:0};
  S.steps = 0; S.hist = []; S.track = []; S.phase = 0;
}

function reset(){
  S.w = S.w0.slice(); S.b = S.b0;
  S.steps = 0; S.hist = []; S.track = []; S.phase = 0;
  refresh();
}

// PyTorch initialises nn.Linear uniformly on ±1/√fan_in
function randomise(){
  const k = 1/Math.sqrt(S.nin);
  S.w = S.w.map(() => r2((Math.random()*2 - 1)*k));
  S.b = r2((Math.random()*2 - 1)*k);
  S.w0 = S.w.slice(); S.b0 = S.b;
  S.steps = 0; S.hist = []; S.track = []; S.phase = 0;
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
    pos:   d ? [240,161,50]  : [188,108,0],     // positive
    neg:   d ? [74,163,255]  : [9,105,218],     // negative
    grad:  d ? [210,168,255] : [130,80,223],    // the backward pass
    gradCss: d ? "#d2a8ff" : "#8250df",
    accent:d ? "#4aa3ff" : "#0969da",
    amber: d ? "#f0a132" : "#bc6c00",
  };
}

/* ---------- one colour ramp, used for every number on the page ----------
   Sign picks the hue, magnitude picks the opacity. Near zero fades to nothing,
   so a glance over the page reads which numbers are doing the work — and, once
   the fit converges, which have stopped.
   ---------------------------------------------------------------------- */
function ramp(v, max, P, lo, hi){
  const t = Math.min(1, Math.abs(v)/(max || 1));
  const rgb = v >= 0 ? P.pos : P.neg;
  const a = (lo === undefined ? 0.05 : lo);
  return `rgba(${rgb[0]},${rgb[1]},${rgb[2]},${a + ((hi === undefined ? 0.82 : hi) - a)*t})`;
}
function gramp(v, P, lo, hi){                  // the same, in the gradient hue
  const t = Math.min(1, Math.abs(v)/GMAX);
  const [r, g, b] = P.grad;
  const a = (lo === undefined ? 0.05 : lo);
  return `rgba(${r},${g},${b},${a + ((hi === undefined ? 0.82 : hi) - a)*t})`;
}

// the reference each quantity is measured against. Fixed rather than
// auto-scaled: when the gradients vanish they have to *look* vanished.
const GMAX = 0.5;
let XMAX = 1, ZMAX = 1;
function scales(){
  XMAX = 1e-6; ZMAX = 1e-6;
  for(const d of S.data) for(let i=0;i<S.nin;i++) XMAX = Math.max(XMAX, Math.abs(d.x[i]));
  for(const g of GS) ZMAX = Math.max(ZMAX, Math.abs(g.z.d));
}
const WMAX = 2.5;

const fmt = (v, d=2) => {
  if(!isFinite(v)) return "∞";
  if(Math.abs(v) < 5e-3 && d <= 2) v = 0;
  const s = Math.abs(v - Math.round(v)) < 1e-9 ? String(Math.round(v)) : v.toFixed(d);
  return s === "-0" ? "0" : s;
};
// a negative operand gets brackets, so a product never reads as "0.20·-4.34"
const par = s => s.startsWith("-") ? `(${s})` : s;
// gradients span many orders of magnitude once an activation saturates
const fmtg = v => {
  if(!isFinite(v)) return "∞";
  const a = Math.abs(v);
  if(a === 0) return "0";
  if(a < 1e-3) return v.toExponential(1);
  return v.toFixed(a < 1 ? 3 : 2);
};

function fitCanvas(cv, w, h){
  const dpr = devicePixelRatio || 1;
  const W = Math.round(w*dpr), H = Math.round(h*dpr);
  if(cv.width !== W || cv.height !== H){       // only resize when it changed — this repaints per frame
    cv.width = W; cv.height = H;
    cv.style.width = w + "px"; cv.style.height = h + "px";
  }
  const g = cv.getContext("2d");
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  return g;
}
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
function seg(g, x0, y0, x1, y1){ g.beginPath(); g.moveTo(x0, y0); g.lineTo(x1, y1); g.stroke(); }

/* ---------- the two bands ----------
   Forward on top, backward directly underneath, three stations wide. Card
   widths are fixed at the narrow end and share whatever extra room the column
   gives them, so the sparklines grow with the window instead of the gaps.
   ---------------------------------------------------------------------- */
const IN_X = 62, IN_R = 17;
const FY = 58, FH = 128;                 // forward band
const BY = 218;                          // backward band (height depends on nin)
const UH = 38;                           // the update row
const CW = {sum:210, act:166, loss:166};
const ROWH = 21;

let LY = null;                           // the layout the last paint used — hit tests read it

function layout(){
  const avail = Math.max(832, Math.min(1120, (el("vizwrap").clientWidth || 900) - 2));
  const lead = 124, gap = 40;                    // the wire run, then the gaps between cards
  const sx = IN_X + IN_R + lead;
  const extra = Math.max(0, avail - 12 - sx - 2*gap - (CW.sum + CW.act + CW.loss));
  const sum = CW.sum + extra*0.42, act = CW.act + extra*0.29, loss = CW.loss + extra*0.29;
  const ax = sx + sum + gap, lx = ax + act + gap;
  const grows = S.nin === 1 ? 3 : 4;                     // ∂L/∂w… , ∂L/∂b , ∂L/∂x
  const bh = 24 + grows*ROWH + 8;
  const uy = BY + bh + 16;
  return {w:avail, h: uy + UH + 10, gap, sum, act, loss, sx, ax, lx, right: lx + loss, bh, uy};
}

// the sum card writes one row per term, and every wire lands on its own row
const rowsTop = () => FY + 30 + (S.nin === 1 ? 10 : 0);
const rowY  = i => rowsTop() + i*ROWH + ROWH/2;
const ruleY = () => rowsTop() + (S.nin + 1)*ROWH + 5;
const inY   = i => S.nin === 1 ? FY + 32 : FY + 22 + i*46;
const biasY = () => S.nin === 1 ? FY + 98 : FY + 112;

function segDist(px, py, s){
  const dx = s.x1 - s.x0, dy = s.y1 - s.y0;
  const l2 = dx*dx + dy*dy || 1;
  const t = Math.max(0, Math.min(1, ((px - s.x0)*dx + (py - s.y0)*dy)/l2));
  return Math.hypot(px - (s.x0 + t*dx), py - (s.y0 + t*dy));
}
const wireSeg = i => ({x0: IN_X + IN_R, y0: inY(i),   x1: LY.sx, y1: rowY(i)});
const biasSeg = () => ({x0: IN_X + IN_R, y0: biasY(), x1: LY.sx, y1: rowY(S.nin)});

/* ---------- what stays lit ----------
   A path is a connected chain: one input, its wire, its term, its gradient —
   plus the spine every path shares. Everything else fades out.
   ---------------------------------------------------------------------- */
const SPINE = ["sum", "act", "loss", "gp", "gz", "ga", "upd"];
function wPath(i){ return new Set(["x" + i, "e" + i, "r" + i, "gw" + i, "gx", ...SPINE]); }
function bPath(){ return new Set(["b", "eb", "r" + S.nin, "gb", ...SPINE]); }

function chipParts(){
  if(!S.chip) return null;
  if(S.chip === "dLda") return new Set(["loss", "ga"]);
  if(S.chip === "dadz") return new Set(["act", "gz"]);
  if(S.chip === "dzdp") return S.sel.k === "b" ? bPath() : wPath(S.sel.i);
  const m = /^t(\d+|b)$/.exec(S.chip);
  if(m) return m[1] === "b" ? bPath() : wPath(+m[1]);
  return null;
}
function liveSet(){
  const c = chipParts();
  if(c) return c;
  const h = S.hover;
  if(!h) return null;
  // the bias occupies row S.nin, which for one input is "r1" — so it has to be
  // tested before the numbered-path regex, which would otherwise claim it
  if(h === "b" || h === "eb" || h === "gb" || h === "r" + S.nin) return bPath();
  const m = /^(?:x|e|r|gw)(\d)$/.exec(h);
  if(m) return wPath(+m[1]);
  if(h === "act" || h === "gz")  return new Set(["act", "gz"]);
  if(h === "loss" || h === "ga") return new Set(["loss", "ga"]);
  return null;
}

/* ============================ the two bands ============================ */
function paint(){
  const P = palette();
  LY = layout();
  const g = fitCanvas(el("viz"), LY.w, LY.h);
  g.clearRect(0, 0, LY.w, LY.h);

  const A = ACT[S.act];
  const live = liveSet();
  const al = (...parts) => (live && !parts.some(p => live.has(p))) ? 0.15 : 1;
  const p = S.phase, ph = (frame % 120)/120;
  const showG = p >= 3 && !diverged();

  g.lineJoin = "round";
  g.textBaseline = "middle";

  bandLabel(g, P, "forward — this point's numbers", 8, FY - 30);

  /* ---- the wires, one per term ---- */
  for(let i=0;i<S.nin;i++)
    drawWire(g, P, wireSeg(i), S.w[i], `w${SUB[i]}`, false, al("e" + i), S.hover === "e" + i);
  // no label on the bias wire — the node it leaves already carries the number
  drawWire(g, P, biasSeg(), S.b, "", true, al("eb"), S.hover === "eb");

  /* ---- the inputs ---- */
  for(let i=0;i<S.nin;i++)
    node(g, P, IN_X, inY(i), IN_R, `x${SUB[i]}`, fmt(G.d.x[i]),
         ramp(G.d.x[i], XMAX, P), al("x" + i));
  node(g, P, IN_X, biasY(), IN_R, "b", fmt(S.b), ramp(S.b, WMAX, P), al("b"), true);

  /* ---- station 1: the weighted sum, written out ---- */
  card(g, P, LY.sx, FY, LY.sum, FH, al("sum"));
  g.globalAlpha = al("sum");
  label(g, P, LY.sx + 10, FY + 14, "1 · z = Σ wᵢ·xᵢ + b");
  g.globalAlpha = 1;
  for(let i=0;i<S.nin;i++)
    termRow(g, P, LY.sx, LY.sum, rowY(i), `w${SUB[i]}·x${SUB[i]}`,
            `${fmt(S.w[i])} · ${fmt(G.d.x[i])}`, G.prod[i].d, al("r" + i));
  termRow(g, P, LY.sx, LY.sum, rowY(S.nin), "b", "", S.b, al("r" + S.nin));
  g.globalAlpha = al("sum");
  g.strokeStyle = P.frameLine; g.lineWidth = 1;
  seg(g, LY.sx + 10, ruleY(), LY.sx + LY.sum - 10, ruleY());
  g.fillStyle = ramp(G.z.d, ZMAX, P, 0.08, 0.55);
  roundRect(g, LY.sx + 8, ruleY() + 5, LY.sum - 16, 22, 6); g.fill();
  g.textAlign = "left"; g.font = "700 14px ui-monospace,Menlo,monospace"; g.fillStyle = P.text;
  g.fillText("z = " + fmt(G.z.d), LY.sx + 15, ruleY() + 16);
  g.globalAlpha = 1;

  arrow(g, P, LY.sx + LY.sum, FY + FH/2, LY.ax, FY + FH/2, al("sum", "act"), "z");

  /* ---- station 2: the activation, and its slope drawn as a tangent ---- */
  card(g, P, LY.ax, FY, LY.act, FH, al("act"), p >= 1 ? ramp(G.a.d, 1.2, P, 0.02, 0.18) : null);
  g.globalAlpha = al("act");
  label(g, P, LY.ax + 10, FY + 14, `2 · a = ${A.call}`);
  if(p >= 1){
    sparkAct(g, P, LY.ax + 12, FY + 24, LY.act - 24, 62);
    g.textAlign = "left"; g.font = "700 14px ui-monospace,Menlo,monospace"; g.fillStyle = P.text;
    g.fillText("a = " + fmt(G.a.d), LY.ax + 12, FY + 102);
    g.font = "10.5px ui-monospace,Menlo,monospace"; g.fillStyle = P.gradCss;
    g.fillText(`slope ∂a/∂z = ${A.der} = ${fmtg(G.a.local)}`, LY.ax + 12, FY + 119);
  } else waiting(g, P, LY.ax, FY, LY.act, FH, "a = ?");
  g.globalAlpha = 1;

  arrow(g, P, LY.ax + LY.act, FY + FH/2, LY.lx, FY + FH/2, al("act", "loss"), "a");

  /* ---- station 3: the loss, as a parabola with this point on it ---- */
  const tcx = LY.lx + LY.loss/2;
  g.globalAlpha = al("loss");
  g.fillStyle = ramp(G.d.t, 1.2, P, 0.10, 0.6);
  roundRect(g, tcx - 46, 14, 92, 24, 7); g.fill();
  g.setLineDash([4,3]); g.strokeStyle = P.frameLine; g.lineWidth = 1; g.stroke();
  seg(g, tcx, 38, tcx, FY); g.setLineDash([]);
  g.textAlign = "center"; g.font = "600 11px ui-monospace,Menlo,monospace"; g.fillStyle = P.text;
  g.fillText("target " + fmt(G.d.t), tcx, 26);
  g.globalAlpha = 1;

  card(g, P, LY.lx, FY, LY.loss, FH, al("loss"), p >= 2 ? ramp(G.L.d, 1, P, 0.02, 0.2) : null);
  g.globalAlpha = al("loss");
  label(g, P, LY.lx + 10, FY + 14, "3 · L = (a − t)²");
  if(p >= 2){
    sparkLoss(g, P, LY.lx + 12, FY + 24, LY.loss - 24, 62);
    g.textAlign = "left"; g.font = "700 14px ui-monospace,Menlo,monospace"; g.fillStyle = P.text;
    g.fillText("L = " + fmt(G.L.d, 4), LY.lx + 12, FY + 102);
    g.font = "10.5px ui-monospace,Menlo,monospace"; g.fillStyle = P.dim;
    g.fillText(`mean over all ${S.data.length} = ${fmt(LOSS, 4)}`, LY.lx + 12, FY + 119);
  } else waiting(g, P, LY.lx, FY, LY.loss, FH, "L = ?");
  g.globalAlpha = 1;

  /* ---- the backward band, station for station under the forward one ---- */
  bandLabel(g, P, showG ? "backward — the same wires, the other way"
                        : "backward — not run yet", 8, BY - 14);
  const boxes = backBoxes();
  if(!showG){
    ghost(g, P, boxes.p.x, BY, boxes.p.w, LY.bh, "∂L/∂w, ∂L/∂b");
    ghost(g, P, boxes.z.x, BY, boxes.z.w, LY.bh, "∂L/∂z");
    ghost(g, P, boxes.a.x, BY, boxes.a.w, LY.bh, "∂L/∂a");
  }
  if(showG){
    // it starts at the loss: dL/dL is 1 by definition
    downArrow(g, P, tcx, FY + FH, BY, al("loss", "ga"), "∂L/∂L = 1", ph);

    gradBox(g, P, boxes.a, "∂L/∂a", "2(a − t)",
            `2( ${fmt(G.a.d)} − ${par(fmt(G.d.t))} )`, G.a.g, al("ga"));
    gradBox(g, P, boxes.z, "∂L/∂z", "∂L/∂a · ∂a/∂z",
            `${fmtg(G.a.g)} · ${par(fmtg(G.a.local))}`, G.z.g, al("gz"));
    backArrow(g, P, boxes.a.x, boxes.z.x + boxes.z.w, BY + LY.bh/2, al("ga", "gz"), ph);
    backArrow(g, P, boxes.z.x, boxes.p.x + boxes.p.w, BY + LY.bh/2, al("gz", "gp"), ph);

    paramBox(g, P, boxes.p, al);

    // what the backward pass reuses from the forward one — drawn only for the
    // path being pointed at, because that is the whole reason activations are cached
    if(live){
      g.setLineDash([3,4]); g.strokeStyle = P.gradCss; g.lineWidth = 1; g.globalAlpha = 0.6;
      const drop = x => seg(g, x, FY + FH, x, BY);
      // out of the node's left side, down the margin, into its own gradient row —
      // going straight down would run through whichever node sits underneath
      const elbow = (y0, y1) => {
        g.beginPath();
        g.moveTo(IN_X - IN_R, y0); g.lineTo(20, y0); g.lineTo(20, y1); g.lineTo(30, y1);
        g.stroke();
      };
      const rowMid = i => BY + 26 + i*ROWH + ROWH/2;
      if(live.has("gz") && live.size === 2) drop(LY.ax + LY.act/2);
      if(live.has("ga") && live.size === 2) drop(LY.lx + 26);
      for(let i=0;i<S.nin;i++) if(live.has("gw" + i)) elbow(inY(i), rowMid(i));
      if(live.has("gb")) elbow(biasY(), rowMid(S.nin));
      g.setLineDash([]); g.globalAlpha = 1;
    }
  }

  /* ---- the update, using the mean over every point ---- */
  if(p >= 4 && !diverged()) updateRow(g, P, LY.uy, al("upd"));
  else if(p >= 4) ghost(g, P, 8, LY.uy, LY.right - 8, UH, "diverged — nothing to apply");
  else            ghost(g, P, 8, LY.uy, LY.right - 8, UH, "w ← w − η · ∂L/∂w");
}

/* ---------- pieces ---------- */
function bandLabel(g, P, txt, x, y){
  g.textAlign = "left"; g.textBaseline = "middle";
  g.font = "600 10px ui-sans-serif,system-ui"; g.fillStyle = P.dim2;
  g.fillText(txt.toUpperCase(), x, y);
}
function label(g, P, x, y, txt){
  g.textAlign = "left"; g.font = "9.5px ui-monospace,Menlo,monospace"; g.fillStyle = P.dim;
  g.fillText(txt, x, y);
}
function card(g, P, x, y, w, h, alpha, fill){
  g.globalAlpha = alpha;
  g.fillStyle = P.frame; roundRect(g, x, y, w, h, 10); g.fill();
  if(fill){ g.fillStyle = fill; roundRect(g, x, y, w, h, 10); g.fill(); }
  g.strokeStyle = P.frameLine; g.lineWidth = 1.2; roundRect(g, x, y, w, h, 10); g.stroke();
  g.globalAlpha = 1;
}
// a station the pass has not reached yet: outlined, named, holding its place so
// nothing on the canvas moves when it fills in
function ghost(g, P, x, y, w, h, txt){
  g.setLineDash([4,4]); g.strokeStyle = P.line; g.lineWidth = 1;
  roundRect(g, x, y, w, h, 10); g.stroke(); g.setLineDash([]);
  g.textAlign = "center"; g.font = "10.5px ui-monospace,Menlo,monospace"; g.fillStyle = P.dim2;
  g.fillText(txt, x + w/2, y + h/2);
}
function waiting(g, P, x, y, w, h, txt){
  g.textAlign = "center"; g.font = "700 15px ui-monospace,Menlo,monospace"; g.fillStyle = P.dim2;
  g.fillText(txt, x + w/2, y + h/2 + 6);
}

// one line of the weighted sum, tinted by the product it contributes
function termRow(g, P, x, w, y, lhs, mid, val, alpha){
  g.globalAlpha = alpha;
  g.fillStyle = ramp(val, ZMAX, P, 0.04, 0.34);
  roundRect(g, x + 8, y - ROWH/2 + 1, w - 16, ROWH - 3, 5); g.fill();
  g.textAlign = "left"; g.font = "600 10.5px ui-monospace,Menlo,monospace"; g.fillStyle = P.dim;
  g.fillText(lhs, x + 14, y);
  if(mid){
    g.font = "10.5px ui-monospace,Menlo,monospace"; g.fillStyle = P.dim2;
    g.fillText(mid, x + 58, y);
  }
  g.textAlign = "right"; g.font = "700 11.5px ui-monospace,Menlo,monospace"; g.fillStyle = P.text;
  g.fillText(fmt(val), x + w - 14, y);
  g.globalAlpha = 1;
}

function node(g, P, x, y, r, name, val, fill, alpha, draggable){
  g.globalAlpha = alpha;
  g.fillStyle = P.frame;
  g.beginPath(); g.arc(x, y, r, 0, 6.2832); g.fill();
  g.fillStyle = fill;
  g.beginPath(); g.arc(x, y, r, 0, 6.2832); g.fill();
  g.strokeStyle = draggable && S.hover === "b" ? P.accent : P.frameLine;
  g.lineWidth = draggable && S.hover === "b" ? 2 : 1.2;
  g.stroke();
  g.textAlign = "center";
  g.font = "700 12px ui-monospace,Menlo,monospace"; g.fillStyle = P.text;
  g.fillText(val, x, y);
  // the name goes beside the circle, not above it — with two inputs and a bias
  // stacked in one column there is no room overhead
  g.textAlign = "right";
  g.font = "600 11px ui-monospace,Menlo,monospace"; g.fillStyle = P.dim;
  g.fillText(name, x - r - 7, y);
  g.globalAlpha = 1;
}

// thickness is |w|, hue is its sign — and it is the handle you drag to change it
function drawWire(g, P, s, w, name, dashed, alpha, hot){
  const t = Math.min(1, Math.abs(w)/WMAX);
  const rgb = w >= 0 ? P.pos : P.neg;
  g.globalAlpha = alpha;
  if(hot){
    g.strokeStyle = P.accent; g.lineWidth = 3 + 4.4*t;
    if(dashed) g.setLineDash([6,4]);
    seg(g, s.x0, s.y0, s.x1, s.y1); g.setLineDash([]);
  }
  g.strokeStyle = `rgba(${rgb[0]},${rgb[1]},${rgb[2]},${0.30 + 0.66*t})`;
  g.lineWidth = 1 + 4.4*t;
  if(dashed) g.setLineDash([6,4]);
  seg(g, s.x0, s.y0, s.x1, s.y1);
  g.setLineDash([]);

  const mx = s.x0 + (s.x1 - s.x0)*0.5, my = s.y0 + (s.y1 - s.y0)*0.5;
  g.textAlign = "center";
  if(name){
    g.font = "600 11px ui-monospace,Menlo,monospace";
    g.fillStyle = `rgb(${rgb[0]},${rgb[1]},${rgb[2]})`;
    g.fillText(`${name} = ${fmt(w)}`, mx, my - 11);
  }
  if(hot){
    g.font = "9px ui-sans-serif,system-ui"; g.fillStyle = P.dim2;
    g.fillText("drag ↕", mx, my + 12);
  }
  g.globalAlpha = 1;
}

function arrow(g, P, x0, y, x1, y1, alpha, name){
  g.globalAlpha = alpha;
  g.strokeStyle = P.dim2; g.lineWidth = 1.4;
  seg(g, x0, y, x1 - 7, y1);
  g.fillStyle = P.dim2;
  g.beginPath(); g.moveTo(x1, y1); g.lineTo(x1 - 8, y1 - 4.5); g.lineTo(x1 - 8, y1 + 4.5);
  g.closePath(); g.fill();
  if(name){
    g.textAlign = "center"; g.font = "600 10px ui-monospace,Menlo,monospace"; g.fillStyle = P.dim;
    g.fillText(name, (x0 + x1)/2, y - 11);
  }
  g.globalAlpha = 1;
}

/* the backward arrows: violet, pointing left, with a pulse travelling the way
   the gradient does — the one animated thing on the page */
function backArrow(g, P, x0, x1, y, alpha, ph){
  g.globalAlpha = alpha;
  g.strokeStyle = P.gradCss; g.lineWidth = 1.6;
  seg(g, x0, y, x1 + 8, y);
  g.fillStyle = P.gradCss;
  g.beginPath(); g.moveTo(x1, y); g.lineTo(x1 + 9, y - 4.6); g.lineTo(x1 + 9, y + 4.6);
  g.closePath(); g.fill();
  pulse(g, P, x0 + (x1 - x0)*ph, y, ph);
  g.globalAlpha = 1;
}
function downArrow(g, P, x, y0, y1, alpha, name, ph){
  g.globalAlpha = alpha;
  g.strokeStyle = P.gradCss; g.lineWidth = 1.6;
  seg(g, x, y0, x, y1 - 8);
  g.fillStyle = P.gradCss;
  g.beginPath(); g.moveTo(x, y1); g.lineTo(x - 4.6, y1 - 9); g.lineTo(x + 4.6, y1 - 9);
  g.closePath(); g.fill();
  pulse(g, P, x, y0 + (y1 - y0)*ph, ph);
  g.textAlign = "left"; g.font = "10px ui-monospace,Menlo,monospace"; g.fillStyle = P.gradCss;
  g.fillText(name, x + 8, (y0 + y1)/2);
  g.globalAlpha = 1;
}
function pulse(g, P, x, y, ph){
  const [r, gg, b] = P.grad;
  g.fillStyle = `rgba(${r},${gg},${b},${0.9*(1 - Math.abs(ph - 0.5)*1.4)})`;
  g.beginPath(); g.arc(x, y, 3.4, 0, 6.2832); g.fill();
}

function backBoxes(){
  return {
    p: {x: 8,     w: LY.sx + LY.sum - 8},
    z: {x: LY.ax, w: LY.act},
    a: {x: LY.lx, w: LY.loss},
  };
}

// one backward station: the rule on top, this point's numbers under it
function gradBox(g, P, box, name, rule, numbers, val, alpha){
  card(g, P, box.x, BY, box.w, LY.bh, alpha, gramp(val, P, 0.02, 0.22));
  g.globalAlpha = alpha;
  g.textAlign = "left"; g.font = "600 11px ui-monospace,Menlo,monospace"; g.fillStyle = P.gradCss;
  g.fillText(name + " = " + rule, box.x + 12, BY + 19);
  g.font = "11px ui-monospace,Menlo,monospace"; g.fillStyle = P.dim;
  g.fillText(numbers, box.x + 12, BY + LY.bh/2 + 2);
  g.font = "700 17px ui-monospace,Menlo,monospace"; g.fillStyle = P.text;
  g.fillText("= " + fmtg(val), box.x + 12, BY + LY.bh - 20);
  g.globalAlpha = 1;
}

// the station where the gradient finally reaches the parameters
function paramBox(g, P, box, al){
  card(g, P, box.x, BY, box.w, LY.bh, al("gp"));
  g.globalAlpha = al("gp");
  g.textAlign = "left"; g.font = "600 11px ui-monospace,Menlo,monospace"; g.fillStyle = P.gradCss;
  g.fillText("∂L/∂z spreads onto every parameter", box.x + 12, BY + 16);
  g.globalAlpha = 1;

  const row = (i, lhs, rhs, val, part, faint) => {
    const y = BY + 26 + i*ROWH + ROWH/2;
    g.globalAlpha = al(part) * (faint ? 0.7 : 1);
    g.fillStyle = gramp(val, P, 0.03, 0.34);
    roundRect(g, box.x + 8, y - ROWH/2 + 1, box.w - 16, ROWH - 3, 5); g.fill();
    g.textAlign = "left"; g.font = "600 10.5px ui-monospace,Menlo,monospace";
    g.fillStyle = faint ? P.dim2 : P.gradCss;
    g.fillText(lhs, box.x + 14, y);
    g.font = "10.5px ui-monospace,Menlo,monospace"; g.fillStyle = P.dim2;
    g.fillText(rhs, box.x + 78, y);
    g.textAlign = "right"; g.font = "700 11.5px ui-monospace,Menlo,monospace";
    g.fillStyle = faint ? P.dim : P.text;
    g.fillText(fmtg(val), box.x + box.w - 14, y);
    g.globalAlpha = 1;
  };

  for(let i=0;i<S.nin;i++)
    row(i, `∂L/∂w${SUB[i]}`, `= ∂L/∂z · x${SUB[i]} = ${fmtg(G.z.g)} · ${par(fmt(G.d.x[i]))}`,
        G.w[i].g, "gw" + i);
  row(S.nin, "∂L/∂b", `= ∂L/∂z · 1`, G.b.g, "gb");
  // not a parameter — this is the number a deeper net would hand to the layer below
  row(S.nin + 1, `∂L/∂x${S.nin === 1 ? SUB[0] : ""}`,
      S.nin === 1 ? `= ∂L/∂z · w₁ — handed to the layer below`
                  : `= ∂L/∂z · wᵢ = ${G.x.map(n => fmtg(n.g)).join(", ")}`,
      G.x[0].g, "gx", true);
}

// the only step that uses more than one point: the mean gradient, then one move
function updateRow(g, P, y, alpha){
  g.globalAlpha = alpha;
  g.fillStyle = P.frame; roundRect(g, 8, y, LY.right - 8, UH, 9); g.fill();
  g.strokeStyle = P.frameLine; g.lineWidth = 1.2; roundRect(g, 8, y, LY.right - 8, UH, 9); g.stroke();
  g.textAlign = "left"; g.font = "9.5px ui-monospace,Menlo,monospace"; g.fillStyle = P.dim;
  g.fillText(`5 · w ← w − η·∂L/∂w   with η = ${fmt(S.lr)} and the gradient averaged over all ${S.data.length} points`,
             18, y + 11);

  let x = 18;
  const pill = (name, cur, gr) => {
    const nv = cur - S.lr*gr;
    const txt = `${name} ${fmt(cur)} − ${fmt(S.lr)}·${par(fmtg(gr))} = ${fmt(nv)}`;
    g.font = "700 11px ui-monospace,Menlo,monospace";
    const w = g.measureText(txt).width + 16;
    g.fillStyle = ramp(nv, WMAX, P, 0.05, 0.38);
    roundRect(g, x, y + 17, w, 17, 5); g.fill();
    g.fillStyle = P.text; g.fillText(txt, x + 8, y + 26);
    x += w + 8;
  };
  for(let i=0;i<S.nin;i++) pill(`w${SUB[i]} ←`, S.w[i], GRAD.w[i]);
  pill("b ←", S.b, GRAD.b);
  g.globalAlpha = 1;
}

/* ---------- the two sparklines: a derivative you can see ---------- */
function sparkFrame(g, P, x, y, w, h, sx, sy, xr, yr){
  g.fillStyle = P.panel; roundRect(g, x, y, w, h, 5); g.fill();
  g.strokeStyle = P.line; g.lineWidth = 1; g.strokeRect(x + .5, y + .5, w, h);
  g.setLineDash([2,3]); g.strokeStyle = P.dim2;
  if(xr[0] < 0 && xr[1] > 0) seg(g, sx(0), y, sx(0), y + h);
  if(yr[0] < 0 && yr[1] > 0) seg(g, x, sy(0), x + w, sy(0));
  g.setLineDash([]);
}
function sparkCurve(g, P, f, sx, sy, x, y, w, h, n){
  g.save();
  g.beginPath(); g.rect(x, y, w, h); g.clip();
  g.strokeStyle = P.dim; g.lineWidth = 1.6;
  g.beginPath();
  for(let i=0;i<=n;i++){
    const u = i/n, v = f(u);
    i ? g.lineTo(sx(v[0]), sy(v[1])) : g.moveTo(sx(v[0]), sy(v[1]));
  }
  g.stroke();
  g.restore();
}
// slope drawn as a tangent through the point — the number ∂a/∂z, as a picture
function tangent(g, P, sx, sy, cx, cy, slope, half){
  g.strokeStyle = P.gradCss; g.lineWidth = 1.6;
  seg(g, sx(cx - half), sy(cy - slope*half), sx(cx + half), sy(cy + slope*half));
}
function dot(g, P, x, y){
  g.fillStyle = P.amber;
  g.beginPath(); g.arc(x, y, 3.4, 0, 6.2832); g.fill();
  g.strokeStyle = P.panel; g.lineWidth = 1.2; g.stroke();
}

function sparkAct(g, P, x, y, w, h){
  const A = ACT[S.act], z = G.z.d, a = G.a.d;
  const r = Math.max(3, Math.abs(z)*1.4);
  const xr = [-r, r];
  let lo = Infinity, hi = -Infinity;
  for(let i=0;i<=48;i++){ const v = A.raw(xr[0] + i/48*(xr[1] - xr[0]));
                          lo = Math.min(lo, v); hi = Math.max(hi, v); }
  const pad = Math.max(0.08, (hi - lo)*0.14);
  const yr = [lo - pad, hi + pad];
  const sx = v => x + (v - xr[0])/(xr[1] - xr[0])*w;
  const sy = v => y + h - (v - yr[0])/(yr[1] - yr[0])*h;

  sparkFrame(g, P, x, y, w, h, sx, sy, xr, yr);
  sparkCurve(g, P, u => { const zz = xr[0] + u*(xr[1] - xr[0]); return [zz, A.raw(zz)]; },
             sx, sy, x, y, w, h, 72);
  g.save(); g.beginPath(); g.rect(x, y, w, h); g.clip();
  tangent(g, P, sx, sy, z, a, G.a.local, (xr[1] - xr[0])*0.22);
  g.setLineDash([2,3]); g.strokeStyle = P.dim2; g.lineWidth = 1;
  seg(g, sx(z), y + h, sx(z), sy(a));
  g.setLineDash([]);
  dot(g, P, sx(z), sy(a));
  g.restore();
  g.textAlign = "center"; g.font = "9px ui-monospace,Menlo,monospace"; g.fillStyle = P.dim2;
  g.fillText("z", Math.max(x + 6, Math.min(x + w - 6, sx(z))), y + h - 6);
}

function sparkLoss(g, P, x, y, w, h){
  const a = G.a.d, t = G.d.t, e = a - t;
  const r = Math.max(0.7, Math.abs(e)*1.6);
  const xr = [t - r, t + r], yr = [-r*r*0.16, r*r*1.1];
  const sx = v => x + (v - xr[0])/(xr[1] - xr[0])*w;
  const sy = v => y + h - (v - yr[0])/(yr[1] - yr[0])*h;

  sparkFrame(g, P, x, y, w, h, sx, sy, xr, yr);
  sparkCurve(g, P, u => { const v = xr[0] + u*(xr[1] - xr[0]); return [v, (v - t)*(v - t)]; },
             sx, sy, x, y, w, h, 72);
  g.save(); g.beginPath(); g.rect(x, y, w, h); g.clip();
  g.setLineDash([3,3]); g.strokeStyle = P.dim2; g.lineWidth = 1;
  seg(g, sx(t), y, sx(t), y + h);
  g.setLineDash([]);
  tangent(g, P, sx, sy, a, e*e, 2*e, (xr[1] - xr[0])*0.2);
  dot(g, P, sx(a), sy(e*e));
  g.restore();
  g.textAlign = "center"; g.font = "9px ui-monospace,Menlo,monospace"; g.fillStyle = P.dim2;
  g.fillText("t", sx(t), y + h - 6);
  g.fillStyle = P.amber;
  g.fillText("a", Math.max(x + 6, Math.min(x + w - 6, sx(a))), y + 8);
}

/* ====================== the fit, and the loss ======================
   Deliberately plain: the points, the curve the neuron currently is, and one
   short bar per point joining what it says to what it should say. The faint
   trails behind each prediction are the only history kept — that is the
   evolution of every point, one update at a time.
   ================================================================== */
const FH2 = 258, PADL = 34, PADR = 12, PADT = 20, PADB = 24, GAPX = 30;

function fitWidth(){ return Math.max(660, (el("fitwrap").clientWidth || 900) - 2); }

// the plot box, derived once and used by both the drawing and the hover test —
// computing it twice is how hit targets drift off the dots
function plotBox(){
  const W = fitWidth(), splitX = Math.round(W*0.62);
  return {x:PADL, y:PADT, w: splitX - PADL - PADR, h: FH2 - PADT - PADB};
}
const XR = [-2.5, 2.5], XR2 = [-2.6, 2.6];
function yRange(){
  if(S.nin === 2) return XR2;
  const f = x => ACT[S.act].raw(S.w[0]*x + S.b);
  let lo = 0, hi = 1;
  for(const d of S.data){ lo = Math.min(lo, d.t); hi = Math.max(hi, d.t); }
  for(let k=0;k<=40;k++){
    const v = f(XR[0] + k/40*(XR[1] - XR[0]));
    if(isFinite(v)){ lo = Math.min(lo, v); hi = Math.max(hi, v); }
  }
  const pad = Math.max(0.3, (hi - lo)*0.15);
  return [lo - pad, hi + pad];
}
// where each data point lands on the plot
function plotXY(d){
  const B = plotBox();
  const xr = S.nin === 1 ? XR : XR2, yr = yRange();
  const vy = S.nin === 1 ? d.t : d.x[1];
  return {x: B.x + (d.x[0] - xr[0])/(xr[1] - xr[0])*B.w,
          y: B.y + B.h - (vy - yr[0])/(yr[1] - yr[0])*B.h};
}

// up to K states of (w, b), oldest first — the history the trails are drawn from
function history(K){
  const n = S.track.length;
  if(n < 2) return [];
  const out = [];
  for(let k=0;k<K;k++) out.push(S.track[Math.floor(k*(n - 1)/(K - 1))]);
  return out;
}

function paintFit(){
  const P = palette();
  const W = fitWidth();
  const g = fitCanvas(el("fit"), W, FH2);
  g.clearRect(0, 0, W, FH2);

  const splitX = Math.round(W*0.62);
  const A = plotBox();
  const B = {x: splitX + PADL + GAPX, y:PADT, w: W - splitX - PADL - PADR - GAPX, h: FH2 - PADT - PADB};

  (S.nin === 1 ? curvePanel : planePanel)(g, P, A);
  lossPanel(g, P, B);

  el("fitnote").innerHTML = diverged()
    ? `<span class="err">Diverged.</span> η = ${S.lr} takes steps wider than the bowl, so each one lands
       further out than the last. Lower η and reset.`
    : SETS[S.set].why +
      `<br><br>Each point has two marks: <b>●</b> where it should be, <b>○</b> where the neuron currently
       puts it. The bar between them is that point's error, and its colour is the number
       <span class="gv">∂L/∂a</span> is built from. The faint tails show where each ○ has been —
       ${S.steps ? `${S.steps} update${S.steps === 1 ? "" : "s"} of it so far.` : `press ▶ and they appear.`}`;
}

function axes(g, P, B, xr, yr, xlbl, ylbl, title){
  const sx = v => B.x + (v - xr[0])/(xr[1] - xr[0])*B.w;
  const sy = v => B.y + B.h - (v - yr[0])/(yr[1] - yr[0])*B.h;

  g.textAlign = "left"; g.textBaseline = "alphabetic";
  g.font = "600 10px ui-sans-serif,system-ui"; g.fillStyle = P.dim;
  g.fillText(title, B.x, B.y - 7);

  g.strokeStyle = P.line; g.lineWidth = 1;
  g.strokeRect(B.x + 0.5, B.y + 0.5, B.w, B.h);
  g.setLineDash([2,3]); g.strokeStyle = P.dim2;
  if(xr[0] < 0 && xr[1] > 0) seg(g, sx(0), B.y, sx(0), B.y + B.h);
  if(yr[0] < 0 && yr[1] > 0) seg(g, B.x, sy(0), B.x + B.w, sy(0));
  g.setLineDash([]);

  g.font = "9.5px ui-monospace,Menlo,monospace"; g.fillStyle = P.dim2;
  g.textAlign = "right"; g.textBaseline = "middle";
  g.fillText(fmt(yr[1], 1), B.x - 5, B.y + 5);
  g.fillText(fmt(yr[0], 1), B.x - 5, B.y + B.h - 5);
  g.textAlign = "center"; g.textBaseline = "alphabetic";
  g.fillText(fmt(xr[0], 1), B.x + 9, B.y + B.h + 13);
  g.fillText(fmt(xr[1], 1), B.x + B.w - 9, B.y + B.h + 13);
  g.fillText(xlbl, B.x + B.w/2, B.y + B.h + 13);
  g.save();
  g.translate(B.x - 23, B.y + B.h/2); g.rotate(-Math.PI/2);
  g.textAlign = "center"; g.fillText(ylbl, 0, 0);
  g.restore();
  return {sx, sy};
}

// one input: the neuron is a curve over x, and every point has a target and a prediction
function curvePanel(g, P, B){
  const xr = XR, yr = yRange();
  const f = (w, b, x) => ACT[S.act].raw(w*x + b);
  const {sx, sy} = axes(g, P, B, xr, yr, "x", "value",
                        `the fit — ${S.data.length} points, and where the neuron puts each of them`);

  g.save();
  g.beginPath(); g.rect(B.x, B.y, B.w, B.h); g.clip();

  const hist = S.trails ? history(Math.min(14, S.track.length)) : [];

  // where the curve started, so the whole move is visible at a glance
  if(S.track.length){
    const t0 = S.track[0];
    g.strokeStyle = P.dim2; g.globalAlpha = 0.45; g.lineWidth = 1;
    g.setLineDash([4,4]);
    g.beginPath();
    for(let j=0;j<=80;j++){
      const x = xr[0] + j/80*(xr[1] - xr[0]);
      j ? g.lineTo(sx(x), sy(f(t0.w[0], t0.b, x))) : g.moveTo(sx(x), sy(f(t0.w[0], t0.b, x)));
    }
    g.stroke(); g.setLineDash([]); g.globalAlpha = 1;
  }

  // the tail each prediction has dragged behind it, fading towards the oldest end
  // so it reads as a direction and not as a stray line
  if(hist.length > 1)
    S.data.forEach((d, i) => {
      const ys = hist.map(t => sy(f(t.w[0], t.b, d.x[0])));
      ys.push(sy(f(S.w[0], S.b, d.x[0])));
      const top = i === S.sample ? 0.6 : 0.24;
      g.strokeStyle = P.dim2;
      g.lineWidth = i === S.sample ? 1.6 : 1;
      for(let k=1;k<ys.length;k++){
        g.globalAlpha = top*(0.15 + 0.85*k/(ys.length - 1));
        seg(g, sx(d.x[0]), ys[k-1], sx(d.x[0]), ys[k]);
      }
      g.globalAlpha = 1;
    });

  // the error of every point: target to prediction
  S.data.forEach((d, i) => {
    const y = f(S.w[0], S.b, d.x[0]), r = y - d.t;
    g.strokeStyle = ramp(r, 1, P, 0.12, 0.9);
    g.lineWidth = i === S.sample ? 4 : 2.4;
    seg(g, sx(d.x[0]), sy(d.t), sx(d.x[0]), sy(y));
  });

  // the curve the neuron currently is
  g.strokeStyle = P.accent; g.lineWidth = 2;
  g.beginPath();
  for(let j=0;j<=120;j++){
    const x = xr[0] + j/120*(xr[1] - xr[0]);
    j ? g.lineTo(sx(x), sy(f(S.w[0], S.b, x))) : g.moveTo(sx(x), sy(f(S.w[0], S.b, x)));
  }
  g.stroke();

  // ● the target, ○ what the neuron says
  S.data.forEach((d, i) => {
    const on = i === S.sample;
    const py = sy(f(S.w[0], S.b, d.x[0]));
    g.fillStyle = P.panel;
    g.beginPath(); g.arc(sx(d.x[0]), py, on ? 4 : 2.8, 0, 6.2832); g.fill();
    g.strokeStyle = P.accent; g.lineWidth = on ? 1.8 : 1.2; g.stroke();

    g.fillStyle = ramp(d.t, Math.max(1, Math.abs(yr[0]), Math.abs(yr[1])), P, 0.3, 0.95);
    g.beginPath(); g.arc(sx(d.x[0]), sy(d.t), on ? 5.5 : 4, 0, 6.2832); g.fill();
    g.strokeStyle = on ? P.text : P.frameLine; g.lineWidth = on ? 2 : 1; g.stroke();
  });
  g.restore();
}

// two inputs: the neuron is a plane through a squash, so its boundary is a line
function planePanel(g, P, B){
  const xr = XR2, yr = XR2;
  const CELL = 6, nx = Math.ceil(B.w/CELL), ny = Math.ceil(B.h/CELL);
  g.save();
  g.beginPath(); g.rect(B.x, B.y, B.w, B.h); g.clip();       // the last row and column overhang
  for(let i=0;i<nx;i++) for(let j=0;j<ny;j++){
    const x1 = xr[0] + (i + 0.5)/nx*(xr[1] - xr[0]);
    const x2 = yr[1] - (j + 0.5)/ny*(yr[1] - yr[0]);
    const a = ACT[S.act].raw(S.w[0]*x1 + S.w[1]*x2 + S.b);
    g.fillStyle = ramp(a - 0.5, 0.5, P, 0.02, 0.38);          // 0.5 is the boundary, so centre on it
    g.fillRect(B.x + i*CELL, B.y + j*CELL, CELL, CELL);
  }
  g.restore();

  const {sx, sy} = axes(g, P, B, xr, yr, "x₁", "x₂",
                        `the fit — ${S.data.length} points, and the line the neuron draws between them`);

  // w·x + b = 0 is a straight line whatever the squash does to it afterwards
  const bound = (w, b, col, dash, width) => {
    if(Math.abs(w[1]) < 1e-9 && Math.abs(w[0]) < 1e-9) return;
    g.save(); g.beginPath(); g.rect(B.x, B.y, B.w, B.h); g.clip();
    g.strokeStyle = col; g.lineWidth = width;
    if(dash) g.setLineDash([4,4]);
    if(Math.abs(w[1]) > Math.abs(w[0])){
      const at = u => -(w[0]*u + b)/w[1];
      seg(g, sx(xr[0]), sy(at(xr[0])), sx(xr[1]), sy(at(xr[1])));
    } else {
      const at = v => -(w[1]*v + b)/w[0];
      seg(g, sx(at(yr[0])), sy(yr[0]), sx(at(yr[1])), sy(yr[1]));
    }
    g.setLineDash([]); g.restore();
  };
  if(S.track.length) bound(S.track[0].w, S.track[0].b, P.dim2, true, 1);
  bound(S.w, S.b, P.accent, false, 2);

  S.data.forEach((d, i) => {
    const p = plotXY(d);
    const a = ACT[S.act].raw(S.w[0]*d.x[0] + S.w[1]*d.x[1] + S.b);
    const r = a - d.t;
    g.lineWidth = 1 + 5*Math.min(1, Math.abs(r));            // the ring is the error
    g.strokeStyle = ramp(r, 1, P, 0.10, 0.95);
    g.beginPath(); g.arc(p.x, p.y, 8, 0, 6.2832); g.stroke();
    g.fillStyle = ramp(d.t - 0.5, 0.5, P, 0.35, 0.95);
    g.beginPath(); g.arc(p.x, p.y, i === S.sample ? 5.5 : 4, 0, 6.2832); g.fill();
    g.strokeStyle = i === S.sample ? P.text : P.frameLine;
    g.lineWidth = i === S.sample ? 2 : 1;
    g.stroke();
  });
}

function lossPanel(g, P, B){
  const pts = S.hist.concat([isFinite(LOSS) ? LOSS : 0]);
  const top = Math.max(1e-9, ...pts.filter(v => isFinite(v)));
  const {sx, sy} = axes(g, P, B, [0, Math.max(1, pts.length - 1)], [0, top], "update", "L",
                        `the loss it produces — ${S.steps} update${S.steps === 1 ? "" : "s"}`);

  if(pts.length > 1){
    g.save(); g.beginPath(); g.rect(B.x, B.y, B.w, B.h); g.clip();
    g.beginPath();
    pts.forEach((v, i) => i ? g.lineTo(sx(i), sy(v)) : g.moveTo(sx(i), sy(v)));
    g.strokeStyle = P.amber; g.lineWidth = 1.8; g.stroke();
    g.lineTo(sx(pts.length - 1), sy(0)); g.lineTo(sx(0), sy(0)); g.closePath();
    g.fillStyle = ramp(1, 1, P, 0, 0.10); g.fill();
    g.restore();
  }
  const last = pts.length - 1;
  g.fillStyle = P.amber;
  g.beginPath(); g.arc(sx(last), sy(pts[last]), 3.5, 0, 6.2832); g.fill();
  g.font = "600 11px ui-monospace,Menlo,monospace";
  g.textAlign = "right"; g.textBaseline = "alphabetic";
  g.fillText("L = " + fmt(pts[last], 4), B.x + B.w - 5, B.y + 14);
}

/* ======================= the ledger under the canvas ======================= */
const fv  = v => `<span class="fv">${fmt(v)}</span>`;
const fvp = v => `<span class="fv">${par(fmt(v))}</span>`;
const wv = v => `<span class="wv">${fmt(v)}</span>`;
const gv  = v => `<span class="gv">${fmtg(v)}</span>`;
const gvp = v => `<span class="gv">${par(fmtg(v))}</span>`;
function mrow(phase, label, html){
  return `<div class="mrow${phase === S.phase ? " now" : ""}"><span class="mlbl">${label}</span>
          <span class="mval">${html}</span></div>`;
}

// every chip carries the same colour its number has on the canvas
function chip(id, body, fill){
  return `<span class="term${S.chip === id ? " on" : ""}"${fill ? ` style="background:${fill}"` : ""}
          data-chip="${id}">${body}</span>`;
}
function plain(body, fill){
  return `<span class="term"${fill ? ` style="background:${fill}"` : ""}>${body}</span>`;
}

function selInfo(){
  const s = S.sel;
  if(s.k === "b") return {name:"b", node:G.b, batch:GRAD.b, local:1, localTxt:"1"};
  if(s.k === "x") return {name:"x" + SUB[s.i], node:G.x[s.i], batch:null,
                          local:S.w[s.i], localTxt:"w" + SUB[s.i]};
  return {name:"w" + SUB[s.i], node:G.w[s.i], batch:GRAD.w[s.i],
          local:G.d.x[s.i], localTxt:"x" + SUB[s.i]};
}

function renderMath(){
  const P = palette(), A = ACT[S.act], p = S.phase, out = [];

  out.push(mrow(-1, "point", `<b>${S.sample + 1} of ${S.data.length}</b>
    <span style="color:var(--dim2)">— hover a point on the fit below, or use ◀ ▶</span>`));

  const terms = G.prod.map((prod, i) =>
    chip("t" + i, `${wv(S.w[i])} · ${fv(G.d.x[i])} = <b>${fmt(prod.d)}</b>`, ramp(prod.d, ZMAX, P, 0.06, 0.4)));
  out.push(mrow(0, "1 · sum", terms.join(" + ") + " + " +
    chip("tb", `b = ${wv(S.b)}`, ramp(S.b, WMAX, P, 0.06, 0.4)) +
    ` &nbsp;=&nbsp; <span class="res">${fmt(G.z.d)}</span>`));

  if(p >= 1) out.push(mrow(1, "2 · activate",
    `a = ${A.lbl}( ${fv(G.z.d)} ) &nbsp;=&nbsp; <span class="res">${fmt(G.a.d)}</span>
     <span style="color:var(--dim2)">&nbsp; slope here ∂a/∂z = ${A.der} =</span> ${gv(G.a.local)}`));

  if(p >= 2) out.push(mrow(2, "3 · loss",
    `L = ( ${fv(G.a.d)} − ${fvp(G.d.t)} )² &nbsp;=&nbsp; <span class="res">${fmt(G.L.d, 4)}</span>
     <span style="color:var(--dim2)">&nbsp; batch L = mean of ${S.data.length} =</span>
     <b>${fmt(LOSS, 4)}</b>`));

  if(p >= 3){
    const s = selInfo();
    out.push(mrow(3, "4 · chain rule",
      `∂L/∂${s.name} &nbsp;=&nbsp; ` +
      chip("dLda", "∂L/∂a") + " · " + chip("dadz", "∂a/∂z") + " · " + chip("dzdp", `∂z/∂${s.name}`) +
      " &nbsp;=&nbsp; " +
      chip("dLda", `${gv(G.a.g)}`, gramp(G.a.g, P, 0.06, 0.4)) + " · " +
      chip("dadz", `${gv(G.a.local)}`, gramp(G.a.local, P, 0.06, 0.4)) + " · " +
      chip("dzdp", `${fv(s.local)}`, ramp(s.local, XMAX, P, 0.06, 0.4)) +
      ` &nbsp;=&nbsp; <span class="res">${fmtg(s.node.g)}</span>
        <span style="color:var(--dim2)">&nbsp;— point the cursor at a wire to switch parameter</span>`));
    out.push(mrow(3, "all " + S.data.length,
      GRAD.w.map((v, i) => plain(`∂L/∂w${SUB[i]} = ${gv(v)}`, gramp(v, P, 0.06, 0.45))).join(" ") +
      plain(`∂L/∂b = ${gv(GRAD.b)}`, gramp(GRAD.b, P, 0.06, 0.45)) +
      ` <span style="color:var(--dim2)">— the mean over every point, and what the update actually uses</span>`));
  }

  if(p >= 4){
    const rows = GRAD.w.map((v, i) =>
      plain(`w${SUB[i]} ← ${wv(S.w[i])} − ${fmt(S.lr)}·${gvp(v)} = <b>${fmt(S.w[i] - S.lr*v)}</b>`,
            ramp(S.w[i] - S.lr*v, WMAX, P, 0.06, 0.4)));
    rows.push(plain(`b ← ${wv(S.b)} − ${fmt(S.lr)}·${gvp(GRAD.b)} = <b>${fmt(S.b - S.lr*GRAD.b)}</b>`,
                    ramp(S.b - S.lr*GRAD.b, WMAX, P, 0.06, 0.4)));
    out.push(mrow(4, "5 · update", rows.join(" ")));
  }

  el("readout").innerHTML = out.join("");
}

function markChips(){
  el("readout").querySelectorAll("[data-chip]").forEach(n =>
    n.classList.toggle("on", !!S.chip && n.dataset.chip === S.chip));
}

/* ============================== side panels ============================== */
function renderFormula(){
  const A = ACT[S.act], n = S.nin;
  el("formula").innerHTML =
    `<span class="lbl">pytorch</span>nn.Linear(${n}, 1)${S.act === "linear" ? "" : " → " + S.act}
     <span class="lbl">then</span>nn.MSELoss()
     <span class="lbl">forward</span>z = Σ <span class="wv">wᵢ</span>·<span class="fv">xᵢ</span> + <span class="wv">b</span>
     a = ${A.call}
     L = mean over the batch of (a − t)²
     <span class="lbl">batch</span><span class="big">${S.data.length}</span> points, all of them, every step
     <span class="lbl">parameters</span><span class="big">${n + 1}</span> — ${n} weight${n === 1 ? "" : "s"} + 1 bias
     <span class="lbl">gradients</span><span class="big">${n + 1}</span> — one per parameter, same shapes`;

  el("shapenote").innerHTML =
    `Every point gets its own forward <em>and</em> backward pass; the ${n + 1} gradients that come back are
     averaged, and that average is the single step the parameters take. Backprop costs about one multiply-add
     per parameter in each direction, which is why training is roughly three forward passes' worth of work.`;
}

function renderChain(){
  const A = ACT[S.act], s = selInfo();
  el("chain").innerHTML =
    `<span class="lbl">the two local pieces</span>
     <span class="gv">∂L/∂a</span> = 2(a − t)
     <span class="gv">∂a/∂z</span> = ${A.der}
     <span class="lbl">every parameter</span>
     ∂L/∂wᵢ = <span class="gv">∂L/∂a</span> · <span class="gv">∂a/∂z</span> · <span class="fv">xᵢ</span>
     ∂L/∂b &nbsp;= <span class="gv">∂L/∂a</span> · <span class="gv">∂a/∂z</span>
     <span class="lbl">handed to the layer below</span>
     ∂L/∂xᵢ = <span class="gv">∂L/∂a</span> · <span class="gv">∂a/∂z</span> · <span class="wv">wᵢ</span>
     <span class="lbl">update</span>
     w ← w − <b>η</b> · ∂L/∂w &nbsp;<span style="color:var(--dim2)">η = ${S.lr}</span>
     <span class="lbl">point ${S.sample + 1} — ∂L/∂${s.name}</span>
     ${gv(G.a.g)} · ${gv(G.a.local)} · ${fv(s.local)} = <span class="big gv">${fmtg(s.node.g)}</span>
     ${s.batch === null ? "" :
       `<span class="lbl">averaged over all ${S.data.length}</span>
        <span class="big gv">${fmtg(s.batch)}</span>`}`;
}

function renderDiag(){
  const {rows, worst} = gradCheck();
  const ok = worst < 1e-4;
  const out = [];

  out.push(`<div style="margin-bottom:8px">
    <span class="flag ${ok ? "ok" : "bad"}">gradient check ${ok ? "ok" : "off"}</span>
    <span style="color:var(--dim);font-size:11.5px;margin-left:7px">worst relative error
      ${worst === 0 ? "0" : worst.toExponential(1)}</span></div>`);

  out.push(`<table class="stats">
    <tr><th>parameter</th><th>engine</th><th>(L(p+h)−L(p−h))/2h</th></tr>
    ${rows.map(r => `<tr><td>${r.name}</td><td>${fmtg(r.engine)}</td><td>${fmtg(r.num)}</td></tr>`).join("")}
    </table>`);

  const notes = [`The right column never touches the graph — it nudges each parameter by h = 1e-4 and
                  re-runs the whole batch twice. If the engine and the difference quotient ever disagreed,
                  every number on this page would be a lie.`];

  const slopes = GS.map(g => Math.abs(g.a.local));
  const maxSlope = Math.max(...slopes, 0);

  if(S.act === "relu" && GS.some(g => Math.abs(g.z.d) < 1e-6))
    notes.push(`<span class="err">Some point has z = 0 exactly.</span> relu has no derivative there — the
                engine takes the subgradient 0, the difference quotient straddles the kink. Both are
                defensible; nudge a weight and it goes away.`);

  if(maxSlope < 1e-3 && S.act !== "linear")
    notes.push(`<b>∂a/∂z ≈ 0 at every point.</b> ${S.act === "relu"
      ? `z is negative everywhere, so relu outputs 0 and blocks the gradient completely. Every gradient is
         exactly 0 and training does <em>nothing</em> — a dead ReLU. Look at the fit: the curve is flat on
         the floor and the error bars are at full intensity, yet nothing moves.`
      : `The activation is saturated on every point — the violet tangent inside the neuron is flat. Each
         gradient gets multiplied by that, so the weights barely move however wrong the fit is. This is the
         vanishing gradient, in one layer.`}`);
  else if(maxSlope < 0.05 && S.act !== "linear")
    notes.push(`The steepest ∂a/∂z in the batch is only ${fmtg(maxSlope)}. Whatever the loss wants, the
                weights get a fraction of it — and the bars on the fit shrink very slowly as a result.`);

  if(diverged())
    notes.push(`<span class="err">The loss has blown up.</span> η = ${S.lr} takes steps larger than the
                bowl is wide, so each one lands further out than the last. Gradient descent only converges
                when η is below 2/curvature.`);
  else if(S.steps > 0 && LOSS < 1e-5)
    notes.push(`Converged: L = ${LOSS.toExponential(1)}. The error bars have faded to nothing and the
                gradients with them, so the updates have nothing left to do.`);

  if(S.act === "linear")
    notes.push(`With a linear activation this is least squares: the loss surface is a single bowl, so there
                is exactly one minimum and gradient descent always finds it.`);

  out.push(`<div class="note">${notes.join("<br><br>")}</div>`);
  el("diag").innerHTML = out.join("");
}

/* ============================== controls ============================== */
const CAPS = {
  z: `<b>The weighted sum.</b> Each wire carries one product w·x into the card, which adds them and the bias.
      Wire thickness is |w| and its hue is the sign — <b>drag a wire up or down to change that weight</b> and
      watch every number downstream follow.`,
  a: `<b>The activation.</b> The little plot is the function itself, with this point marked on it. The violet
      line through the mark is ∂a/∂z — flat means the gradient dies here, steep means it passes through.`,
  L: `<b>The loss.</b> The bowl is (a − t)² as a would move; the mark is where this point sits on it. Its
      violet tangent is 2(a − t), the first number the backward pass produces.`,
  g: `<b>The backward pass</b>, station for station under the forward one. Each violet card is one
      multiplication by a local derivative. Point at a wire and the dashed line shows which forward value
      its gradient reuses — that is why activations have to be kept.`,
  u: `<b>The update.</b> The gradients above are for this one point; the step uses their mean over all of
      them. Press ▶ to apply it — the curve moves and one point lands on the loss plot.`,
};

function slider(id, label, cls, min, max, step, hint){
  return `<div class="ctrl" id="bp_ctrl_${id}">
    <label for="bp_r_${id}"><span class="${cls}">${label}</span><b id="bp_v_${id}"></b></label>
    <input type="range" id="bp_r_${id}" min="${min}" max="${max}" step="${step}">
    ${hint ? `<div class="hint">${hint}</div>` : ""}
  </div>`;
}

function buildControls(){
  el("setseg").innerHTML = Object.entries(SETS).map(([k, d]) =>
    `<button data-set="${k}"><b>${d.name}</b><i>${d.note}</i></button>`).join("");

  el("rail").innerHTML = PHASES.map((p, i) =>
    `<button data-phase="${i}"><b>${i + 1} · ${p.lbl}</b><i id="bp_eq${i}"></i></button>`).join("");

  el("wctrls").innerHTML =
    slider("w0", "w₁", "wv", -4, 4, 0.05) +
    slider("w1", "w₂", "wv", -4, 4, 0.05) +
    slider("b",  "b",  "wv", -4, 4, 0.05, "Or drag the wires on the diagram directly.");

  const bind = (id, fn) => el("r_" + id).addEventListener("input", e => { fn(+e.target.value); refresh(); });
  bind("w0", v => { S.w[0] = v; });
  bind("w1", v => { S.w[1] = v; });
  bind("b",  v => { S.b = v; });

  const segs = (id, attr, fn) => document.querySelectorAll(`#bp_${id} button`).forEach(b =>
    b.addEventListener("click", () => fn(b.dataset[attr])));
  segs("setseg", "set", v => { stop(); loadSet(v); refresh(); });
  segs("actseg", "act", v => { S.act = v; refresh(); });
  segs("lrseg",  "lr",  v => { S.lr = +v; refresh(); });
  segs("rail", "phase", v => { stop(); S.phase = +v; refresh(); });

  el("rand").addEventListener("click", randomise);
  el("train").addEventListener("click", () => train(50));
  el("reset").addEventListener("click", reset);
  el("prev").addEventListener("click", () => { stop(); S.phase = Math.max(0, S.phase - 1); refresh(); });
  el("next").addEventListener("click", () => { stop(); step(); });
  el("play").addEventListener("click", () => S.playing ? stop() : start());
  el("sprev").addEventListener("click", () => pick((S.sample - 1 + S.data.length) % S.data.length));
  el("snext").addEventListener("click", () => pick((S.sample + 1) % S.data.length));
  el("trails").addEventListener("change", e => { S.trails = e.target.checked; paintFit(); });
}

function pick(i){
  S.sample = i;
  G = GS[i];
  renderMath(); renderChain(); paint(); paintFit(); syncControls();
}

function step(){
  if(S.phase < PHASES.length - 1) S.phase++;
  else { applyUpdate(); S.phase = 0; }
  refresh();
}
let timer = 0;
function start(){ S.playing = true; timer = setInterval(step, 850); syncControls(); }
// the tab bar stops the loop on the way out, which can happen before the
// controls have ever been built — hence the guard
function stop(){
  S.playing = false;
  clearInterval(timer); timer = 0;
  if(started) syncControls();
}

function syncControls(){
  const set = (id, v) => { el("r_" + id).value = v; el("v_" + id).textContent = fmt(v); };
  set("w0", S.w[0]); set("w1", S.w[1]); set("b", S.b);
  el("ctrl_w1").classList.toggle("off", S.nin < 2);

  const on = (id, attr, val) => document.querySelectorAll(`#bp_${id} button`).forEach(b =>
    b.classList.toggle("on", b.dataset[attr] === String(val)));
  on("setseg", "set", S.set);
  on("actseg", "act", S.act);
  on("lrseg",  "lr",  S.lr);

  document.querySelectorAll("#bp_rail button").forEach((b, i) => {
    b.classList.toggle("on", i === S.phase);
    b.classList.toggle("done", i < S.phase);
    el("eq" + i).textContent = PHASES[i].eq();
  });

  el("acthint").innerHTML = ACT[S.act].hint;
  el("vizcap").innerHTML = CAPS[PHASES[S.phase].id];   // not in paint(), which runs per frame
  el("play").textContent = S.playing ? "❚❚ pause" : "▶ play";
  el("prev").disabled = S.phase === 0;
  el("slabel").textContent = `point ${S.sample + 1} / ${S.data.length}`;
  el("trails").checked = S.trails;
  el("stephint").innerHTML = S.phase === PHASES.length - 1
    ? `▶ applies the update and starts the next pass.`
    : `${S.steps} update${S.steps === 1 ? "" : "s"} so far.`;
  el("trainhint").innerHTML = diverged()
    ? `<span class="err">Diverged — reset before training further.</span>`
    : `Runs 50 full passes over all ${S.data.length} points.`;
  el("setnote").innerHTML = `<b>${S.data.length}</b> points, <b>${S.nin}</b> input${S.nin === 1 ? "" : "s"}
    — the neuron is fitted to all of them at once.`;
}

/* ============================== interaction ============================== */
function hit(ev){
  if(!LY) return null;
  const r = el("viz").getBoundingClientRect();
  const x = (ev.clientX - r.left)*(LY.w/r.width), y = (ev.clientY - r.top)*(LY.h/r.height);

  for(let i=0;i<S.nin;i++)
    if(Math.hypot(x - IN_X, y - inY(i)) < IN_R + 7) return {part:"x" + i, sel:{k:"x", i}};
  if(Math.hypot(x - IN_X, y - biasY()) < IN_R + 7) return {part:"b", sel:{k:"b"}, drag:{k:"b"}};

  const inBox = (bx, bw, by, bh) => x >= bx && x <= bx + bw && y >= by && y <= by + bh;
  if(inBox(LY.sx, LY.sum, FY, FH)){
    for(let i=0;i<=S.nin;i++)
      if(Math.abs(y - rowY(i)) < ROWH/2)
        return i < S.nin ? {part:"r" + i, sel:{k:"w", i}} : {part:"r" + S.nin, sel:{k:"b"}};
    return {part:"sum", sel:null};
  }
  if(inBox(LY.ax, LY.act, FY, FH))  return {part:"act",  sel:null};
  if(inBox(LY.lx, LY.loss, FY, FH)) return {part:"loss", sel:null};

  if(S.phase >= 3){
    const B = backBoxes();
    if(inBox(B.a.x, B.a.w, BY, LY.bh)) return {part:"ga", sel:null};
    if(inBox(B.z.x, B.z.w, BY, LY.bh)) return {part:"gz", sel:null};
    if(inBox(B.p.x, B.p.w, BY, LY.bh)){
      const i = Math.floor((y - BY - 26)/ROWH);
      if(i >= 0 && i < S.nin)   return {part:"gw" + i, sel:{k:"w", i}};
      if(i === S.nin)           return {part:"gb", sel:{k:"b"}};
      if(i === S.nin + 1)       return {part:"gx", sel:{k:"x", i:0}};
      return {part:"gp", sel:null};
    }
  }

  for(let i=0;i<S.nin;i++)
    if(segDist(x, y, wireSeg(i)) < 11) return {part:"e" + i, sel:{k:"w", i}, drag:{k:"w", i}};
  if(segDist(x, y, biasSeg()) < 11) return {part:"eb", sel:{k:"b"}, drag:{k:"b"}};
  return null;
}

// which data point is nearest the cursor on the fit plot
function hitPoint(ev){
  const r = el("fit").getBoundingClientRect();
  const x = (ev.clientX - r.left)*(fitWidth()/r.width), y = (ev.clientY - r.top)*(FH2/r.height);
  const B = plotBox();
  if(x < B.x - 8 || x > B.x + B.w + 8) return -1;

  let best = -1, bd = 18;
  S.data.forEach((d, i) => {
    const p = plotXY(d);
    const dd = Math.hypot(x - p.x, y - p.y);
    if(dd < bd){ bd = dd; best = i; }
  });
  return best;
}

function wire(){
  const cv = el("viz");

  cv.addEventListener("mousemove", e => {
    if(S.drag){                                     // a weight is being dragged on its own wire
      const dy = e.clientY - S.drag.y0;
      const v = Math.max(-4, Math.min(4, S.drag.v0 - dy*0.012));
      if(S.drag.k === "b") S.b = v; else S.w[S.drag.i] = v;
      refresh();
      return;
    }
    const h = hit(e);
    const part = h ? h.part : null;
    const sel  = h && h.sel ? h.sel : S.sel;
    cv.style.cursor = h && h.drag ? "ns-resize" : "default";
    if(part === S.hover && sel.k === S.sel.k && sel.i === S.sel.i) return;
    S.hover = part; S.sel = sel;
    renderMath(); renderChain(); paint();
  });
  cv.addEventListener("mousedown", e => {
    const h = hit(e);
    if(!h || !h.drag) return;
    e.preventDefault();
    S.drag = {...h.drag, y0: e.clientY, v0: h.drag.k === "b" ? S.b : S.w[h.drag.i]};
  });
  addEventListener("mouseup", () => { if(S.drag){ S.drag = null; refresh(); } });
  cv.addEventListener("mouseleave", () => {
    if(S.hover === null || S.drag) return;
    S.hover = null; paint();
  });

  // hovering the fit scrubs through the points — the diagram fills with whichever
  // one is under the cursor, and keeps the last when the cursor leaves
  el("fit").addEventListener("mousemove", e => {
    const i = hitPoint(e);
    el("fit").style.cursor = i >= 0 ? "pointer" : "default";
    if(i >= 0 && i !== S.sample) pick(i);
  });

  el("readout").addEventListener("mouseover", e => {
    const t = e.target.closest("[data-chip]");
    const id = t ? t.dataset.chip : "";
    if(id === S.chip) return;
    S.chip = id; markChips(); paint();
  });
  el("readout").addEventListener("mouseleave", () => {
    if(!S.chip) return;
    S.chip = ""; markChips(); paint();
  });

  addEventListener("keydown", e => {
    if(hidden() || e.target.tagName === "INPUT") return;
    if(e.key === "ArrowRight"){ e.preventDefault(); stop(); step(); }
    else if(e.key === "ArrowLeft"){ e.preventDefault(); stop(); S.phase = Math.max(0, S.phase - 1); refresh(); }
    else if(e.key === "ArrowDown"){ e.preventDefault(); pick((S.sample + 1) % S.data.length); }
    else if(e.key === "ArrowUp"){ e.preventDefault(); pick((S.sample - 1 + S.data.length) % S.data.length); }
  });

  let t;
  addEventListener("resize", () => {
    clearTimeout(t);
    t = setTimeout(() => { if(!hidden()){ paint(); paintFit(); } }, 120);
  });
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
    if(!hidden()){ paint(); paintFit(); }
  });
}

const hidden = () => document.getElementById("view-backprop").classList.contains("hidden");

/* the pulse in the backward band only animates while it is on screen */
let raf = 0, frame = 0;
function tick(){
  raf = 0;
  if(hidden() || S.phase < 3) return;
  frame++;
  paint();
  raf = requestAnimationFrame(tick);
}
function animate(){
  if(!raf && S.phase >= 3 && !hidden()) raf = requestAnimationFrame(tick);
}

/* ================================ main ================================ */
function refresh(){
  if(S.sel.k !== "b" && S.sel.i >= S.nin) S.sel = {k:"w", i:0};
  compute();
  scales();
  if(S.playing && diverged()) stop();
  syncControls();
  renderMath();
  renderFormula();
  renderChain();
  renderDiag();
  paint();
  paintFit();
  animate();
}

let started = false;
return {
  show(){
    if(!started){ started = true; loadSet(S.set); buildControls(); wire(); }
    refresh();
  },
  hide(){ stop(); }
};
})();
