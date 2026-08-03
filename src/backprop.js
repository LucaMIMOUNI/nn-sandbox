/* ==================================================================
   MODULE 3 — Backpropagation

   The page is one neuron drawn twice: a forward band along the top,
   and the backward band directly underneath it, station by station,
   so every gradient sits below the value it came from. Each station
   carries its own equation with this point's numbers substituted in,
   and is filled with the colour of the number it holds — so the whole
   pass can be read as a picture first and as arithmetic second.

   Seven steps, not five: the backward half is broken up the same way
   the forward half is, one multiplication per step, and the chain rule
   printed above the diagram gains a factor at each of them.
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
    for:"one", nin:1, act:"linear", w:[-0.6], b:-0.9, lr:0.2,
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
    for:"one", nin:1, act:"sigmoid", w:[0.4], b:0.1, lr:1.5,
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
    name:"two clusters", note:"one line is enough",
    for:"both", nin:2, act:"sigmoid", w:[0.3, -0.4], b:0, lr:1.0, netlr:1.0, netact:"tanh",
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

  /* ---- the two that a single neuron cannot do at all ---- */
  xor: {
    name:"xor", note:"two lines needed — no single one works",
    for:"net", nin:2, netact:"tanh", netlr:0.6,
    make(){
      const r = prng(4242), out = [];
      for(const c of [[-1.15,-1.15,0], [1.15,1.15,0], [-1.15,1.15,1], [1.15,-1.15,1]])
        for(let i=0;i<8;i++)
          out.push({x:[r2(c[0] + 0.4*gauss(r)), r2(c[1] + 0.4*gauss(r))], t:c[2]});
      return out;
    },
    why:`Opposite corners share a class, so <b>no straight line separates them</b> — a single neuron is
         stuck at L ≈ 0.25 forever. Each hidden neuron draws one line; the output neuron combines them.
         That is the whole reason for a hidden layer.`,
  },
  circle: {
    name:"a circle", note:"inside vs outside — a closed boundary",
    for:"net", nin:2, netact:"tanh", netlr:0.6,
    make(){
      const r = prng(90210), out = [];
      for(let i=0;i<16;i++){
        const th = r()*6.2832, rad = 0.25 + 0.6*r();
        out.push({x:[r2(rad*Math.cos(th)), r2(rad*Math.sin(th))], t:1});
      }
      for(let i=0;i<20;i++){
        const th = r()*6.2832, rad = 1.45 + 0.55*r();
        out.push({x:[r2(rad*Math.cos(th)), r2(rad*Math.sin(th))], t:0});
      }
      return out;
    },
    why:`The boundary has to close on itself, which no half-plane can do. With enough hidden neurons the
         network fences the middle in with straight lines and rounds them off — watch the heat map on the
         right go from a smear to a blob.`,
  },
};

/* ============================ the network ============================
   The page is one neuron until you give it a hidden layer. Everything from
   here on is written for a list of layers; the single neuron is the case
   where that list holds one layer of one neuron, and it keeps its own
   drawing code, because that picture is what the page is for.
   ===================================================================== */
const NETS = {
  one: {lbl:"1 neuron",       note:"a = f(w·x + b)",     hidden:0},
  h1:  {lbl:"1 hidden layer", note:"2 → h → 1",          hidden:1},
  h2:  {lbl:"2 hidden layers",note:"2 → h → h → 1",      hidden:2},
};
const isNet = () => S.net !== "one";
const shape = () => isNet() ? [2, ...Array(NETS[S.net].hidden).fill(S.width), 1]
                            : [S.nin, 1];
const NLAY = () => shape().length - 1;
// hidden layers use the activation you picked; the last one is always a sigmoid,
// because every dataset worth a hidden layer here has 0/1 targets
const layerAct = l => l === NLAY() - 1 ? ACT.sigmoid : ACT[S.act];
const nparams = () => shape().slice(1).reduce((s, n, l) => s + n*(shape()[l] + 1), 0);

const SUB = ["₁","₂","₃","₄"];
const SUP = ["¹","²","³"];
const el = id => document.getElementById("bp_" + id);

/* ================================ state ================================ */
const S = {
  net:"one", width:4,             // architecture
  set:"line", nin:1, data:[],
  w:[-0.6, 0], b:-0.9,            // the single neuron
  w0:[-0.6, 0], b0:-0.9,          // what "reset" goes back to
  W:[], B:[], W0:[], B0:[],       // the layered net — W[layer][out][in]
  act:"linear", lr:0.2,
  sample: 6,                      // which point fills the stations
  phase: 0,
  sel: {k:"w", i:0, l:0, j:0},    // which parameter the chain rule is written for
  selN: {l:0, j:0},               // whose stations the strip below the net shows
  hover: null,                    // which station is under the cursor
  chip: "",                       // which chain-rule factor is hovered in the ledger
  drag: null,                     // a weight being dragged on its own wire
  trails: true, spd: 1,
  steps: 0,
  hist: [],                       // batch loss, one per update
  track: [],                      // (w, b) at each update — where every prediction has been
  playing: false,
};

/* Seven stations for one neuron: the backward half is broken up exactly the way
   the forward half is, one multiplication per step. A net gets the same
   treatment, two stations per layer on the way back — which is how you watch the
   chain rule grow a factor for every layer you add. */
const ONE_PHASES = [
  {id:"z",  grp:"forward",  lbl:"weighted sum", eq:() => "z = Σ wᵢ·xᵢ + b"},
  {id:"a",  grp:"forward",  lbl:"activation",   eq:() => "a = " + ACT[S.act].call},
  {id:"L",  grp:"forward",  lbl:"loss",         eq:() => "L = (a − t)²"},
  {id:"ga", grp:"backward", lbl:"loss slope",   eq:() => "∂L/∂a = 2(a − t)"},
  {id:"gz", grp:"backward", lbl:"through f",    eq:() => "∂L/∂z = ∂L/∂a · ∂a/∂z"},
  {id:"gw", grp:"backward", lbl:"onto w and b", eq:() => "∂L/∂w = ∂L/∂z · x"},
  {id:"u",  grp:"step",     lbl:"update",       eq:() => "w ← w − η · ∂L/∂w"},
];
let PHASES = ONE_PHASES;
let PH = {z:0, a:1, L:2, ga:3, gz:4, gw:5, u:6};

const aName = l => l < 0 ? "x" : "a" + SUP[l];
function setPhases(){
  if(!isNet()) PHASES = ONE_PHASES;
  else {
    const nl = NLAY(), P = [];
    for(let l=0;l<nl;l++)
      P.push({id:"f" + l, l, grp:"forward", lbl:`layer ${l + 1}`,
              eq:() => `${aName(l)} = f( W${SUP[l]}·${aName(l-1)} + b${SUP[l]} )`});
    P.push({id:"L",  grp:"forward",  lbl:"loss",  eq:() => "L = (a − t)²"});
    P.push({id:"ga", grp:"backward", lbl:"∂L/∂a", eq:() => `∂L/∂${aName(nl-1)} = 2(a − t)`});
    for(let l=nl-1;l>=0;l--){
      P.push({id:"d" + l, l, grp:"backward", lbl:`δ${SUP[l]}`,
              eq:() => `δ${SUP[l]} = ∂L/∂${aName(l)} ⊙ f′(z${SUP[l]})`});
      P.push({id:"g" + l, l, grp:"backward", lbl:`∂L/∂W${SUP[l]}`,
              eq:() => `∂L/∂W${SUP[l]} = δ${SUP[l]}·${aName(l-1)}ᵀ` +
                       (l ? `,  ∂L/∂${aName(l-1)} = W${SUP[l]}ᵀ·δ${SUP[l]}` : "")});
    }
    P.push({id:"u", grp:"step", lbl:"update", eq:() => "W ← W − η · ∂L/∂W"});
    PHASES = P;
  }
  PH = {};
  PHASES.forEach((p, i) => PH[p.id] = i);
  S.phase = Math.min(S.phase, PHASES.length - 1);
}

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

/* the same forward pass, layer by layer. Every weight of every layer gets its own
   node, so the gradient the page draws on a wire is the one the engine produced
   for exactly that wire — nothing here is a formula re-derived for the picture. */
function netGraph(d){
  const sh = shape(), layers = [];
  let a = d.x.slice(0, sh[0]).map(v => V(v));
  const a0 = a;
  for(let l=0;l<sh.length-1;l++){
    const W = S.W[l].map(row => row.map(v => V(v)));
    const B = S.B[l].map(v => V(v));
    const A = layerAct(l), z = [], prod = [];
    for(let j=0;j<W.length;j++){
      const pr = W[j].map((w, i) => mul(w, a[i]));
      let s = pr[0];
      for(let i=1;i<pr.length;i++) s = add(s, pr[i]);
      z.push(add(s, B[j]));
      prod.push(pr);
    }
    const out = z.map(zz => A.f(zz));
    layers.push({W, B, z, a:out, prod, in:a, A});
    a = out;
  }
  const t = V(d.t), e = sub(a[0], t), L = mul(e, e);
  backward(L);
  return {a0, layers, out:a[0], t, e, L, d};
}

// the mean of the per-point gradients, in the shape of the parameters themselves
let GW = [], GB = [];
function compute(){
  GS = S.data.map(isNet() ? netGraph : graph);
  const n = GS.length || 1;
  LOSS = GS.reduce((s, g) => s + g.L.d, 0)/n;
  // L = mean of the per-point losses, so dL/dw is the mean of the per-point gradients
  if(isNet()){
    GW = S.W.map((Wl, l) => Wl.map((row, j) => row.map((_, i) =>
           GS.reduce((s, g) => s + g.layers[l].W[j][i].g, 0)/n)));
    GB = S.B.map((Bl, l) => Bl.map((_, j) =>
           GS.reduce((s, g) => s + g.layers[l].B[j].g, 0)/n));
  } else {
    GRAD = {w: Array.from({length:S.nin}, (_, i) => GS.reduce((s, g) => s + g.w[i].g, 0)/n),
            b: GS.reduce((s, g) => s + g.b.g, 0)/n};
  }
  S.sample = Math.min(S.sample, GS.length - 1);
  G = GS[S.sample];
}

/* the forward pass with no graph at all — the heat map calls it thousands of
   times per repaint, so it has to be plain arithmetic */
function predict(x){
  if(!isNet()) return ACT[S.act].raw(S.w[0]*x[0] + (S.nin > 1 ? S.w[1]*x[1] : 0) + S.b);
  let a = x.slice(0, 2);
  for(let l=0;l<S.W.length;l++){
    const A = layerAct(l), out = [];
    for(let j=0;j<S.W[l].length;j++){
      let s = S.B[l][j];
      for(let i=0;i<a.length;i++) s += S.W[l][j][i]*a[i];
      out.push(A.raw(s));
    }
    a = out;
  }
  return a[0];
}

// the finite-difference check that used to sit in a Diagnostics panel now lives
// in verify/backprop.py, which runs it over 1331 configurations instead of one
const diverged = () => !isFinite(LOSS) || LOSS > 1e8;

function applyUpdate(){
  if(diverged()) return;
  S.hist.push(LOSS);                          // the loss this step was taken at
  if(isNet()){
    for(let l=0;l<S.W.length;l++){
      for(let j=0;j<S.W[l].length;j++){
        for(let i=0;i<S.W[l][j].length;i++) S.W[l][j][i] -= S.lr*GW[l][j][i];
        S.B[l][j] -= S.lr*GB[l][j];
      }
    }
  } else {
    S.track.push({w:S.w.slice(), b:S.b});
    for(let i=0;i<S.nin;i++) S.w[i] -= S.lr*GRAD.w[i];
    S.b -= S.lr*GRAD.b;
  }
  if(S.hist.length > 1200){ S.hist.shift(); if(S.track.length) S.track.shift(); }
  S.steps++;
  compute();
}

function train(n){
  for(let k=0;k<n && !diverged();k++) applyUpdate();
  S.phase = Math.max(S.phase, 2);
  refresh();
}

const clone = a => a.map(r => Array.isArray(r) ? clone(r) : r);

// PyTorch initialises nn.Linear uniformly on ±1/√fan_in, per layer
function initNet(rnd){
  const sh = shape();
  S.W = []; S.B = [];
  for(let l=0;l<sh.length-1;l++){
    const k = 1/Math.sqrt(sh[l]);
    S.W.push(Array.from({length:sh[l+1]}, () =>
             Array.from({length:sh[l]}, () => r2((rnd()*2 - 1)*k))));
    S.B.push(Array.from({length:sh[l+1]}, () => 0));
  }
  S.W0 = clone(S.W); S.B0 = clone(S.B);
}
// the page has to open the same way every time, so the default init is seeded
const seeded = () => initNet(prng(12345 + S.width*7 + NETS[S.net].hidden*131));

// a net opens on the first hidden layer: it is where the forward pass starts,
// where the recursion lives, and where the chain rule is at its longest
function selDefault(){
  const nl = NLAY();
  S.selN = {l:0, j:0};
  S.sel  = isNet() ? {k:"w", l:0, j:0, i:0} : {k:"w", i:0, l:0, j:0};
}

function loadSet(key){
  const D = SETS[key];
  S.set = key;
  S.data = D.make();
  S.act = (isNet() ? D.netact : D.act) || D.act;
  S.lr  = (isNet() ? D.netlr  : D.lr)  || D.lr;
  if(isNet()){ S.nin = 2; seeded(); }
  else {
    S.nin = D.nin;
    S.w = [D.w[0], D.w[1] || 0]; S.b = D.b;
    S.w0 = S.w.slice(); S.b0 = S.b;
  }
  // open on the point furthest from the origin: at x = 0 every product is 0 and
  // the whole diagram fades to nothing, which is a poor first impression
  S.sample = S.data.reduce((best, d, i, a) =>
    Math.hypot(...d.x.slice(0, S.nin)) > Math.hypot(...a[best].x.slice(0, S.nin)) ? i : best, 0);
  selDefault();
  setPhases();
  S.steps = 0; S.hist = []; S.track = []; S.phase = 0;
}

// switching architecture keeps the dataset when it still makes sense — the two
// clusters are worth seeing with and without a hidden layer
function loadNet(key){
  S.net = key;
  const ok = k => SETS[k].for === "both" || SETS[k].for === (isNet() ? "net" : "one");
  loadSet(ok(S.set) ? S.set : (isNet() ? "xor" : "line"));
}

function reset(){
  if(isNet()){ S.W = clone(S.W0); S.B = clone(S.B0); }
  else { S.w = S.w0.slice(); S.b = S.b0; }
  S.steps = 0; S.hist = []; S.track = []; S.phase = 0;
  refresh();
}

function randomise(){
  if(isNet()) initNet(Math.random);
  else {
    const k = 1/Math.sqrt(S.nin);
    S.w = S.w.map(() => r2((Math.random()*2 - 1)*k));
    S.b = r2((Math.random()*2 - 1)*k);
    S.w0 = S.w.slice(); S.b0 = S.b;
  }
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
  for(const g of GS)
    if(isNet()) for(const ly of g.layers) for(const z of ly.z) ZMAX = Math.max(ZMAX, Math.abs(z.d));
    else ZMAX = Math.max(ZMAX, Math.abs(g.z.d));
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

// w, h are drawing units; css is the width to display them at. Below the width
// the diagram needs, the whole thing scales down rather than growing a
// scrollbar — and the hit tests keep working, since they convert through
// the element's real size either way.
function fitCanvas(cv, w, h, css){
  const dpr = devicePixelRatio || 1;
  const W = Math.round(w*dpr), H = Math.round(h*dpr);
  const cw = css || w;
  if(cv.width !== W || cv.height !== H || parseFloat(cv.style.width) !== cw){
    cv.width = W; cv.height = H;                // only resize when it changed — this repaints per frame
    cv.style.width = cw + "px"; cv.style.height = Math.round(h*cw/w) + "px";
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
  const room = (el("vizwrap").clientWidth || 900) - 2;
  const avail = Math.max(800, Math.min(1240, room));
  // the wire run, then the gaps between cards. The wires are the one part that
  // can give ground when the column is tight, so they do it first
  const lead = avail < 900 ? 84 : 124, gap = 40;
  const sx = IN_X + IN_R + lead;
  const extra = Math.max(0, avail - 12 - sx - 2*gap - (CW.sum + CW.act + CW.loss));
  const sum = CW.sum + extra*0.42, act = CW.act + extra*0.29, loss = CW.loss + extra*0.29;
  const ax = sx + sum + gap, lx = ax + act + gap;
  const grows = S.nin === 1 ? 3 : 4;                     // ∂L/∂w… , ∂L/∂b , ∂L/∂x
  const bh = 24 + grows*ROWH + 8;
  const uy = BY + bh + 16;
  return {w:avail, css: Math.min(room, avail), h: uy + UH + 10,
          gap, sum, act, loss, sx, ax, lx, right: lx + loss, bh, uy};
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
  if(isNet()) return paintNet();
  const P = palette();
  LY = layout();
  const g = fitCanvas(el("viz"), LY.w, LY.h, LY.css);
  g.clearRect(0, 0, LY.w, LY.h);

  const A = ACT[S.act];
  const live = liveSet();
  const al = (...parts) => (live && !parts.some(p => live.has(p))) ? 0.15 : 1;
  const p = S.phase, ph = (frame % 120)/120;
  const on = k => p >= PH[k] && !diverged();      // has the pass reached this station
  const showG = on("ga");

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
  label(g, P, LY.sx + 10, FY + 14, "1)  z = Σ wᵢ·xᵢ + b");
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
  card(g, P, LY.ax, FY, LY.act, FH, al("act"), p >= PH.a ? ramp(G.a.d, 1.2, P, 0.02, 0.18) : null);
  g.globalAlpha = al("act");
  label(g, P, LY.ax + 10, FY + 14, `2)  a = ${A.call}`);
  if(p >= PH.a){
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

  card(g, P, LY.lx, FY, LY.loss, FH, al("loss"), p >= PH.L ? ramp(G.L.d, 1, P, 0.02, 0.2) : null);
  g.globalAlpha = al("loss");
  label(g, P, LY.lx + 10, FY + 14, "3)  L = (a − t)²");
  if(p >= PH.L){
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

  // one station per step, each holding its place as a dashed outline until the
  // pass gets there — the row never reflows underneath you
  if(on("ga")){
    // it starts at the loss: dL/dL is 1 by definition
    downArrow(g, P, tcx, FY + FH, BY, al("loss", "ga"), "∂L/∂L = 1", ph);
    gradBox(g, P, boxes.a, "4)  ∂L/∂a", "2(a − t)",
            `2( ${fmt(G.a.d)} − ${par(fmt(G.d.t))} )`, G.a.g, al("ga"));
  } else ghost(g, P, boxes.a.x, BY, boxes.a.w, LY.bh, "4)  ∂L/∂a");

  if(on("gz")){
    gradBox(g, P, boxes.z, "5)  ∂L/∂z", "∂L/∂a · ∂a/∂z",
            `${fmtg(G.a.g)} · ${par(fmtg(G.a.local))}`, G.z.g, al("gz"));
    backArrow(g, P, boxes.a.x, boxes.z.x + boxes.z.w, BY + LY.bh/2, al("ga", "gz"), ph);
  } else ghost(g, P, boxes.z.x, BY, boxes.z.w, LY.bh, "5)  ∂L/∂z");

  if(on("gw")){
    paramBox(g, P, boxes.p, al);
    backArrow(g, P, boxes.z.x, boxes.p.x + boxes.p.w, BY + LY.bh/2, al("gz", "gp"), ph);
  } else ghost(g, P, boxes.p.x, BY, boxes.p.w, LY.bh, "6)  ∂L/∂w, ∂L/∂b");

  if(showG){
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
      if(on("gz") && live.has("gz") && live.size === 2) drop(LY.ax + LY.act/2);
      if(live.has("ga") && live.size === 2) drop(LY.lx + 26);
      if(on("gw")){
        for(let i=0;i<S.nin;i++) if(live.has("gw" + i)) elbow(inY(i), rowMid(i));
        if(live.has("gb")) elbow(biasY(), rowMid(S.nin));
      }
      g.setLineDash([]); g.globalAlpha = 1;
    }
  }

  /* ---- the update, using the mean over every point ---- */
  if(on("u"))            updateRow(g, P, LY.uy, al("upd"));
  else if(p >= PH.u)     ghost(g, P, 8, LY.uy, LY.right - 8, UH, "diverged — nothing to apply");
  else                   ghost(g, P, 8, LY.uy, LY.right - 8, UH, "7)  w ← w − η · ∂L/∂w");
}

/* ======================= the network, drawn once =======================
   One picture, both directions. Every wire carries its weight in the forward
   lane and, once the backward pass has reached that layer, ∂L/∂w in a violet
   lane beside it running the other way. Every node holds its activation, with
   the gradient that came back to it in a pill directly underneath — the same
   "gradients sit under the values they came from" as the single neuron, folded
   into one drawing because a net has too many stations to lay out twice.

   Underneath, the four station cards of the single-neuron page, pointed at
   whichever neuron you select: an MLP is that neuron, many times over.
   ======================================================================= */
const NR = 15, ROWSP = 58, SGAP = 12;

function netLayout(){
  const room = (el("vizwrap").clientWidth || 900) - 2;
  const avail = Math.max(800, Math.min(1240, room));
  const sh = shape(), rows = Math.max(...sh);
  const lossW = 172, lossGap = 46;
  const x0 = 74, xn = avail - lossW - lossGap - 10;
  const NY = 40;
  const cy = NY + 14 + NR + (rows - 1)*ROWSP/2;
  const bandH = 14 + 2*NR + (rows - 1)*ROWSP + 30;
  const sy = NY + bandH + 24;
  const strip = 136;
  const uy = sy + strip + 16;
  // the strip: sum, activation, δ, then the weight gradients
  const inner = avail - 16 - 3*SGAP;
  const cw = [0.31, 0.20, 0.21, 0.28].map(f => Math.round(inner*f));
  const cx = [8];
  for(let i=1;i<4;i++) cx.push(cx[i-1] + cw[i-1] + SGAP);
  return {w:avail, css:Math.min(room, avail), h: uy + UH + 12,
          x0, dx:(xn - x0)/(sh.length - 1), NY, bandH, cy, sy, strip, uy,
          lossX: xn + lossGap, lossW, cx, cw, right: avail - 8};
}
const nodeX = l => LY.x0 + l*LY.dx;
const nodeY = (l, j) => LY.cy + (j - (shape()[l] - 1)/2)*ROWSP;

// column l holds the activations layer l−1 produced; column 0 holds the inputs
const colVal  = (l, j) => l ? G.layers[l-1].a[j] : G.a0[j];
const colDelta = (l, j) => l ? G.layers[l-1].z[j] : null;

function paintNet(){
  const P = palette();
  LY = netLayout();
  const g = fitCanvas(el("viz"), LY.w, LY.h, LY.css);
  g.clearRect(0, 0, LY.w, LY.h);

  const sh = shape(), nl = NLAY(), p = S.phase, ph = (frame % 120)/120;
  const fwd = l => p >= PH["f" + l];                 // layer l has produced its values
  const grd = l => p >= PH["g" + l] && !diverged();  // its weight gradients exist
  const dlt = l => p >= PH["d" + l] && !diverged();  // its δ exists

  g.lineJoin = "round"; g.textBaseline = "middle";
  bandLabel(g, P, `the network — ${sh.join(" → ")}, ${nparams()} parameters, `
                + `${S.data.length} points through it`, 8, 20);

  /* ---- wires: the weight one way, its gradient the other ---- */
  for(let l=0;l<nl;l++)
    for(let j=0;j<sh[l+1];j++)
      for(let i=0;i<sh[l];i++) netWire(g, P, l, j, i, grd(l), ph);

  /* ---- nodes, with the gradient that reached them underneath ---- */
  for(let l=0;l<=nl;l++)
    for(let j=0;j<sh[l];j++)
      netNode(g, P, l, j, l === 0 || fwd(l - 1), l === 0 ? grd(0) : dlt(l - 1));

  /* ---- the loss, at the end of the line ---- */
  const out = G.out, lx = LY.lx = LY.lossX, ly = LY.cy - 62;
  const lit = p >= PH.L;
  card(g, P, lx, ly, LY.lossW, 124, 1, lit ? ramp(G.L.d, 1, P, 0.02, 0.2) : null);
  g.textAlign = "left"; g.font = "9.5px ui-monospace,Menlo,monospace"; g.fillStyle = P.dim;
  g.fillText(`${PH.L + 1})  L = (a − t)²`, lx + 10, ly + 14);
  if(lit){
    sparkLoss(g, P, lx + 12, ly + 22, LY.lossW - 24, 58, out.d, G.d.t);
    g.textAlign = "left"; g.font = "700 13px ui-monospace,Menlo,monospace"; g.fillStyle = P.text;
    g.fillText("L = " + fmt(G.L.d, 4), lx + 12, ly + 96);
    g.font = "10px ui-monospace,Menlo,monospace"; g.fillStyle = P.dim;
    g.fillText(`target ${fmt(G.d.t)} · batch ${fmt(LOSS, 4)}`, lx + 12, ly + 112);
  } else waiting(g, P, lx, ly, LY.lossW, 124, "L = ?");
  arrow(g, P, nodeX(nl) + NR, LY.cy, lx, LY.cy, 1, "a");
  if(p >= PH.ga && !diverged())
    backArrow(g, P, lx - 6, nodeX(nl) + NR + 4, LY.cy + 16, 1, ph);

  /* ---- the four stations, for the neuron you picked ---- */
  netStrip(g, P, ph);

  /* ---- the step ---- */
  if(p >= PH.u && !diverged()) netUpdate(g, P, LY.uy);
  else ghost(g, P, 8, LY.uy, LY.right - 8, UH,
             p >= PH.u ? "diverged — nothing to apply"
                       : `${PH.u + 1})  W ← W − η · ∂L/∂W  —  all ${nparams()} at once`);
}

/* one wire, drawn twice: the weight going right, its gradient coming back */
function netWire(g, P, l, j, i, showGrad, ph){
  const x0 = nodeX(l) + NR, y0 = nodeY(l, i);
  const x1 = nodeX(l + 1) - NR, y1 = nodeY(l + 1, j);
  const w = S.W[l][j][i];
  const sel = S.sel.k === "w" && S.sel.l === l && S.sel.j === j && S.sel.i === i;
  const hot = S.hover === `w${l}_${j}_${i}`;
  const t = Math.min(1, Math.abs(w)/WMAX);
  const rgb = w >= 0 ? P.pos : P.neg;

  if(sel || hot){
    g.strokeStyle = P.accent; g.lineWidth = 3 + 3.4*t;
    seg(g, x0, y0, x1, y1);
  }
  g.strokeStyle = `rgba(${rgb[0]},${rgb[1]},${rgb[2]},${0.22 + 0.7*t})`;
  g.lineWidth = 0.8 + 3.4*t;
  seg(g, x0, y0, x1, y1);

  if(!showGrad) return;
  // the return lane, offset to the side so the two are never the same line
  const gr = G.layers[l].W[j][i].g;
  const dx = x1 - x0, dy = y1 - y0, len = Math.hypot(dx, dy) || 1;
  const ox = -dy/len*5, oy = dx/len*5;
  const gt = Math.min(1, Math.abs(gr)/GMAX);
  g.strokeStyle = gramp(gr, P, 0.10, 0.85); g.lineWidth = 0.8 + 2.8*gt;
  seg(g, x0 + ox, y0 + oy, x1 + ox, y1 + oy);
  pulse(g, P, x1 + ox + (x0 - x1)*ph, y1 + oy + (y0 - y1)*ph, ph);
}

function netNode(g, P, l, j, lit, gradLit){
  const x = nodeX(l), y = nodeY(l, j), v = colVal(l, j);
  const sel = S.selN.l === l - 1 && S.selN.j === j;
  g.globalAlpha = 1;
  g.fillStyle = P.frame;
  g.beginPath(); g.arc(x, y, NR, 0, 6.2832); g.fill();
  if(lit){ g.fillStyle = ramp(v.d, l ? 1.1 : XMAX, P, 0.05, 0.8);
           g.beginPath(); g.arc(x, y, NR, 0, 6.2832); g.fill(); }
  g.strokeStyle = sel ? P.accent : P.frameLine;
  g.lineWidth = sel ? 2.4 : 1.2;
  g.beginPath(); g.arc(x, y, NR, 0, 6.2832); g.stroke();

  g.textAlign = "center";
  g.font = "700 11px ui-monospace,Menlo,monospace"; g.fillStyle = P.text;
  g.fillText(lit ? fmt(v.d) : "?", x, y);

  // the gradient that came back to this node, directly underneath it
  if(gradLit){
    const gr = l ? colDelta(l, j).g : v.g;
    g.fillStyle = gramp(gr, P, 0.06, 0.5);
    roundRect(g, x - 25, y + NR + 3, 50, 15, 4); g.fill();
    g.font = "600 9.5px ui-monospace,Menlo,monospace"; g.fillStyle = P.text;
    g.fillText(fmtg(gr), x, y + NR + 10.5);
  }
  if(l === 0 || j === 0){
    g.font = "600 9.5px ui-sans-serif,system-ui"; g.fillStyle = P.dim2;
    g.fillText(l === 0 ? "x" : aName(l - 1), x, LY.NY + 4);
  }
}

/* ---------- the four stations, for the selected neuron ---------- */
function netStrip(g, P, ph){
  const l = S.selN.l, j = S.selN.j, ly = G.layers[l], A = ly.A;
  const p = S.phase, sh = shape();
  const z = ly.z[j], a = ly.a[j], nin = sh[l];
  const name = `${aName(l)}${SUB[j]}`;
  const seen = p >= PH["f" + l], dSeen = p >= PH["d" + l] && !diverged();
  const gSeen = p >= PH["g" + l] && !diverged();

  bandLabel(g, P, `neuron ${name} — the same four stations as one neuron on its own`, 8, LY.sy - 12);

  /* 1 — the weighted sum */
  const b0 = {x:LY.cx[0], w:LY.cw[0]};
  const rt = LY.sy + 24, rh = Math.min(ROWH, (LY.strip - 62)/(nin + 1));
  card(g, P, b0.x, LY.sy, b0.w, LY.strip, 1);
  label(g, P, b0.x + 10, LY.sy + 13,
        `${PH["f" + l] + 1})  z${SUP[l]}${SUB[j]} = Σ W${SUP[l]}${SUB[j]}ᵢ·${aName(l-1)}ᵢ + b`);
  if(seen){
    for(let i=0;i<nin;i++)
      termRow(g, P, b0.x, b0.w, rt + i*rh + rh/2,
              `w${SUB[j]}${SUB[i]}·${aName(l-1)}${SUB[i]}`,
              `${fmt(S.W[l][j][i])} · ${fmt(ly.in[i].d)}`, ly.prod[j][i].d, 1, rh);
    termRow(g, P, b0.x, b0.w, rt + nin*rh + rh/2, "b", "", S.B[l][j], 1, rh);
    g.strokeStyle = P.frameLine; g.lineWidth = 1;
    const rule = rt + (nin + 1)*rh + 4;
    seg(g, b0.x + 10, rule, b0.x + b0.w - 10, rule);
    g.fillStyle = ramp(z.d, ZMAX, P, 0.08, 0.55);
    roundRect(g, b0.x + 8, rule + 5, b0.w - 16, 21, 6); g.fill();
    g.textAlign = "left"; g.font = "700 13px ui-monospace,Menlo,monospace"; g.fillStyle = P.text;
    g.fillText(`z = ${fmt(z.d)}`, b0.x + 15, rule + 15.5);
  } else waiting(g, P, b0.x, LY.sy, b0.w, LY.strip, "z = ?");

  /* 2 — through the activation */
  const b1 = {x:LY.cx[1], w:LY.cw[1]};
  card(g, P, b1.x, LY.sy, b1.w, LY.strip, 1, seen ? ramp(a.d, 1.2, P, 0.02, 0.18) : null);
  label(g, P, b1.x + 10, LY.sy + 13, `${aName(l)}${SUB[j]} = ${A.call}`);
  if(seen){
    sparkAct(g, P, b1.x + 12, LY.sy + 22, b1.w - 24, 62, A, z.d, a.d, a.local);
    g.textAlign = "left"; g.font = "700 13px ui-monospace,Menlo,monospace"; g.fillStyle = P.text;
    g.fillText("a = " + fmt(a.d), b1.x + 12, LY.sy + 98);
    g.font = "10px ui-monospace,Menlo,monospace"; g.fillStyle = P.gradCss;
    g.fillText(`f′(z) = ${fmtg(a.local)}`, b1.x + 12, LY.sy + 116);
  } else waiting(g, P, b1.x, LY.sy, b1.w, LY.strip, "a = ?");

  /* 3 — δ, the one new equation a layer brings */
  const b2 = {x:LY.cx[2], w:LY.cw[2]};
  const last = l === NLAY() - 1;
  if(dSeen){
    card(g, P, b2.x, LY.sy, b2.w, LY.strip, 1, gramp(z.g, P, 0.02, 0.22));
    g.textAlign = "left"; g.font = "600 10.5px ui-monospace,Menlo,monospace"; g.fillStyle = P.gradCss;
    g.fillText(`δ${SUP[l]}${SUB[j]} = ∂L/∂${name} · f′(z)`, b2.x + 12, LY.sy + 15);
    g.font = "10px ui-monospace,Menlo,monospace"; g.fillStyle = P.dim;
    g.fillText(last ? `∂L/∂a = 2( ${fmt(a.d)} − ${par(fmt(G.d.t))} )`
                    : shape()[l+2] > 1 ? `∂L/∂a = Σ over the ${shape()[l+2]} paths above`
                                       : `∂L/∂a = the one path above`,
               b2.x + 12, LY.sy + 36);
    g.fillStyle = P.text; g.font = "600 11px ui-monospace,Menlo,monospace";
    g.fillText(`= ${fmtg(a.g)}`, b2.x + 12, LY.sy + 54);
    g.font = "10px ui-monospace,Menlo,monospace"; g.fillStyle = P.dim;
    g.fillText(`${fmtg(a.g)} · ${par(fmtg(a.local))}`, b2.x + 12, LY.sy + 78);
    g.font = "700 17px ui-monospace,Menlo,monospace"; g.fillStyle = P.text;
    g.fillText("δ = " + fmtg(z.g), b2.x + 12, LY.sy + 104);
  } else ghost(g, P, b2.x, LY.sy, b2.w, LY.strip, `δ${SUP[l]}${SUB[j]}`);

  /* 4 — and onto the weights that feed it */
  const b3 = {x:LY.cx[3], w:LY.cw[3]};
  if(gSeen){
    card(g, P, b3.x, LY.sy, b3.w, LY.strip, 1);
    g.textAlign = "left"; g.font = "600 10.5px ui-monospace,Menlo,monospace"; g.fillStyle = P.gradCss;
    g.fillText(`∂L/∂w${SUB[j]}ᵢ = δ${SUP[l]}${SUB[j]} · ${aName(l-1)}ᵢ`, b3.x + 12, LY.sy + 15);
    const gy = LY.sy + 26;
    for(let i=0;i<nin;i++)
      gradRow(g, P, b3, gy + i*rh + rh/2, `∂L/∂w${SUB[j]}${SUB[i]}`,
              `${fmtg(z.g)} · ${par(fmt(ly.in[i].d))}`, ly.W[j][i].g, rh,
              S.sel.k === "w" && S.sel.l === l && S.sel.j === j && S.sel.i === i);
    gradRow(g, P, b3, gy + nin*rh + rh/2, `∂L/∂b${SUB[j]}`, `${fmtg(z.g)} · 1`, ly.B[j].g, rh,
            S.sel.k === "b" && S.sel.l === l && S.sel.j === j);
  } else ghost(g, P, b3.x, LY.sy, b3.w, LY.strip, `∂L/∂W${SUP[l]}`);
}

function gradRow(g, P, box, y, lhs, mid, val, rh, sel){
  g.fillStyle = gramp(val, P, 0.03, 0.34);
  roundRect(g, box.x + 8, y - rh/2 + 1, box.w - 16, rh - 3, 5); g.fill();
  if(sel){ g.strokeStyle = P.accent; g.lineWidth = 1.4;
           roundRect(g, box.x + 8, y - rh/2 + 1, box.w - 16, rh - 3, 5); g.stroke(); }
  g.textAlign = "left"; g.font = "600 10px ui-monospace,Menlo,monospace"; g.fillStyle = P.gradCss;
  g.fillText(lhs, box.x + 14, y);
  g.font = "9.5px ui-monospace,Menlo,monospace"; g.fillStyle = P.dim2;
  g.fillText(mid, box.x + 74, y);
  g.textAlign = "right"; g.font = "700 11px ui-monospace,Menlo,monospace"; g.fillStyle = P.text;
  g.fillText(fmtg(val), box.x + box.w - 14, y);
}

// the step, for a net: every parameter moves, so the row shows the one you have
// selected doing it and the size of the move the rest are making
function netUpdate(g, P, y){
  const s = S.sel, cur = s.k === "b" ? S.B[s.l][s.j] : S.W[s.l][s.j][s.i];
  const gr  = s.k === "b" ? GB[s.l][s.j] : GW[s.l][s.j][s.i];
  const nm  = s.k === "b" ? `b${SUP[s.l]}${SUB[s.j]}` : `w${SUP[s.l]}${SUB[s.j]}${SUB[s.i]}`;
  g.fillStyle = P.frame; roundRect(g, 8, y, LY.right - 8, UH, 9); g.fill();
  g.strokeStyle = P.frameLine; g.lineWidth = 1.2; roundRect(g, 8, y, LY.right - 8, UH, 9); g.stroke();
  g.textAlign = "left"; g.font = "9.5px ui-monospace,Menlo,monospace"; g.fillStyle = P.dim;
  g.fillText(`${PH.u + 1})  W ← W − η·∂L/∂W   with η = ${fmt(S.lr)}, every one of the ${nparams()} `
           + `parameters at once, gradients averaged over all ${S.data.length} points`, 18, y + 11);
  const txt = `${nm} ← ${fmt(cur)} − ${fmt(S.lr)}·${par(fmtg(gr))} = ${fmt(cur - S.lr*gr)}`;
  g.font = "700 11px ui-monospace,Menlo,monospace";
  const w = g.measureText(txt).width + 16;
  g.fillStyle = ramp(cur - S.lr*gr, WMAX, P, 0.05, 0.38);
  roundRect(g, 18, y + 17, w, 17, 5); g.fill();
  g.fillStyle = P.text; g.fillText(txt, 26, y + 26);
  g.font = "10px ui-monospace,Menlo,monospace"; g.fillStyle = P.dim2;
  g.fillText(`— and the same for the other ${nparams() - 1}`, 18 + w + 10, y + 26);
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
function termRow(g, P, x, w, y, lhs, mid, val, alpha, rh){
  rh = rh || ROWH;
  g.globalAlpha = alpha;
  g.fillStyle = ramp(val, ZMAX, P, 0.04, 0.34);
  roundRect(g, x + 8, y - rh/2 + 1, w - 16, rh - 3, 5); g.fill();
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
  g.fillText("6)  ∂L/∂z spreads onto every parameter", box.x + 12, BY + 16);
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
  g.fillText(`7)  w ← w − η·∂L/∂w   with η = ${fmt(S.lr)} and the gradient averaged over all ${S.data.length} points`,
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

function sparkAct(g, P, x, y, w, h, A, z, a, local){
  A = A || ACT[S.act];
  if(z === undefined){ z = G.z.d; a = G.a.d; local = G.a.local; }
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
  tangent(g, P, sx, sy, z, a, local, (xr[1] - xr[0])*0.22);
  g.setLineDash([2,3]); g.strokeStyle = P.dim2; g.lineWidth = 1;
  seg(g, sx(z), y + h, sx(z), sy(a));
  g.setLineDash([]);
  dot(g, P, sx(z), sy(a));
  g.restore();
  g.textAlign = "center"; g.font = "9px ui-monospace,Menlo,monospace"; g.fillStyle = P.dim2;
  g.fillText("z", Math.max(x + 6, Math.min(x + w - 6, sx(z))), y + h - 6);
}

function sparkLoss(g, P, x, y, w, h, a, t){
  if(a === undefined){ a = G.a.d; t = G.d.t; }
  const e = a - t;
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
   Three marks and nothing else: a dot per point, one curve for the neuron, and
   a hairline joining each dot to the curve — that hairline is the point's
   error, so watching them shorten *is* watching the loss fall. History is one
   dashed line where the curve started, and a tail on the selected point only.
   ================================================================== */
const PADL = 32, PADR = 10, PADT = 18, PADB = 22;
const AH = 196, BH = 128;                      // the fit, then the loss under it
const FH2 = PADT + AH + PADB + 14 + PADT + BH + PADB;

function fitWidth(){ return Math.max(228, (el("fitwrap").clientWidth || 300) - 2); }

// the plot box, derived once and used by both the drawing and the hover test —
// computing it twice is how hit targets drift off the dots
function plotBox(){
  return {x:PADL, y:PADT, w: fitWidth() - PADL - PADR, h:AH};
}
function lossBox(){
  return {x:PADL, y: PADT + AH + PADB + 14 + PADT, w: fitWidth() - PADL - PADR, h:BH};
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

  (S.nin === 1 ? curvePanel : planePanel)(g, P, plotBox());
  lossPanel(g, P, lossBox());

  el("fitnote").innerHTML = notes().join("<br><br>");
}

/* nothing under the plots unless there is something to say — and a dead ReLU or
   a saturated sigmoid shows up here before it shows up anywhere else */
function notes(){
  if(diverged())
    return [`<span class="err">Diverged.</span> η = ${S.lr} takes steps wider than the bowl, so each one
             lands further out than the last. Lower η and reset.`];

  const out = [];
  if(isNet() && S.act === "linear")
    out.push(`<b>The hidden layers are linear, so the whole stack collapses.</b> W²(W¹x + b¹) + b² is just
              another matrix times x — depth buys nothing at all without a nonlinearity, and the boundary
              stays a straight line however many layers you add.`);
  const slopes = [];
  for(const g of GS)
    if(isNet()) for(const ly of g.layers) for(const a of ly.a) slopes.push(Math.abs(a.local));
    else slopes.push(Math.abs(g.a.local));
  const maxSlope = Math.max(0, ...slopes);
  if(maxSlope < 1e-3 && S.act !== "linear")
    out.push(S.act === "relu"
      ? `<b>Dead ReLU.</b> z is negative at every point, so the unit outputs 0 and blocks the gradient
         completely. The errors are at full intensity and nothing moves — press ▶ and watch it not learn.`
      : `<b>Vanishing gradient.</b> The activation is saturated at every point, so the violet tangent
         inside the neuron is flat. Every gradient is multiplied by that, and the weights barely move
         however wrong the fit is.`);
  else if(maxSlope < 0.05 && S.act !== "linear")
    out.push(`The steepest ∂a/∂z in the batch is only ${fmtg(maxSlope)}, so the weights get a fraction of
              whatever the loss asks for — these hairlines will shrink very slowly.`);
  else if(S.steps > 0 && LOSS < 1e-5)
    out.push(`Converged: L = ${LOSS.toExponential(1)}. The errors have faded to nothing and the gradients
              with them, so the updates have nothing left to do.`);

  return out;
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
                        `the fit — ${S.data.length} points`);

  g.save();
  g.beginPath(); g.rect(B.x, B.y, B.w, B.h); g.clip();

  const curve = (w, b, col, dash, width, alpha) => {
    g.strokeStyle = col; g.lineWidth = width; g.globalAlpha = alpha;
    if(dash) g.setLineDash([4,4]);
    g.beginPath();
    for(let j=0;j<=120;j++){
      const x = xr[0] + j/120*(xr[1] - xr[0]);
      j ? g.lineTo(sx(x), sy(f(w, b, x))) : g.moveTo(sx(x), sy(f(w, b, x)));
    }
    g.stroke();
    g.setLineDash([]); g.globalAlpha = 1;
  };

  // where the curve started, so the whole move is one glance
  if(S.track.length) curve(S.track[0].w[0], S.track[0].b, P.dim2, true, 1, 0.5);

  // one hairline per point, dot to curve: that is its error
  S.data.forEach((d, i) => {
    const y = f(S.w[0], S.b, d.x[0]);
    g.strokeStyle = ramp(y - d.t, 1, P, 0.25, 0.85);
    g.lineWidth = i === S.sample ? 3 : 1.4;
    seg(g, sx(d.x[0]), sy(d.t), sx(d.x[0]), sy(y));
  });

  curve(S.w[0], S.b, P.accent, false, 2, 1);

  // only the selected point keeps a tail — thirteen of them was a thicket
  if(S.trails && S.track.length > 1){
    const d = S.data[S.sample];
    const ys = history(Math.min(20, S.track.length)).map(t => sy(f(t.w[0], t.b, d.x[0])));
    ys.push(sy(f(S.w[0], S.b, d.x[0])));
    g.strokeStyle = P.accent; g.lineWidth = 1.4;
    for(let k=1;k<ys.length;k++){
      g.globalAlpha = 0.12 + 0.5*k/(ys.length - 1);
      seg(g, sx(d.x[0]), ys[k-1], sx(d.x[0]), ys[k]);
    }
    g.globalAlpha = 1;
  }

  // the points themselves — where each one should be
  S.data.forEach((d, i) => {
    const sel = i === S.sample;
    g.fillStyle = P.text;
    g.beginPath(); g.arc(sx(d.x[0]), sy(d.t), sel ? 4.5 : 3, 0, 6.2832); g.fill();
    if(sel){
      g.strokeStyle = P.accent; g.lineWidth = 1.6;
      g.beginPath(); g.arc(sx(d.x[0]), sy(d.t), 8.5, 0, 6.2832); g.stroke();
    }
  });
  g.restore();
}

// two inputs: what the model says over the whole plane. One neuron can only
// split it with a straight line; a hidden layer bends that line, so the same
// panel shows both — the heat map is the model, whatever the model is.
function planePanel(g, P, B){
  const xr = XR2, yr = XR2;
  const CELL = isNet() ? 5 : 6, nx = Math.ceil(B.w/CELL), ny = Math.ceil(B.h/CELL);
  const at = (i, j) => predict([xr[0] + (i + 0.5)/nx*(xr[1] - xr[0]),
                                yr[1] - (j + 0.5)/ny*(yr[1] - yr[0])]);
  const grid = [];
  g.save();
  g.beginPath(); g.rect(B.x, B.y, B.w, B.h); g.clip();       // the last row and column overhang
  for(let i=0;i<nx;i++){
    grid.push([]);
    for(let j=0;j<ny;j++){
      const a = at(i, j);
      grid[i].push(a);
      g.fillStyle = ramp(a - 0.5, 0.5, P, 0.02, 0.38);        // 0.5 is the boundary, so centre on it
      g.fillRect(B.x + i*CELL, B.y + j*CELL, CELL, CELL);
    }
  }
  // where the model crosses 0.5 — for a net the boundary is a curve, so it gets
  // traced out of the grid rather than solved for
  if(isNet()){
    g.fillStyle = P.accent;
    for(let i=0;i<nx-1;i++) for(let j=0;j<ny-1;j++){
      const c = grid[i][j] >= 0.5;
      if(c !== (grid[i+1][j] >= 0.5) || c !== (grid[i][j+1] >= 0.5))
        g.fillRect(B.x + i*CELL + CELL/2 - 1, B.y + j*CELL + CELL/2 - 1, 2.2, 2.2);
    }
  }
  g.restore();

  const {sx, sy} = axes(g, P, B, xr, yr, "x₁", "x₂",
                        `the boundary — ${S.data.length} points`);

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
  if(!isNet()){
    if(S.track.length) bound(S.track[0].w, S.track[0].b, P.dim2, true, 1);
    bound(S.w, S.b, P.accent, false, 2);
  }

  S.data.forEach((d, i) => {
    const p = plotXY(d);
    const r = predict(d.x) - d.t;
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
                        `the loss — ${S.steps} update${S.steps === 1 ? "" : "s"}`);

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

/* ======================= the sheet under the canvas ======================= */
const fv = v => `<span class="fv">${fmt(v)}</span>`;
const wv = v => `<span class="wv">${fmt(v)}</span>`;
const gv = v => `<span class="gv">${fmtg(v)}</span>`;
// ∂L/∂a must not be shouted into ∂L/∂A, nor δ into Δ — any label carrying a
// symbol keeps its own case
const mlbl = l => `<span class="mlbl${/[^\x00-\x7F]/.test(l) ? " raw" : ""}">${l}</span>`;
function mrow(phase, label, html){
  return `<div class="mrow${phase === S.phase ? " now" : ""}">${mlbl(label)}
          <span class="mval">${html}</span></div>`;
}
// the same row, but its body is an equation: symbols on the line, the numbers
// they currently hold hung underneath. The equation is what stays put.
function erow(phase, label, parts){
  return `<div class="mrow${phase === S.phase ? " now" : ""}">${mlbl(label)}
          <span class="eq">${parts.join("")}</span></div>`;
}
function tt(sym, num, o){
  o = o || {};
  const cls = ["tt", o.cls, o.chip && S.chip === o.chip ? "on" : ""].filter(Boolean).join(" ");
  return `<span class="${cls}"${o.chip ? ` data-chip="${o.chip}"` : ""}` +
         `${o.fill ? ` style="background:${o.fill}"` : ""}>` +
         `<b>${sym}</b><u${o.vc ? ` class="${o.vc}"` : ""}>${num == null ? "" : num}</u></span>`;
}
const op  = s => tt(s, "", {cls:"op"});
const res = v => `<span class="res">${v}</span>`;

// every pill carries the same colour its number has on the canvas
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

/* ==================== the sheet, for a net ====================
   A layer is a matrix times a vector, so the sheet draws matrices — real ones,
   with brackets, their shapes underneath and the page's colour ramp in the
   cells. Every equation is the same one the canvas above is drawing; here you
   see the whole layer at once instead of one neuron of it, and the shapes make
   the arithmetic obvious: (4×2)·(2×1) can only come out (4×1).

   The neuron you have selected is a row, so it lights up in every matrix it
   appears in — follow one row across and you have followed one neuron.
   ============================================================== */
function mat(name, rows, cols, get, o){
  o = o || {};
  const cells = [];
  for(let j=0;j<rows;j++)
    for(let i=0;i<cols;i++){
      const v = get(j, i);
      const hi = o.cell && o.cell.j === j && o.cell.i === i ? " hi"
               : o.row === j ? (o.grad ? " hi2" : " hi") : "";
      cells.push(`<span class="cell${hi}" style="background:${o.fill(v)}">${o.txt(v)}</span>`);
    }
  return `<span class="mat"><span class="mname">${name}</span>` +
         `<span class="mgrid" style="grid-template-columns:repeat(${cols},minmax(0,1fr))">` +
         cells.join("") + `</span><span class="mdim">${rows}×${cols}</span></span>`;
}
const mop = (s, sm) => `<span class="mop${sm ? " sm" : ""}">${s}</span>`;

function netMath(){
  const P = palette(), p = S.phase, out = [], sh = shape(), nl = NLAY();
  const l = S.selN.l, j = S.selN.j, sel = S.sel;
  const V_ = v => ramp(v, 1.2, P, 0.05, 0.42);          // an activation
  const Z_ = v => ramp(v, ZMAX, P, 0.05, 0.42);         // a pre-activation
  const W_ = v => ramp(v, WMAX, P, 0.05, 0.42);         // a parameter
  const X_ = v => ramp(v, XMAX, P, 0.05, 0.42);         // an input
  const D_ = v => gramp(v, P, 0.05, 0.42);              // anything backward
  const num = v => fmt(v);
  const grd = v => fmtg(v);
  const eq = (phase, label, parts) =>
    `<div class="mrow${phase === S.phase ? " now" : ""}">${mlbl(label)}
     <span class="meq">${parts.join("")}</span></div>`;
  // the row this neuron occupies, in whichever layer it lives
  const row = k => k === l ? j : undefined;

  out.push(mrow(-1, "point", `<b>${S.sample + 1} of ${S.data.length}</b>
    <span style="color:var(--dim2)">— hover a point on the boundary plot, or use ◀ ▶.
    Point at a wire or a neuron and it lights up in every matrix below.</span>`));

  /* ---- forward: one matrix product per layer ---- */
  for(let k=0;k<nl;k++){
    if(p < PH["f" + k]) break;
    const ly = G.layers[k], m = sh[k], n = sh[k+1];
    const inFill = k ? V_ : X_;
    out.push(eq(PH["f" + k], `${k + 1}) layer ${k + 1}`, [
      mat(`z${SUP[k]}`, n, 1, r => ly.z[r].d, {fill:Z_, txt:num, row:row(k)}),
      mop("="),
      mat(`W${SUP[k]}`, n, m, (r, c) => S.W[k][r][c],
          {fill:W_, txt:num, row:row(k),
           cell: sel.k === "w" && sel.l === k ? {j:sel.j, i:sel.i} : null}),
      mop("·"),
      mat(aName(k-1), m, 1, r => ly.in[r].d, {fill:inFill, txt:num}),
      mop("+"),
      mat(`b${SUP[k]}`, n, 1, r => S.B[k][r], {fill:W_, txt:num, row:row(k)}),
      mop(`→ ${ly.A.lbl} →`, true),
      mat(aName(k), n, 1, r => ly.a[r].d, {fill:V_, txt:num, row:row(k)}),
    ]));
  }

  if(p >= PH.L) out.push(erow(PH.L, `${PH.L + 1}) loss`, [
    tt("L", "", {cls:"hd"}), op("="), op("("),
    tt("a", fv(G.out.d), {fill:ramp(G.out.d, 1.2, P, 0.05, 0.3)}), op("−"),
    tt("t", fv(G.d.t), {fill:ramp(G.d.t, 1.2, P, 0.05, 0.3)}), op(")²"),
    op("="), res(fmt(G.L.d, 4)),
    op("&nbsp;&nbsp;"), tt(`batch L — the mean over all ${S.data.length}`,
                           `<b>${fmt(LOSS, 4)}</b>`, {cls:"op"}),
  ]));

  if(p >= PH.ga) out.push(erow(PH.ga, `${PH.ga + 1}) ∂L/∂a`, [
    tt(`∂L/∂${aName(nl-1)}`, "", {cls:"hd"}), op("="), op("2 ("),
    tt("a", fv(G.out.d)), op("−"), tt("t", fv(G.d.t)), op(")"),
    op("="), res(fmtg(G.out.g)),
  ]));

  /* ---- backward: the same products, transposed ---- */
  for(let k=nl-1;k>=0;k--){
    if(p < PH["d" + k]) break;
    const ly = G.layers[k], m = sh[k], n = sh[k+1];

    // δ = ∂L/∂a ⊙ f′(z), elementwise — one number per neuron, nothing more
    out.push(eq(PH["d" + k], `${PH["d" + k] + 1}) δ${SUP[k]}`, [
      mat(`δ${SUP[k]}`, n, 1, r => ly.z[r].g, {fill:D_, txt:grd, row:row(k), grad:1}),
      mop("="),
      mat(`∂L/∂${aName(k)}`, n, 1, r => ly.a[r].g, {fill:D_, txt:grd, row:row(k), grad:1}),
      mop("⊙"),
      mat(`f′(z${SUP[k]})`, n, 1, r => ly.a[r].local, {fill:D_, txt:grd, row:row(k), grad:1}),
      `<span class="mop sm">elementwise — every neuron keeps its own</span>`,
    ]));

    if(p < PH["g" + k]) break;
    // the outer product: a column times a row is the whole weight matrix
    out.push(eq(PH["g" + k], `${PH["g" + k] + 1}) ∂L/∂W${SUP[k]}`, [
      mat(`∂L/∂W${SUP[k]}`, n, m, (r, c) => ly.W[r][c].g,
          {fill:D_, txt:grd, row:row(k), grad:1,
           cell: sel.k === "w" && sel.l === k ? {j:sel.j, i:sel.i} : null}),
      mop("="),
      mat(`δ${SUP[k]}`, n, 1, r => ly.z[r].g, {fill:D_, txt:grd, row:row(k), grad:1}),
      mop("·"),
      mat(`${aName(k-1)}ᵀ`, 1, m, (r, c) => ly.in[c].d, {fill:k ? V_ : X_, txt:num}),
      `<span class="mop sm">and ∂L/∂b${SUP[k]} = δ${SUP[k]}</span>`,
    ]));

    // and the one equation a network adds: the gradient handed down a layer
    if(k > 0) out.push(eq(PH["g" + k], "", [
      mat(`∂L/∂${aName(k-1)}`, m, 1, r => G.layers[k-1].a[r].g,
          {fill:D_, txt:grd, row:row(k-1), grad:1}),
      mop("="),
      mat(`W${SUP[k]}ᵀ`, m, n, (r, c) => S.W[k][c][r], {fill:W_, txt:num}),
      mop("·"),
      mat(`δ${SUP[k]}`, n, 1, r => ly.z[r].g, {fill:D_, txt:grd, grad:1}),
      `<span class="mop sm">the same wires, transposed — this is the whole recursion</span>`,
    ]));
  }

  /* ---- the step ---- */
  if(p >= PH.u){
    const k = sel.l;
    out.push(eq(PH.u, `${PH.u + 1}) update`, [
      mat(`W${SUP[k]}`, sh[k+1], sh[k], (r, c) => S.W[k][r][c] - S.lr*GW[k][r][c],
          {fill:W_, txt:num, cell: sel.k === "w" ? {j:sel.j, i:sel.i} : null}),
      mop("="),
      mat(`W${SUP[k]}`, sh[k+1], sh[k], (r, c) => S.W[k][r][c], {fill:W_, txt:num}),
      mop("−"),
      `<span class="mop">${fmt(S.lr)}</span>`, mop("·"),
      mat(`∂L/∂W${SUP[k]}`, sh[k+1], sh[k], (r, c) => GW[k][r][c], {fill:D_, txt:grd, grad:1}),
      `<span class="mop sm">the batch mean, over all ${S.data.length} points —
       and the same for every other layer</span>`,
    ]));
  }

  el("readout").innerHTML = out.join("");
}

/* The sheet is written symbolically once — the letters never move. Under each
   letter hangs the number it is holding for this point, and those are the only
   things that change as you step, drag a wire or scrub through the data. */
function renderMath(){
  if(isNet()) return netMath();
  const P = palette(), A = ACT[S.act], p = S.phase, out = [];
  const gfill = v => gramp(v, P, 0.06, 0.34);

  out.push(mrow(-1, "point", `<b>${S.sample + 1} of ${S.data.length}</b>
    <span style="color:var(--dim2)">— hover a point on the fit, or use ◀ ▶</span>`));

  /* 1) the weighted sum, one term per wire */
  const e1 = [tt("z", "", {cls:"hd"}), op("=")];
  for(let i=0;i<S.nin;i++){
    if(i) e1.push(op("+"));
    e1.push(tt(`w${SUB[i]}`, wv(S.w[i]), {chip:"t" + i, fill:ramp(S.w[i], WMAX, P, 0.05, 0.3)}),
            op("·"),
            tt(`x${SUB[i]}`, fv(G.d.x[i]), {chip:"t" + i, fill:ramp(G.d.x[i], XMAX, P, 0.05, 0.3)}));
  }
  e1.push(op("+"), tt("b", wv(S.b), {chip:"tb", fill:ramp(S.b, WMAX, P, 0.05, 0.3)}),
          op("="), res(fmt(G.z.d)));
  out.push(erow(PH.z, "1) sum", e1));

  /* 2) through the activation */
  if(p >= PH.a) out.push(erow(PH.a, "2) activate", [
    tt("a", "", {cls:"hd"}), op("="), op(`${A.lbl}(`),
    tt("z", fv(G.z.d), {fill:ramp(G.z.d, ZMAX, P, 0.05, 0.3)}), op(")"),
    op("="), res(fmt(G.a.d)),
  ]));

  /* 3) the loss */
  if(p >= PH.L) out.push(erow(PH.L, "3) loss", [
    tt("L", "", {cls:"hd"}), op("="), op("("),
    tt("a", fv(G.a.d), {fill:ramp(G.a.d, 1.2, P, 0.05, 0.3)}), op("−"),
    tt("t", fv(G.d.t), {fill:ramp(G.d.t, 1.2, P, 0.05, 0.3)}), op(")²"),
    op("="), res(fmt(G.L.d, 4)),
    op("&nbsp;&nbsp;"), tt(`batch L — the mean over all ${S.data.length}`, `<b>${fmt(LOSS, 4)}</b>`, {cls:"op"}),
  ]));

  /* 4) the first derivative — where the backward pass starts */
  if(p >= PH.ga) out.push(erow(PH.ga, "4) ∂L/∂a", [
    tt("∂L/∂a", "", {cls:"hd"}), op("="), op("2 ("),
    tt("a", fv(G.a.d), {chip:"dLda"}), op("−"), tt("t", fv(G.d.t), {chip:"dLda"}), op(")"),
    op("="), res(fmtg(G.a.g)),
  ]));

  /* 5) one multiplication by the local slope of the activation */
  if(p >= PH.gz) out.push(erow(PH.gz, "5) ∂L/∂z", [
    tt("∂L/∂z", "", {cls:"hd"}), op("="),
    tt("∂L/∂a", gv(G.a.g), {chip:"dLda", fill:gfill(G.a.g)}), op("·"),
    tt("∂a/∂z", gv(G.a.local), {chip:"dadz", fill:gfill(G.a.local)}),
    op("="), res(fmtg(G.z.g)),
    op("&nbsp;&nbsp;"), tt(`∂a/∂z = ${A.der}`, "", {cls:"op"}),
  ]));

  /* 6) and it spreads onto the parameters, one x or w each */
  if(p >= PH.gw){
    for(let i=0;i<S.nin;i++) out.push(erow(PH.gw, i ? "" : "6) ∂L/∂w", [
      tt(`∂L/∂w${SUB[i]}`, "", {cls:"hd"}), op("="),
      tt("∂L/∂z", gv(G.z.g), {chip:"dadz", fill:gfill(G.z.g)}), op("·"),
      tt(`x${SUB[i]}`, fv(G.d.x[i]), {chip:"t" + i, fill:ramp(G.d.x[i], XMAX, P, 0.05, 0.3)}),
      op("="), res(fmtg(G.w[i].g)),
    ]));
    out.push(erow(PH.gw, "", [
      tt("∂L/∂b", "", {cls:"hd"}), op("="),
      tt("∂L/∂z", gv(G.z.g), {chip:"dadz", fill:gfill(G.z.g)}), op("·"),
      tt("1", fv(1), {chip:"tb"}),
      op("="), res(fmtg(G.b.g)),
    ]));
    // not parameters — this is the row a deeper net would hand to the layer below
    for(let i=0;i<S.nin;i++) out.push(erow(PH.gw, "", [
      tt(`∂L/∂x${SUB[i]}`, "", {cls:"op"}), op("="),
      tt("∂L/∂z", gv(G.z.g), {cls:"op"}), op("·"),
      tt(`w${SUB[i]}`, wv(S.w[i]), {cls:"op"}),
      op("="), op(`<b>${fmtg(G.x[i].g)}</b>`),
      i ? "" : op("&nbsp;&nbsp;— not a parameter: this is what a deeper net hands to the layer below"),
    ]));
    out.push(mrow(PH.gw, "all " + S.data.length,
      GRAD.w.map((v, i) => plain(`∂L/∂w${SUB[i]} = ${gv(v)}`, gramp(v, P, 0.06, 0.45))).join(" ") +
      plain(`∂L/∂b = ${gv(GRAD.b)}`, gramp(GRAD.b, P, 0.06, 0.45)) +
      ` <span style="color:var(--dim2)">— the mean over every point, and what the update actually uses</span>`));
  }

  /* 7) the step itself */
  if(p >= PH.u){
    const upd = (name, cur, gr, i) => erow(PH.u, i ? "" : "7) update", [
      tt(name, "", {cls:"hd"}), op("←"),
      tt(name, wv(cur), {fill:ramp(cur, WMAX, P, 0.05, 0.3)}), op("−"),
      tt("η", fmt(S.lr), {cls:"op"}), op("·"),
      tt(`∂L/∂${name}`, gv(gr), {fill:gfill(gr)}),
      op("="), res(fmt(cur - S.lr*gr)),
    ]);
    for(let i=0;i<S.nin;i++) out.push(upd(`w${SUB[i]}`, S.w[i], GRAD.w[i], i));
    out.push(upd("b", S.b, GRAD.b, 1));
  }

  el("readout").innerHTML = out.join("");
}

/* ---------- the chain, for a weight somewhere inside a net ----------
   Walk from the weight up to the loss. Every layer above it that has a single
   neuron contributes two more factors, so the product visibly grows with depth —
   which is exactly why deep networks have a gradient problem. Where the layer
   above has several neurons the paths have to be added rather than multiplied,
   and that factor is shown as the sum it is instead of being faked.
   -------------------------------------------------------------------- */
function netChain(){
  const P = palette(), p = S.phase, sh = shape(), nl = NLAY(), s = S.sel;
  const l = s.l, j = s.j;
  const gfl = v => gramp(v, P, 0.06, 0.34);
  const F = [];

  const ly = G.layers[l];
  // the output layer holds one neuron, so from the layer below it there is
  // exactly one path to the loss and the product can be written out in full.
  // Further down the paths fan out and have to be summed instead.
  if(l >= nl - 2){
    const top = G.layers[nl-1];
    F.push({at:PH.ga, sym:`∂L/∂a${SUP[nl-1]}`, rule:"2(a − t)",
            num:fmtg(top.a[0].g), fill:gfl(top.a[0].g)});
    if(l === nl - 2){
      F.push({at:PH["d" + (nl-1)], sym:`f′(z${SUP[nl-1]})`, rule:top.A.der,
              num:fmtg(top.a[0].local), fill:gfl(top.a[0].local)});
      F.push({at:PH["g" + (nl-1)], sym:`w${SUP[nl-1]}${SUB[0]}${SUB[j]}`,
              rule:"the wire out of it", val:true,
              num:fmt(S.W[nl-1][0][j]), fill:ramp(S.W[nl-1][0][j], WMAX, P, 0.06, 0.34)});
    }
  } else {
    F.push({at:PH["d" + l], sym:`∂L/∂${aName(l)}${SUB[j]}`,
            rule:sh[l+2] > 1 ? `Σ over the ${sh[l+2]} paths above` : "the one path above",
            num:fmtg(ly.a[j].g), fill:gfl(ly.a[j].g)});
  }
  F.push({at:PH["d" + l], sym:`f′(z${SUP[l]}${SUB[j]})`, rule:ly.A.der,
          num:fmtg(ly.a[j].local), fill:gfl(ly.a[j].local)});
  const loc = s.k === "b" ? 1 : ly.in[s.i].d;
  F.push({at:PH["g" + l], val:true,
          sym:s.k === "b" ? "∂z/∂b" : `${aName(l-1)}${SUB[s.i]}`,
          rule:s.k === "b" ? "1" : "the value it multiplies",
          num:fmt(loc), fill:ramp(loc, XMAX, P, 0.06, 0.34)});

  const nm = s.k === "b" ? `b${SUP[l]}${SUB[j]}` : `w${SUP[l]}${SUB[j]}${SUB[s.i]}`;
  const node = s.k === "b" ? ly.B[j] : ly.W[j][s.i];
  const html = [`<span class="lhs">∂L/∂${nm}</span><span class="dot">=</span>`];
  F.forEach((f, i) => {
    if(i) html.push(`<span class="dot">·</span>`);
    const done = p >= f.at && !diverged();
    const cls = ["cbf", f.val ? "val" : "", done ? "" : "off",
                 p === f.at ? "now" : ""].filter(Boolean).join(" ");
    html.push(`<span class="${cls}"${done ? ` style="background:${f.fill}"` : ""}>` +
              `<b>${f.sym}</b><i>${f.rule}</i><u>${done ? f.num : "?"}</u></span>`);
  });
  html.push(`<span class="dot">=</span>`,
            `<span class="out">${p >= PH["g" + l] && !diverged() ? fmtg(node.g) : "?"}</span>`);
  el("chainbar").innerHTML = html.join("");
}

/* ---------- the chain rule, large, on top of the diagram ----------
   Always the same three factors in the same order. Each stays greyed until the
   backward pass has actually produced it, so stepping fills the product in from
   the left — which is the order the gradient is multiplied in.
   ----------------------------------------------------------------- */
function renderChainBar(){
  if(isNet()) return netChain();
  const P = palette(), A = ACT[S.act], s = selInfo(), p = S.phase;
  const F = [
    {at:PH.ga, chip:"dLda", sym:"∂L/∂a", rule:"2(a − t)",
     num:fmtg(G.a.g),     fill:gramp(G.a.g, P, 0.06, 0.34)},
    {at:PH.gz, chip:"dadz", sym:"∂a/∂z", rule:A.der,
     num:fmtg(G.a.local), fill:gramp(G.a.local, P, 0.06, 0.34)},
    {at:PH.gw, chip:"dzdp", sym:`∂z/∂${s.name}`, rule:s.localTxt, val:true,
     num:fmt(s.local),    fill:ramp(s.local, S.sel.k === "w" ? XMAX : WMAX, P, 0.06, 0.34)},
  ];

  const html = [`<span class="lhs">∂L/∂${s.name}</span><span class="dot">=</span>`];
  F.forEach((f, i) => {
    if(i) html.push(`<span class="dot">·</span>`);
    const done = p >= f.at && !diverged();
    const cls = ["cbf", f.val ? "val" : "", done ? "" : "off",
                 p === f.at ? "now" : "", S.chip === f.chip ? "lit" : ""].filter(Boolean).join(" ");
    html.push(`<span class="${cls}" data-chip="${f.chip}"${done ? ` style="background:${f.fill}"` : ""}>` +
              `<b>${f.sym}</b><i>${f.rule}</i><u>${done ? f.num : "?"}</u></span>`);
  });
  html.push(`<span class="dot">=</span>`,
            `<span class="out">${p >= PH.gw && !diverged() ? fmtg(s.node.g) : "?"}</span>`);
  el("chainbar").innerHTML = html.join("");
}

function markChips(){
  el("readout").querySelectorAll("[data-chip]").forEach(n =>
    n.classList.toggle("on", !!S.chip && n.dataset.chip === S.chip));
  renderChainBar();
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
  ga:`<b>The backward pass starts.</b> ∂L/∂L is 1 by definition, and the first real derivative is the slope
      of the bowl at this point: 2(a − t). It is the leftmost factor of the chain rule above — watch that
      factor come out of grey.`,
  gz:`<b>One factor further.</b> The gradient crosses the neuron by being multiplied by ∂a/∂z, the violet
      tangent drawn inside the activation card. That single multiply is where a saturated unit kills
      learning: multiply by a flat slope and there is nothing left.`,
  gw:`<b>Onto the parameters.</b> The last factor is just ∂z/∂w = x, so every weight gets ∂L/∂z times its own
      input. Point at a wire and the dashed line shows which forward value its gradient reuses — that is why
      activations have to be kept.`,
  u: `<b>The update.</b> The gradients above are for this one point; the step uses their mean over all of
      them. Press ▶ to apply it — the curve moves and one point lands on the loss plot.`,
};

// a net has one caption per kind of station, not per station
function caption(){
  const ph = PHASES[S.phase];
  if(!isNet()) return CAPS[ph.id];
  const k = ph.id[0], l = ph.l, nl = NLAY();
  if(k === "f") return `<b>Layer ${l + 1} fires.</b> Every neuron in the column does exactly what the single
      neuron does — sum its wires, add its bias, squash. The four cards below are that arithmetic for the
      neuron you have selected; <b>click another node</b> and they re-point at it.`;
  if(ph.id === "L")  return CAPS.L;
  if(ph.id === "ga") return CAPS.ga;
  if(k === "d") return `<b>Through the activation of layer ${l + 1}.</b> Each neuron multiplies the gradient
      that reached it by its own f′(z) — the violet tangent in the second card. That product is
      <b>δ</b>, one number per neuron, and it is all the layer below needs to know.
      ${l === nl - 1 ? "" : `∂L/∂a here is a <b>sum over every path</b> to the loss, which is the one
      equation a network adds to a neuron.`}`;
  if(k === "g") return `<b>Onto layer ${l + 1}'s weights.</b> ∂L/∂w is δ for the neuron the wire enters,
      times the value the wire carried in — so each wire now has a second, violet lane running the other
      way, as thick as its gradient. ${l ? `The same δ, pushed back through Wᵀ, becomes ∂L/∂a for the layer
      below, and the whole thing repeats.` : `There is nothing below layer 1, so the pass stops here.`}`;
  return `<b>The update.</b> Every one of the ${nparams()} parameters moves at once, each by η times its own
      gradient averaged over all ${S.data.length} points. Press ▶ to apply it and watch the boundary bend.`;
}

function slider(id, label, cls, min, max, step, hint){
  return `<div class="ctrl" id="bp_ctrl_${id}">
    <label for="bp_r_${id}"><span class="${cls}">${label}</span><b id="bp_v_${id}"></b></label>
    <input type="range" id="bp_r_${id}" min="${min}" max="${max}" step="${step}">
    ${hint ? `<div class="hint">${hint}</div>` : ""}
  </div>`;
}

/* the rail is rebuilt whenever the architecture changes, because the steps are
   the architecture — two more of them for every layer you add. Past eight of
   them the per-button equations are unreadable anyway, so they come off and the
   current step's equation is printed once, beside the controls. */
function buildRail(){
  const many = PHASES.length > 8;
  let grp = "", html = [];
  PHASES.forEach((p, i) => {
    if(p.grp !== grp){ grp = p.grp; html.push(`<span class="rsep">${grp}</span>`); }
    html.push(`<button data-phase="${i}"><b>${i + 1}) ${p.lbl}</b>` +
              (many ? "" : `<i id="bp_eq${i}"></i>`) + `</button>`);
  });
  el("rail").innerHTML = html.join("");
  el("rail").classList.toggle("tight", many);
  document.querySelectorAll("#bp_rail button").forEach(b =>
    b.addEventListener("click", () => { stop(); S.phase = +b.dataset.phase; refresh(); }));
}

function buildControls(){
  el("netseg").innerHTML = Object.entries(NETS).map(([k, d]) =>
    `<button data-net="${k}"><b>${d.lbl}</b><i>${d.note}</i></button>`).join("");
  el("setseg").innerHTML = Object.entries(SETS).map(([k, d]) =>
    `<button data-set="${k}"><b>${d.name}</b><i>${d.note}</i></button>`).join("");
  buildRail();

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
  segs("spdseg", "spd", v => { S.spd = +v; if(S.playing){ stop(); start(); } syncControls(); });
  segs("netseg", "net", v => { stop(); loadNet(v); buildRail(); refresh(); });
  segs("widthseg", "w", v => { stop(); S.width = +v; loadSet(S.set); buildRail(); refresh(); });

  el("rand").addEventListener("click", randomise);
  el("train").addEventListener("click", () => train(isNet() ? 200 : 50));
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
  renderMath(); renderChainBar(); paint(); paintFit(); syncControls();
}

function step(){
  if(S.phase < PHASES.length - 1) S.phase++;
  else { applyUpdate(); S.phase = 0; }
  refresh();
}
/* how fast ▶ runs. Up to 2× it walks the stations one at a time; past that the
   stations would be a blur, so it stops walking them and just trains — which is
   the point of the fast settings: watch the boundary and the loss, not the
   arithmetic. */
const SPEEDS = {1:{ms:850, per:0}, 2:{ms:400, per:0}, 10:{ms:110, per:1}, 50:{ms:60, per:5}};
let timer = 0;
function playTick(){
  const sp = SPEEDS[S.spd];
  if(!sp.per) return step();
  for(let k=0;k<sp.per && !diverged();k++) applyUpdate();
  S.phase = PH.u;                            // park on the step it is repeating
  refresh();
  if(diverged()) stop();
}
function start(){ S.playing = true; timer = setInterval(playTick, SPEEDS[S.spd].ms); syncControls(); }
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
  el("wctrls").classList.toggle("gone", isNet());      // a net has too many to slide
  el("paramhead").classList.toggle("gone", isNet());
  el("widthwrap").classList.toggle("gone", !isNet());

  const on = (id, attr, val) => document.querySelectorAll(`#bp_${id} button`).forEach(b =>
    b.classList.toggle("on", b.dataset[attr] === String(val)));
  on("setseg", "set", S.set);
  on("actseg", "act", S.act);
  on("lrseg",  "lr",  S.lr);
  on("spdseg", "spd", S.spd);
  on("netseg", "net", S.net);
  on("widthseg", "w", S.width);
  // a dataset a single neuron cannot do is not offered to a single neuron
  document.querySelectorAll("#bp_setseg button").forEach(b => {
    const f = SETS[b.dataset.set].for;
    b.classList.toggle("off", f !== "both" && f !== (isNet() ? "net" : "one"));
  });

  const many = PHASES.length > 8;
  document.querySelectorAll("#bp_rail button").forEach((b, i) => {
    b.classList.toggle("on", i === S.phase);
    b.classList.toggle("done", i < S.phase);
    if(!many) el("eq" + i).textContent = PHASES[i].eq();
  });
  el("stepeq").textContent = many ? `${S.phase + 1})  ${PHASES[S.phase].eq()}` : "";

  el("acthint").innerHTML = (isNet()
    ? `<b>The hidden layers</b> use this; the output neuron is always a sigmoid, since the targets are 0/1.
       <br>` : "") + ACT[S.act].hint;
  el("vizcap").innerHTML = caption();                  // not in paint(), which runs per frame
  el("play").textContent = S.playing ? "❚❚ pause" : "▶ play";
  el("prev").disabled = S.phase === 0;
  el("slabel").textContent = `point ${S.sample + 1} / ${S.data.length}`;
  el("trails").checked = S.trails;
  el("ctrl_trails").classList.toggle("gone", isNet());
  el("stephint").innerHTML = S.spd > 2
    ? `<b>${S.steps}</b> updates — ▶ trains without stopping at the stations.`
    : S.phase === PHASES.length - 1
      ? `▶ applies the update and starts the next pass.`
      : `${S.steps} update${S.steps === 1 ? "" : "s"} so far.`;
  el("train").innerHTML = `&#9654; train &times;${isNet() ? 200 : 50}`;
  el("trainhint").innerHTML = diverged()
    ? `<span class="err">Diverged — reset before training further.</span>`
    : `Runs ${isNet() ? 200 : 50} full passes over all ${S.data.length} points.`;
  // what this dataset is and why it is worth fitting — it belongs beside the
  // button that picks it, not under the plots
  el("setnote").innerHTML = `<b>${S.data.length}</b> points, <b>${S.nin}</b> input${S.nin === 1 ? "" : "s"}
    — ${isNet() ? "the whole net is" : "the neuron is"} fitted to all of them at once.
    <br><br>${SETS[S.set].why}`;
}

/* ============================== interaction ============================== */
// on a net: the nodes pick whose stations the strip shows, the wires pick which
// weight the chain rule is written for — and, exactly as with one neuron, a wire
// is also the handle that changes it
function hitNet(ev){
  const r = el("viz").getBoundingClientRect();
  const x = (ev.clientX - r.left)*(LY.w/r.width), y = (ev.clientY - r.top)*(LY.h/r.height);
  const sh = shape(), nl = NLAY();

  for(let l=0;l<=nl;l++)
    for(let j=0;j<sh[l];j++)
      if(Math.hypot(x - nodeX(l), y - nodeY(l, j)) < NR + 6){
        if(l === 0) return {part:`x${j}`, node:null};
        return {part:`n${l-1}_${j}`, node:{l:l-1, j}};
      }

  for(let l=0;l<nl;l++)
    for(let j=0;j<sh[l+1];j++)
      for(let i=0;i<sh[l];i++){
        const s = {x0:nodeX(l) + NR, y0:nodeY(l, i), x1:nodeX(l+1) - NR, y1:nodeY(l+1, j)};
        if(segDist(x, y, s) < 6)
          return {part:`w${l}_${j}_${i}`, sel:{k:"w", l, j, i}, node:{l, j},
                  drag:{k:"w", l, j, i}};
      }
  return null;
}

function hit(ev){
  if(!LY) return null;
  if(isNet()) return hitNet(ev);
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

  {
    const B = backBoxes();
    if(S.phase >= PH.ga && inBox(B.a.x, B.a.w, BY, LY.bh)) return {part:"ga", sel:null};
    if(S.phase >= PH.gz && inBox(B.z.x, B.z.w, BY, LY.bh)) return {part:"gz", sel:null};
    if(S.phase >= PH.gw && inBox(B.p.x, B.p.w, BY, LY.bh)){
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
      const D = S.drag;
      if(isNet()) S.W[D.l][D.j][D.i] = v;
      else if(D.k === "b") S.b = v;
      else S.w[D.i] = v;
      refresh();
      return;
    }
    const h = hit(e);
    const part = h ? h.part : null;
    const sel  = h && h.sel ? h.sel : S.sel;
    cv.style.cursor = h && h.drag ? "ns-resize" : (h && h.node ? "pointer" : "default");
    const same = sel === S.sel || (sel.k === S.sel.k && sel.i === S.sel.i &&
                                   sel.l === S.sel.l && sel.j === S.sel.j);
    if(part === S.hover && same) return;
    S.hover = part; S.sel = sel;
    if(h && h.node) S.selN = h.node;
    renderMath(); renderChainBar(); paint();
  });
  cv.addEventListener("mousedown", e => {
    const h = hit(e);
    if(!h || !h.drag) return;
    e.preventDefault();
    const d = h.drag;
    S.drag = {...d, y0: e.clientY,
              v0: isNet() ? S.W[d.l][d.j][d.i] : (d.k === "b" ? S.b : S.w[d.i])};
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

  // the sheet and the big chain rule light the same paths on the canvas
  ["readout", "chainbar"].forEach(id => {
    el(id).addEventListener("mouseover", e => {
      const t = e.target.closest("[data-chip]");
      const c = t ? t.dataset.chip : "";
      if(c === S.chip) return;
      S.chip = c; markChips(); paint();
    });
    el(id).addEventListener("mouseleave", () => {
      if(!S.chip) return;
      S.chip = ""; markChips(); paint();
    });
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
  if(hidden() || S.phase < PH.ga) return;
  frame++;
  paint();
  raf = requestAnimationFrame(tick);
}
function animate(){
  if(!raf && S.phase >= PH.ga && !hidden()) raf = requestAnimationFrame(tick);
}

/* ================================ main ================================ */
function refresh(){
  if(S.sel.k !== "b" && S.sel.i >= S.nin) S.sel = {k:"w", i:0};
  compute();
  scales();
  if(S.playing && diverged()) stop();
  syncControls();
  renderMath();
  renderChainBar();
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
