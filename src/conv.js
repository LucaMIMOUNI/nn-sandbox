/* ==================================================================
   MODULE 1 — Conv2d / ConvTranspose2d
   ================================================================== */
const CONV = (function(){

function unb64(s){
  const bin = atob(s), out = new Uint8Array(bin.length);
  for(let i=0;i<bin.length;i++) out[i] = bin.charCodeAt(i);
  return out;
}
const DIGITS = DIGIT_B64.map(o => ({label:o.label, px:unb64(o.b64), n:28}));
let PHOTO = {px:unb64(PHOTO_B64), n:PHOTO_N, credit:true};

/* ================================ state ================================ */
const S = {
  layer:"conv",                 // "conv" (gather) | "convT" (scatter)
  n:5, k:3, s:2, p:1, opad:0, d:1,
  mode:"matrix", kernel:"box", showCounts:false,
  digit:0, res:28,
  sel:{i:1, j:1},        // the cell the stepper / progressive build-up is on
  focus:null,            // {stage,r,c} — the cell whose maths is on show
  tap:null,              // {a,b} — kernel weight under the mouse
  progressive:false,
  matSeed:1,
  calc:{h:null, w:null},  // shape calculator dims; null = follow the drawn grid
};
const isT = () => S.layer === "convT";

const calcH = () => S.calc.h === null ? S.n : S.calc.h;
const calcW = () => S.calc.w === null ? S.n : S.calc.w;
const calcLinked = () => S.calc.h === null && S.calc.w === null;

// every slider spans the same 10 steps so the thumbs line up across parameters
const PARAMS = [
  {id:"k",  name:"kernel_size", min:1, max:11, cls:"k",
   hint:{conv:"size of the window that slides over the input",
         convT:"size of the stamp each input pixel leaves"}},
  {id:"s",  name:"stride", min:1, max:10, cls:"s",
   hint:{conv:"how far the window jumps = the downsampling factor",
         convT:"gap between stamps = the upsampling factor"}},
  {id:"p",  name:"padding", min:0, max:9, cls:"p",
   hint:{conv:"adds this many zero rows/cols round the input first",
         convT:"crops this many rows/cols off every side"}},
  {id:"opad", name:"output_padding", min:0, max:9, cls:"op", only:"convT",
   hint:{conv:"", convT:"adds rows/cols back on bottom & right only"}},
  {id:"d",  name:"dilation", min:1, max:10, cls:"d",
   hint:{conv:"spreads the window out with holes",
         convT:"spreads the stamp out with holes"}},
];

/* --------- presets: parameters lifted straight out of real networks --------- */
const PRESETS = {
  conv:[
    {t:"LeNet-5 · conv1",              q:{k:5,s:1,p:0,d:1},  n:8,  in:32},
    {t:"AlexNet · conv1",              q:{k:11,s:4,p:2,d:1}, n:12, in:224},
    {t:"VGG · the 3×3 that ate the world", q:{k:3,s:1,p:1,d:1}, n:6, in:224},
    {t:"ResNet · stem",                q:{k:7,s:2,p:3,d:1},  n:8,  in:224},
    {t:"Inception · 1×1 bottleneck",   q:{k:1,s:1,p:0,d:1},  n:6,  in:28},
    {t:"MobileNet · depthwise, s2",    q:{k:3,s:2,p:1,d:1},  n:6,  in:112},
    {t:"DeepLab · dilated 3×3",        q:{k:3,s:1,p:2,d:2},  n:7,  in:28},
  ],
  convT:[
    {t:"DCGAN · first layer (z → 4×4)", q:{k:4,s:1,p:0,opad:0,d:1}, n:1, in:1},
    {t:"DCGAN / pix2pix · ×2 block",    q:{k:4,s:2,p:1,opad:0,d:1}, n:4, in:32},
    {t:"U-Net · up-conv 2×2",           q:{k:2,s:2,p:0,opad:0,d:1}, n:4, in:28},
    {t:"Odena et al. · checkerboard",   q:{k:3,s:2,p:1,opad:1,d:1}, n:4, in:32},
    {t:"FSRCNN · ×3 deconv",            q:{k:9,s:3,p:3,opad:0,d:1}, n:4, in:28},
    {t:"size-preserving 3×3",           q:{k:3,s:1,p:1,opad:0,d:1}, n:5, in:32},
  ],
};

const el = id => document.getElementById(id);
let D  = {};        // derived geometry
let IN = null;      // Float32Array n*n     — the input values
let W  = null;      // Float32Array k*k     — the kernel weights
let CAN = null;     // Float32Array ext*ext — the middle grid, as displayed
let FULLCAN = null; // Float32Array ext*ext — the middle grid, complete
let OUT = null;     // Float32Array out*out — conv only; convT reads its output off CAN
let CNT = null;     // Int32Array ext — per-axis count (stamps landing / windows reading)
let MX = {};        // per-stage max, for colour normalisation
let L = {};         // pixel layout
let baseCv = null;  // cached background render

// the stepper walks input pixels for a transposed conv (one stamp at a time)
// and output pixels for a plain conv (one window position at a time)
const stepStage = () => isT() ? "in" : "out";
const stepN     = () => isT() ? S.n : Math.max(D.out, 0);
const selIdx    = () => S.sel.i*stepN() + S.sel.j;

/* ============================== geometry ==============================
   Both layers are described by the same four grids:
     input  →  (kernel)  →  middle grid  →  output
   convT: the middle grid is the canvas every input pixel stamps into,
          and the output is that canvas cropped by `padding`.
   conv:  the middle grid is the zero-padded input, and the output is what
          the sliding window computes from it.
   `contrib[r]` is, per axis, the list of {i, tap} pairs linking a middle-grid
   position to the input pixels that write it (convT) or the output pixels
   that read it (conv). Everything else is derived from that one table.
   ====================================================================== */
function derive(){
  const {n,k,s,p,opad,d} = S;
  const span = d*(k-1)+1;
  const contrib = [];
  let full, out, ext;

  if(isT()){
    full = (n-1)*s + span;
    out  = full - 2*p + opad;
    ext  = Math.max(full, p + Math.max(out,0));
    CNT  = new Int32Array(ext);
    for(let r=0;r<ext;r++){
      const list = [];
      const lo = Math.max(0, Math.ceil((r - d*(k-1))/s));
      const hi = Math.min(n-1, Math.floor(r/s));
      for(let i=lo;i<=hi;i++){ const t = r - i*s; if(t % d === 0) list.push({i, tap:t/d}); }
      contrib.push(list); CNT[r] = list.length;
    }
  } else {
    full = n + 2*p;                              // the padded input
    out  = Math.floor((full - span)/s) + 1;
    ext  = full;
    CNT  = new Int32Array(ext);
    for(let r=0;r<ext;r++){
      const list = [];
      for(let a=0;a<k;a++){
        const t = r - a*d;
        if(t < 0 || t % s) continue;
        const R = t/s;
        if(R >= 0 && R < out) list.push({i:R, tap:a});
      }
      contrib.push(list); CNT[r] = list.length;
    }
  }

  // how many products land on / read each cell — the checkerboard & coverage story
  let mn = Infinity, mx = 0;
  if(out > 0){
    if(isT()){
      for(let r=p;r<p+out;r++) for(let c=p;c<p+out;c++){
        const v = (r<full?CNT[r]:0) * (c<full?CNT[c]:0);
        if(v<mn) mn=v; if(v>mx) mx=v;
      }
    } else {
      for(let r=p;r<p+n;r++) for(let c=p;c<p+n;c++){
        const v = CNT[r]*CNT[c];
        if(v<mn) mn=v; if(v>mx) mx=v;
      }
    }
  }
  D = {span, full, out, ext, contrib,
       valid: out > 0, opOk: !isT() || opad < Math.max(s,d),
       min: mn===Infinity?0:mn, max: mx};
}

// one axis of the same geometry for any input length — the shape calculator
// works on sizes far too big to draw, so it cannot lean on derive().
function axisStats(hin){
  const {k,s,p,opad,d} = S;
  const span = d*(k-1)+1;
  if(isT()){
    const full = (hin-1)*s + span;
    return {span, full, out: full - 2*p + opad, valid: full - 2*p + opad > 0, drop:0};
  }
  const full = hin + 2*p;
  const out  = Math.floor((full - span)/s) + 1;
  return {span, full, out, valid: out > 0, drop: out > 0 ? (full - span) % s : 0};
}

/* =============================== kernel =============================== */
let seed = 7;
function rnd(){ seed = (seed*1664525 + 1013904223) >>> 0; return seed / 4294967296; }
function buildKernel(){
  const k = S.k; W = new Float32Array(k*k);
  if(S.kernel === "box"){ W.fill(1); }
  else if(S.kernel === "gauss"){
    const c = (k-1)/2, sig = Math.max(k/3, .5);
    for(let a=0;a<k;a++) for(let b=0;b<k;b++)
      W[a*k+b] = Math.round(10*Math.exp(-(((a-c)**2 + (b-c)**2)/(2*sig*sig))))/10;
  } else if(S.kernel === "edge"){
    W.fill(-1);
    const c = Math.floor((k-1)/2);
    W[c*k+c] = k*k - 1;
    if(k === 1) W[0] = 1;
  } else {
    seed = 7 + k*97;
    for(let i=0;i<k*k;i++) W[i] = Math.floor(rnd()*4);      // small integers, readable arithmetic
    if(W.every(v => v === 0)) W[(k*k-1)>>1] = 1;
  }
  el("kernelhint").innerHTML = ({
    box:  isT() ? "All weights = 1, so a canvas value is literally the sum of the input pixels that landed on it."
                : "All weights = 1, so every output value is literally the sum of the window under it — a blur.",
    gauss:isT() ? "Bell-shaped weights — the smooth, bilinear-ish upsample."
                : "Bell-shaped weights — a Gaussian blur.",
    edge: "Negative surround, positive centre — a Laplacian. Flat regions cancel to 0, edges survive."
          + (S.mode === "matrix" ? "" : " Switch to Matrix to read the signs."),
    rand: "Arbitrary small integers, closer to what a trained layer holds.",
  })[S.kernel];
}

/* =============================== input ================================ */
function boxResample(src, sn, dn){
  const out = new Float32Array(dn*dn), scale = sn/dn;
  for(let r=0;r<dn;r++){
    const y0 = r*scale, y1 = (r+1)*scale;
    for(let c=0;c<dn;c++){
      const x0 = c*scale, x1 = (c+1)*scale;
      let acc = 0, wsum = 0;
      for(let y=Math.floor(y0); y<Math.min(sn, Math.ceil(y1)); y++){
        const wy = Math.min(y+1,y1) - Math.max(y,y0);
        for(let x=Math.floor(x0); x<Math.min(sn, Math.ceil(x1)); x++){
          const wx = Math.min(x+1,x1) - Math.max(x,x0);
          acc += src[y*sn+x]*wy*wx; wsum += wy*wx;
        }
      }
      out[r*dn+c] = wsum ? acc/wsum : 0;
    }
  }
  return out;
}

function buildInput(){
  const n = S.n;
  if(S.mode === "matrix"){
    seed = S.matSeed*7919 + 13;                            // deterministic per reroll
    IN = new Float32Array(n*n);
    for(let i=0;i<n*n;i++) IN[i] = Math.floor(rnd()*6);    // integers 0…5
    return;
  }
  const src = S.mode === "mnist" ? DIGITS[S.digit] : PHOTO;
  const f = new Float32Array(src.n*src.n);
  for(let i=0;i<f.length;i++) f[i] = src.px[i]/255;
  IN = (src.n === n) ? f : boxResample(f, src.n, n);
}

/* ============================== the maths ============================== */
// convT — scatter: every input pixel adds value * kernel into the canvas at i*stride.
// `limit` stops after that many input pixels, for the progressive build-up.
function scatterInto(limit){
  const {n,k,s,d} = S, ext = D.ext;
  const acc = new Float32Array(ext*ext);
  const stop = limit === undefined ? n*n : limit;
  for(let t=0; t<stop; t++){
    const i = (t/n)|0, j = t%n, v = IN[t];
    if(v === 0) continue;
    for(let a=0;a<k;a++){
      const r = i*s + a*d;
      for(let b=0;b<k;b++) acc[r*ext + j*s + b*d] += v*W[a*k+b];
    }
  }
  return acc;
}

function compute(){
  const ext = D.ext;
  if(isT()){
    FULLCAN = scatterInto();
    CAN = S.progressive ? scatterInto(selIdx()+1) : FULLCAN;
    OUT = null;
  } else {
    // conv — gather: the middle grid is just the zero-padded input
    CAN = new Float32Array(ext*ext);
    for(let i=0;i<S.n;i++) for(let j=0;j<S.n;j++) CAN[(i+S.p)*ext + j+S.p] = IN[i*S.n+j];
    FULLCAN = CAN;
    const o = Math.max(D.out, 0);
    OUT = new Float32Array(o*o);
    for(let r=0;r<o;r++) for(let c=0;c<o;c++){
      let acc = 0;
      for(let a=0;a<S.k;a++){
        const rr = (r*S.s + a*S.d)*ext;
        for(let b=0;b<S.k;b++) acc += CAN[rr + c*S.s + b*S.d] * W[a*S.k+b];
      }
      OUT[r*o+c] = acc;
    }
  }
  const amax = arr => { let m = 0; for(let i=0;i<arr.length;i++) m = Math.max(m, Math.abs(arr[i])); return m || 1; };
  MX = {in: amax(IN), ker: amax(W),
        can: isT() ? amax(FULLCAN) : amax(IN),
        out: isT() ? amax(FULLCAN) : amax(OUT)};
  D.vmax = MX.can;
}

// convT — every (input pixel, kernel tap) pair that feeds canvas[R][C]
function termsFor(R, C){
  const out = [];
  if(R >= D.full || C >= D.full) return out;
  for(const a of (D.contrib[R] || [])) for(const b of (D.contrib[C] || []))
    out.push({i:a.i, j:b.i, a:a.tap, b:b.tap,
              v:IN[a.i*S.n + b.i], w:W[a.tap*S.k + b.tap]});
  return out;
}

/* ============================== rendering ============================== */
const dark = () => matchMedia("(prefers-color-scheme: dark)").matches;
function palette(){
  const d = dark();
  return {
    line:  d ? "#2a323d" : "#d6dde5",
    dim:   d ? "#8b949e" : "#57606a",
    dim2:  d ? "#6e7681" : "#8c959f",
    fg:    d ? "#e6edf3" : "#1c2128",
    panel: d ? "#161b22" : "#ffffff",
    void_: d ? "#3a4552" : "#dbe2ea",
    accent:d ? "#4aa3ff" : "#0969da",
    amber: d ? "#f0a132" : "#bc6c00",
    amberRGB: d ? "240,161,50" : "188,108,0",
    ok:    "#3fb950",
    veil:  d ? "rgba(22,27,34,.82)" : "rgba(255,255,255,.80)",
  };
}
function heat(v, max){
  if(!v) return null;
  const t = Math.min(1, Math.abs(v)/(max || 1));
  const rgb = v > 0 ? (dark()?"74,163,255":"9,105,218") : (dark()?"248,81,73":"207,34,46");
  return `rgba(${rgb},${(0.16 + 0.74*t).toFixed(3)})`;
}
function grey(v){
  const g = Math.max(0, Math.min(255, Math.round(255*v)));
  return `rgb(${g},${g},${g})`;
}
const fmt = v => Number.isInteger(v) ? String(v)
               : Math.abs(v) >= 10 ? v.toFixed(0) : String(Math.round(v*10)/10);

function cellValue(stage, r, c){
  if(stage === "in")  return IN[r*S.n+c];
  if(stage === "ker") return W[r*S.k+c];
  if(isT()){
    const R = stage === "out" ? r + S.p : r;
    const C = stage === "out" ? c + S.p : c;
    if(R >= D.full || C >= D.full) return null;           // beyond the canvas: exact zero
    return CAN[R*D.ext + C];
  }
  if(stage === "can"){
    const i = r - S.p, j = c - S.p;
    if(i < 0 || j < 0 || i >= S.n || j >= S.n) return null;  // the zero padding
    return IN[i*S.n + j];
  }
  if(!OUT || !D.valid) return null;
  if(S.progressive && r*D.out + c > selIdx()) return null;   // not computed yet
  return OUT[r*D.out + c];
}
function cellCount(stage, r, c){
  if(isT()){
    const R = stage === "out" ? r + S.p : r;
    const C = stage === "out" ? c + S.p : c;
    if(R >= D.full || C >= D.full) return null;
    return CNT[R]*CNT[C];
  }
  return CNT[r]*CNT[c];
}
function cellFill(stage, r, c, P){
  const counted = S.showCounts && (stage === "can" || (isT() && stage === "out"));
  if(counted){
    const n = cellCount(stage, r, c);
    return n === null ? P.void_ : (n === 0 ? P.void_ : heat(n, D.max || 1));
  }
  const v = cellValue(stage, r, c);
  if(v === null) return P.void_;
  if(S.mode === "matrix" || stage === "ker") return heat(v, MX[stage] ?? 1);
  return grey(v / (MX[stage] ?? 1));
}

function layout(){
  const avail = Math.max(300, el("vizwrap").clientWidth - 2);
  const gOp = 34, gArrow = 52, capH = 17, rowGap = 30;
  const outCols = Math.max(D.out, 0);
  const r1 = S.n + S.k, r2 = D.ext + outCols;
  const cs = Math.max(3, Math.min(40, Math.min(
    Math.floor((avail - gOp - 4) / Math.max(r1,1)),
    Math.floor((avail - gArrow - 4) / Math.max(r2,1)))));

  const h1 = Math.max(S.n, S.k)*cs, h2 = Math.max(D.ext, outCols)*cs;
  const y1 = capH, y2 = capH + h1 + rowGap + capH;
  const m1 = n => y1 + (h1 - n*cs)/2, m2 = n => y2 + (h2 - n*cs)/2;
  const xK = S.n*cs + gOp, xO = D.ext*cs + gArrow;

  L = {cs, gOp, gArrow, y1, y2, h1, h2,
    w: Math.max(xK + S.k*cs, xO + outCols*cs),
    h: y2 + h2 + 4,
    stages: [
      {key:"in",  x:0,  y:m1(S.n),   cols:S.n,   label:`input ${S.n}×${S.n}`},
      {key:"ker", x:xK, y:m1(S.k),   cols:S.k,   label:`kernel ${S.k}×${S.k}`},
      {key:"can", x:0,  y:m2(D.ext), cols:D.ext,
       label: isT() ? `canvas ${D.full}×${D.full}` : `padded input ${D.full}×${D.full}`},
      {key:"out", x:xO, y:m2(outCols), cols:outCols,
       label:`output ${Math.max(D.out,0)}×${Math.max(D.out,0)}`},
    ]};
}
const stage = key => L.stages.find(s => s.key === key);

function fitCanvas(cv, w, h){
  const dpr = window.devicePixelRatio || 1;
  cv.width = Math.round(w*dpr); cv.height = Math.round(h*dpr);
  cv.style.width = w+"px"; cv.style.height = h+"px";
  const c = cv.getContext("2d"); c.setTransform(dpr,0,0,dpr,0,0);
  return c;
}

function renderBase(){
  const P = palette(), cs = L.cs;
  baseCv = baseCv || document.createElement("canvas");
  const g = fitCanvas(baseCv, L.w, L.h);
  g.clearRect(0,0,L.w,L.h);
  const showNums = cs >= 15;

  for(const st of L.stages){
    if(st.cols <= 0) continue;
    g.textAlign = "left"; g.textBaseline = "alphabetic";
    g.font = "11px ui-monospace,Menlo,monospace"; g.fillStyle = P.dim;
    g.fillText(st.label, st.x, Math.max(11, st.y - 5));
    g.textAlign = "center"; g.textBaseline = "middle";

    for(let r=0;r<st.cols;r++) for(let c=0;c<st.cols;c++){
      const x = st.x + c*cs, y = st.y + r*cs;
      const f = cellFill(st.key, r, c, P);
      if(f){ g.fillStyle = f; g.fillRect(x, y, cs, cs); }
      if(cs >= 7){ g.strokeStyle = P.line; g.lineWidth = 1; g.strokeRect(x+.5, y+.5, cs-1, cs-1); }
      if(!showNums) continue;
      // the number in a cell is ALWAYS its value — `showCounts` only repaints the background
      const v = cellValue(st.key, r, c);
      if(v === null || v === 0) continue;
      const txt = fmt(v);
      g.font = `600 ${Math.min(12, Math.max(7, Math.floor(cs*0.92/Math.max(txt.length,1.6))))}px ui-monospace,Menlo,monospace`;
      const isGrey = S.mode !== "matrix" && !S.showCounts && st.key !== "ker";
      g.fillStyle = isGrey ? (v/(MX[st.key] ?? 1) > 0.55 ? "#000" : "#fff") : P.fg;
      g.fillText(txt, x + cs/2, y + cs/2 + .5);
    }
  }

  if(isT() && D.valid){                                   // the crop rectangle
    const st = stage("can");
    g.strokeStyle = P.ok; g.lineWidth = 2;
    g.strokeRect(st.x + S.p*cs - 1, st.y + S.p*cs - 1, D.out*cs + 2, D.out*cs + 2);
  }

  g.fillStyle = P.dim2; g.textAlign = "center"; g.textBaseline = "middle";
  g.font = "17px ui-sans-serif,system-ui";
  g.fillText(isT() ? "×" : "⊛", S.n*cs + L.gOp/2, L.y1 + L.h1/2);
  g.fillText("→", D.ext*cs + L.gArrow/2, L.y2 + L.h2/2);
  g.font = "10px ui-sans-serif,system-ui";
  g.fillText(isT() ? "crop" : "slide",   D.ext*cs + L.gArrow/2, L.y2 + L.h2/2 + 16);
  g.fillText(isT() ? "−"+S.p : "by "+S.s, D.ext*cs + L.gArrow/2, L.y2 + L.h2/2 + 27);
  g.textAlign = "left"; g.textBaseline = "alphabetic";
}

/* ---------- what is relevant to the cell under the mouse ----------
   Two shapes of answer, and both layers have both:
     "scatter"  hovering an input pixel  → where does this number GO
     "gather"   hovering an output cell  → where does this number COME FROM
   `contrib` holds, per destination cell, the single product this source pixel
   puts there — that is what gets drawn in amber, so a stamp is never confused
   with the running total underneath it.
   ------------------------------------------------------------------ */
function relevance(){
  const f = S.focus;
  if(!f) return null;
  const key = (r,c) => r+","+c;
  const R = {in:new Set(), ker:new Set(), can:new Set(), out:new Set(),
             terms:[], contrib:new Map(), kind:null};
  const put = (stg,r,c,v) => R.contrib.set(stg+":"+r+","+c, v);

  if(isT()){
    if(f.stage === "in"){
      R.kind = "scatter"; R.src = {i:f.r, j:f.c};
      const v = IN[f.r*S.n + f.c];
      R.in.add(key(f.r, f.c));
      for(let a=0;a<S.k;a++) for(let b=0;b<S.k;b++){
        const w = W[a*S.k+b], cr = f.r*S.s + a*S.d, cc = f.c*S.s + b*S.d;
        R.ker.add(key(a,b)); R.can.add(key(cr,cc)); put("can", cr, cc, v*w);
        if(D.valid && cr>=S.p && cc>=S.p && cr-S.p<D.out && cc-S.p<D.out){
          R.out.add(key(cr-S.p, cc-S.p)); put("out", cr-S.p, cc-S.p, v*w);
        }
        R.terms.push({a, b, w, v, R:cr, C:cc});
      }
      return R;
    }
    R.kind = "gather";
    const cr = f.stage === "out" ? f.r + S.p : f.r;
    const cc = f.stage === "out" ? f.c + S.p : f.c;
    R.can.add(key(cr,cc));
    if(D.valid && cr>=S.p && cc>=S.p && cr-S.p<D.out && cc-S.p<D.out) R.out.add(key(cr-S.p, cc-S.p));
    R.terms = termsFor(cr, cc);
    for(const t of R.terms){ R.in.add(key(t.i,t.j)); R.ker.add(key(t.a,t.b)); }
    R.dst = {R:cr, C:cc};
    return R;
  }

  /* ---- plain convolution ---- */
  if(f.stage === "out"){
    R.kind = "gather"; R.dst = {r:f.r, c:f.c};
    R.out.add(key(f.r, f.c));
    for(let a=0;a<S.k;a++) for(let b=0;b<S.k;b++){
      const cr = f.r*S.s + a*S.d, cc = f.c*S.s + b*S.d;
      const i = cr - S.p, j = cc - S.p;
      const inside = i>=0 && j>=0 && i<S.n && j<S.n;
      R.ker.add(key(a,b)); R.can.add(key(cr,cc));
      if(inside) R.in.add(key(i,j));
      R.terms.push({a, b, w:W[a*S.k+b], v: inside ? IN[i*S.n+j] : 0, i, j, R:cr, C:cc, pad:!inside});
    }
    return R;
  }

  // input pixel (or a padding cell of the middle grid) — where does it go
  const i = f.stage === "in" ? f.r : f.r - S.p;
  const j = f.stage === "in" ? f.c : f.c - S.p;
  const inside = i>=0 && j>=0 && i<S.n && j<S.n;
  const cr = i + S.p, cc = j + S.p;
  const v = inside ? IN[i*S.n+j] : 0;
  R.kind = "scatter"; R.src = {i, j, cr, cc, pad:!inside};
  R.can.add(key(cr, cc));
  if(inside) R.in.add(key(i, j));
  for(const A of (D.contrib[cr] || [])) for(const B of (D.contrib[cc] || [])){
    const w = W[A.tap*S.k + B.tap];
    R.ker.add(key(A.tap, B.tap)); R.out.add(key(A.i, B.i));
    put("out", A.i, B.i, v*w);
    R.terms.push({a:A.tap, b:B.tap, w, v, R:A.i, C:B.i});
  }
  return R;
}

function paint(){
  const ctx = fitCanvas(el("viz"), L.w, L.h);
  const P = palette(), cs = L.cs, dpr = window.devicePixelRatio || 1;
  ctx.clearRect(0,0,L.w,L.h);
  ctx.drawImage(baseCv, 0, 0, L.w, L.h);

  const rel = relevance();
  renderMath(rel);
  el("vizcap").innerHTML = rel ? captionFor(rel) : "";
  if(!rel) return;

  // grey out everything, then punch the relevant cells back through
  ctx.fillStyle = P.veil;
  for(const st of L.stages){
    if(st.cols > 0) ctx.fillRect(st.x, st.y, st.cols*cs, st.cols*cs);
  }
  const inBox = (st,r,c) => st && st.cols>0 && r>=0 && c>=0 && r<st.cols && c<st.cols;
  const restore = (st, r, c) => {
    if(!inBox(st,r,c)) return;
    const x = st.x + c*cs, y = st.y + r*cs;
    ctx.drawImage(baseCv, x*dpr, y*dpr, cs*dpr, cs*dpr, x, y, cs, cs);
  };
  const outline = (st, r, c, col, wgt) => {
    if(!inBox(st,r,c)) return;
    ctx.strokeStyle = col; ctx.lineWidth = wgt;
    ctx.strokeRect(st.x + c*cs + .5, st.y + r*cs + .5, cs-1, cs-1);
  };
  const each = (set, fn) => set.forEach(k => { const [r,c] = k.split(",").map(Number); fn(r,c); });

  each(rel.in,  (r,c) => { restore(stage("in"),  r, c); outline(stage("in"),  r, c, P.amber, 1.5); });
  each(rel.ker, (r,c) => { restore(stage("ker"), r, c); outline(stage("ker"), r, c, P.accent, 1.5); });
  each(rel.can, (r,c) => restore(stage("can"), r, c));
  each(rel.out, (r,c) => restore(stage("out"), r, c));

  // ---- the one thing that makes a scatter readable: show the CONTRIBUTION,
  //      not the accumulated cell value, wherever this pixel writes.
  if(rel.contrib.size){
    let cmax = 0;
    rel.contrib.forEach(v => cmax = Math.max(cmax, Math.abs(v)));
    rel.contrib.forEach((val, kk) => {
      const [stg, rc] = kk.split(":");
      const [r, c] = rc.split(",").map(Number);
      const st = stage(stg);
      if(!inBox(st,r,c)) return;
      const x = st.x + c*cs, y = st.y + r*cs;
      const t = Math.min(1, Math.abs(val)/(cmax || 1));
      ctx.fillStyle = P.panel; ctx.fillRect(x, y, cs, cs);
      ctx.fillStyle = `rgba(${P.amberRGB},${(0.12 + 0.62*t).toFixed(3)})`;
      ctx.fillRect(x, y, cs, cs);
      ctx.strokeStyle = P.amber; ctx.lineWidth = 1.3;
      ctx.strokeRect(x + .5, y + .5, cs-1, cs-1);
      if(cs >= 15){
        const txt = fmt(val);
        ctx.font = `600 ${Math.min(12, Math.max(7, Math.floor(cs*0.92/Math.max(txt.length,1.6))))}px ui-monospace,Menlo,monospace`;
        ctx.fillStyle = P.fg; ctx.textAlign = "center"; ctx.textBaseline = "middle";
        ctx.fillText(txt, x + cs/2, y + cs/2 + .5);
        ctx.textAlign = "left"; ctx.textBaseline = "alphabetic";
      }
    });
  }

  // the cell being asked about gets the strong outline
  if(rel.kind === "scatter"){
    if(isT()) outline(stage("in"), rel.src.i, rel.src.j, P.fg, 2.5);
    else { outline(stage("in"), rel.src.i, rel.src.j, P.fg, 2.5);
           outline(stage("can"), rel.src.i + S.p, rel.src.j + S.p, P.fg, 2.5); }
  } else if(isT()){
    outline(stage("can"), rel.dst.R, rel.dst.C, P.fg, 2.5);
    each(rel.out, (r,c) => outline(stage("out"), r, c, P.fg, 2.5));
  } else {
    outline(stage("out"), rel.dst.r, rel.dst.c, P.fg, 2.5);
    each(rel.can, (r,c) => outline(stage("can"), r, c, P.amber, 1.5));
    // the whole window, dilation holes included
    const st = stage("can");
    ctx.strokeStyle = P.amber; ctx.lineWidth = 1.5; ctx.setLineDash([4,3]);
    ctx.strokeRect(st.x + rel.dst.c*S.s*cs - 1.5, st.y + rel.dst.r*S.s*cs - 1.5,
                   D.span*cs + 3, D.span*cs + 3);
    ctx.setLineDash([]);
  }

  // hovering a kernel weight isolates the single term that uses it
  if(S.tap){
    const t = rel.terms.find(t => t.a === S.tap.a && t.b === S.tap.b);
    outline(stage("ker"), S.tap.a, S.tap.b, P.fg, 2.5);
    if(t){
      if(rel.kind === "scatter") outline(stage(isT() ? "can" : "out"), t.R, t.C, P.fg, 2.5);
      else if(isT())             outline(stage("in"), t.i, t.j, P.fg, 2.5);
      else                       outline(stage("can"), t.R, t.C, P.fg, 2.5);
    }
  }

  if(isT() && D.valid){                                   // keep the crop box readable
    ctx.globalAlpha = .5; ctx.strokeStyle = P.ok; ctx.lineWidth = 2;
    const st = stage("can");
    ctx.strokeRect(st.x + S.p*cs - 1, st.y + S.p*cs - 1, D.out*cs + 2, D.out*cs + 2);
    ctx.globalAlpha = 1;
  }

}

// one line of plain English under the grids, so the amber is never a mystery
function captionFor(rel){
  const A = t => `<b style="color:var(--accent2);font-weight:600">${t}</b>`;
  if(rel.kind === "scatter"){
    if(rel.src.pad)
      return `This cell is ${A("zero padding")} — the window reads it like any other pixel,
              but 0·w = 0, so it contributes nothing anywhere.`;
    const at = `<b>input[${rel.src.i}][${rel.src.j}]</b>`;
    return isT()
      ? `${A("Amber")} = what ${at} <em>alone</em> adds to each of those cells. The grey numbers
         underneath are the running totals of all ${S.n*S.n} stamps, so they are normally
         <em>bigger</em> — tick <em>build up progressively</em> to watch them fill in one stamp at a time.`
      : `${A("Amber")} = ${at}'s <em>share</em> of each of those output cells. The rest of each of
         those sums comes from its ${D.span*D.span - 1} neighbours in the same window.`;
  }
  return isT()
    ? `${A("Amber")} = the input pixels whose stamps reached this one cell, blue = the weight each of
       them used. Every other pixel's stamp missed it.`
    : `The ${A("dashed box")} is the ${S.k}×${S.k} window (spanning ${D.span}×${D.span} pixels) that this
       single output number is the dot product of.`;
}

/* ======================= the formula, written out ======================= */
const iv = v => `<span class="iv">${fmt(v)}</span>`;
const wv = v => `<span class="wv">${fmt(v)}</span>`;
function row(label, html){ return `<div class="mrow"><span class="mlbl">${label}</span><span class="mval">${html}</span></div>`; }

function renderMath(rel){
  if(!rel){ el("readout").innerHTML = `<div class="mrow" style="color:var(--dim2)">
      Hover a number — in the input, the kernel, the middle grid or the output — to see the maths behind it.</div>`; return; }
  const chipOn = x => S.tap && S.tap.a === x.a && S.tap.b === x.b;
  const chip = (x, body, cls) => `<span class="term${chipOn(x)?' on':''}${cls?' '+cls:''}">${body}</span>`;

  /* ------------- scatter: where does this input pixel go ------------- */
  if(rel.kind === "scatter"){
    const {i,j} = rel.src;
    if(rel.src.pad){
      el("readout").innerHTML =
        row("this cell", `<b>padded_input[${rel.src.cr}][${rel.src.cc}]</b> — a zero added by
             <span class="p">padding ${S.p}</span>, not part of the image`) +
        row("maths", `<span style="color:var(--dim2)">The zeros <span class="p">padding</span> adds are read by
             the window just like real pixels, but 0·w = 0, so they only shrink nothing and shift everything.
             They are why the border of the output exists at all.</span>`);
      return;
    }
    const v = IN[i*S.n+j];
    const chips = rel.terms.map(t => isT()
      ? chip(t, `canvas[${t.R}][${t.C}] += ${iv(v)}·${wv(t.w)} = <b>${fmt(v*t.w)}</b>`)
      : chip(t, `output[${t.R}][${t.C}] += ${iv(v)}·${wv(t.w)} = <b>${fmt(v*t.w)}</b>`));
    const shown = chips.slice(0,9).join(" ") + (chips.length>9 ? ` <span class="more">+ ${chips.length-9} more</span>` : "");

    if(isT()){
      const r0 = i*S.s, c0 = j*S.s, r1 = r0+D.span-1, c1 = c0+D.span-1;
      el("readout").innerHTML =
        row("this cell", `<b>input[${i}][${j}] = ${iv(v)}</b> — it is <em>scattered</em>, not gathered`) +
        row("formula",   `canvas[${i}·<span class="s">${S.s}</span>+a·<span class="d">${S.d}</span>][${j}·<span class="s">${S.s}</span>+b·<span class="d">${S.d}</span>] &nbsp;+=&nbsp; input[${i}][${j}] · w[a][b] &nbsp; for all ${S.k}×${S.k} taps`) +
        row("maths",     shown) +
        row("lands on",  D.valid
            ? `canvas rows <b>${r0}…${r1}</b>, cols <b>${c0}…${c1}</b> &nbsp;→&nbsp; output rows <b>${r0-S.p}…${r1-S.p}</b>, cols <b>${c0-S.p}…${c1-S.p}</b>`
              + (r0-S.p<0||c0-S.p<0 ? ` <span class="err">— partly cropped away</span>` : "")
            : `<span class="err">no valid output</span>`);
      return;
    }

    const rs = rel.terms.map(t => t.R), cs2 = rel.terms.map(t => t.C);
    el("readout").innerHTML =
      row("this cell", `<b>input[${i}][${j}] = ${iv(v)}</b> — one pixel, <em>read</em> by ${rel.terms.length} output cell${rel.terms.length===1?"":"s"}`) +
      row("formula",   `output[R][C] &nbsp;+=&nbsp; input[${i}][${j}] · w[a][b] &nbsp; whenever
                        R·<span class="s">${S.s}</span>+a·<span class="d">${S.d}</span> = ${i}+<span class="p">${S.p}</span>`) +
      row("maths",     shown || `<span class="err">no window covers it</span>`) +
      row("read by",   rel.terms.length
          ? `output rows <b>${Math.min(...rs)}…${Math.max(...rs)}</b>, cols <b>${Math.min(...cs2)}…${Math.max(...cs2)}</b>
             — ${rel.terms.length} of the ${S.k*S.k} taps, because the rest of the windows fall off the image.`
          : `<span class="err">this pixel is never read</span> — with <span class="s">stride ${S.s}</span> wider than the
             window span ${D.span}, entire rows and columns are skipped and their information is lost.`);
    return;
  }

  /* ------------- gather: what is this output cell made of ------------- */
  if(!isT()){
    const {r,c} = rel.dst;
    const t = rel.terms;
    const total = t.reduce((a,x) => a + x.v*x.w, 0);
    const pads = t.filter(x => x.pad).length;
    const numeric  = t.map(x => chip(x, `${iv(x.v)}·${wv(x.w)}`, x.pad ? "pad" : "")).join(" + ");
    const products = t.map(x => chip(x, fmt(x.v*x.w), x.pad ? "pad" : "")).join(" + ") || "0";
    el("readout").innerHTML =
      row("this cell", `<b>output[${r}][${c}]</b>`) +
      row("formula",   `Σ<sub>a,b</sub> input[${r}·<span class="s">${S.s}</span>+a·<span class="d">${S.d}</span>−<span class="p">${S.p}</span>][${c}·<span class="s">${S.s}</span>+b·<span class="d">${S.d}</span>−<span class="p">${S.p}</span>] · w[a][b]`) +
      row("values",    numeric) +
      row("maths",     `${products} &nbsp;=&nbsp; <span class="res">${fmt(total)}</span>`) +
      row("why these", `the window's top-left corner sits on input[${r*S.s-S.p}][${c*S.s-S.p}] and covers
         ${D.span}×${D.span} pixels${S.d>1?` with <span class="d">dilation ${S.d}</span> holes`:""}.
         ${pads ? `<span style="color:var(--dim2)">${pads} of the ${S.k*S.k} taps land on the zero
          <span class="p">padding</span> and contribute nothing.</span>` : "Every tap lands inside the image."}`);
    return;
  }

  const {R, C} = rel.dst;
  const beyond = R >= D.full || C >= D.full;
  const inCrop = D.valid && R>=S.p && C>=S.p && R-S.p<D.out && C-S.p<D.out;
  const where = inCrop ? `<b>output[${R-S.p}][${C-S.p}]</b> &nbsp;=&nbsp; canvas[${R}][${C}]`
                       : `canvas[${R}][${C}] <span class="err">— cropped away by padding, never reaches the output</span>`;

  if(beyond){
    el("readout").innerHTML = row("this cell", where) +
      row("maths", `<span style="color:var(--dim2)">outside the canvas → exactly <b>0</b>. These are the rows/cols
           <span class="op">output_padding</span> tacks on; nothing ever stamps here.</span>`);
    return;
  }

  const t = rel.terms;
  const total = t.reduce((a,x) => a + x.v*x.w, 0);
  const symbolic = t.map(x => chip(x, `input[${x.i}][${x.j}]·w[${x.a}][${x.b}]`)).join(" + ") || "—";
  const numeric  = t.map(x => chip(x, `${iv(x.v)}·${wv(x.w)}`)).join(" + ") || "—";
  const products = t.map(x => chip(x, fmt(x.v*x.w))).join(" + ") || "0";

  el("readout").innerHTML =
    row("this cell", where) +
    row("formula",   symbolic) +
    row("values",    numeric) +
    row("maths",     `${products} &nbsp;=&nbsp; <span class="res">${fmt(total)}</span>`) +
    row("why these", t.length
      ? `only these ${t.length} input pixel${t.length===1?"":"s"} have a stamp — anchored at
         i·<span class="s">${S.s}</span> — that covers row ${R}, col ${C}. Every other one misses it.`
      : `no input pixel's stamp reaches row ${R}, col ${C}, so it stays <b>0</b>.`);
}

function renderCountScale(){
  const box = el("countscale");
  if(!S.showCounts || !D.valid){ box.innerHTML = ""; return; }
  const swatches = [];
  for(let n=D.min; n<=D.max && swatches.length<8; n++)
    swatches.push(`<span style="display:inline-flex;align-items:center;gap:3px;margin-right:7px">
      <i style="width:13px;height:13px;border-radius:3px;border:1px solid var(--line);
                background:${heat(n, D.max||1) || "transparent"}"></i>${n}</span>`);
  box.innerHTML = `<b style="color:var(--fg)">${isT() ? "terms summed per output cell" : "times each pixel is read"}</b><br>${swatches.join("")}`;
}

/* ============================== side panels ============================== */
function renderFormula(){
  const {k,s,p,opad,d} = S;
  const h = calcH(), w = calcW();
  const A = axisStats(h), B = (w === h) ? A : axisStats(w);

  const subT = (hin, r) =>
    `(${hin}−1)·<span class="s">${s}</span> − 2·<span class="p">${p}</span> + <span class="d">${d}</span>·(<span class="k">${k}</span>−1) + <span class="op">${opad}</span> + 1
     = <span class="big ${r.valid?'':'err'}">${Math.max(r.out,0)}</span>`;
  const subC = (hin, r) =>
    `⌊(${hin} + 2·<span class="p">${p}</span> − <span class="d">${d}</span>·(<span class="k">${k}</span>−1) − 1) / <span class="s">${s}</span>⌋ + 1
     = <span class="big ${r.valid?'':'err'}">${Math.max(r.out,0)}</span>`;
  const sub = isT() ? subT : subC;

  el("formula").innerHTML =
    `<span class="lbl">pytorch</span>${isT()
        ? `ConvTranspose2d(C, C', ${k}, stride=${s}, padding=${p}${opad?`, output_padding=${opad}`:""}${d>1?`, dilation=${d}`:""})`
        : `Conv2d(C, C', ${k}, stride=${s}, padding=${p}${d>1?`, dilation=${d}`:""})`}
     <span class="lbl">shape</span>(N, C, <b>${h}</b>, <b>${w}</b>) → (N, C', <b>${Math.max(A.out,0)}</b>, <b>${Math.max(B.out,0)}</b>)
     <span class="lbl">formula</span>${isT()
        ? `(H−1)·<span class="s">s</span> − 2·<span class="p">p</span> + <span class="d">d</span>·(<span class="k">k</span>−1) + <span class="op">op</span> + 1`
        : `⌊(H + 2·<span class="p">p</span> − <span class="d">d</span>·(<span class="k">k</span>−1) − 1) / <span class="s">s</span>⌋ + 1`}
     <span class="lbl">${w===h?"substituted":"height"}</span>${sub(h, A)}` +
    (w===h ? "" : `<span class="lbl">width</span>${sub(w, B)}`);

  const tail = calcLinked() ? "" :
    `<br><br><span style="color:var(--dim2)">The grids show <b>${S.n}×${S.n}</b> — too big or not square to draw
     cell by cell, so these numbers are for <b>${h}×${w}</b> only. <em>↺ grid</em> puts them back in sync.</span>`;

  if(isT()){
    el("shapenote").innerHTML = (A.valid && B.valid
      ? `The canvas before cropping is <b>${A.full}${w===h?"×"+A.full:" high"}</b>: ${h} stamps spaced by ${s}
         span ${(h-1)*s}, plus one stamp width ${A.span}. Cropping <span class="p">${p}</span> per side and adding
         <span class="op">${opad}</span> back on bottom/right leaves <b>${A.out}</b>${w===h?"":" rows; the width works the same way"}.`
      : `<span class="err">Invalid: padding removes more than the canvas holds
         (${A.full} − 2·${p} + ${opad} ≤ 0).</span> Lower <span class="p">padding</span>.`) + tail;
  } else {
    el("shapenote").innerHTML = (A.valid && B.valid
      ? `The input is padded to <b>${A.full}</b>. The window spans <span class="d">${d}</span>·(<span class="k">${k}</span>−1)+1
         = <b>${A.span}</b>, so its top-left corner has ${A.full - A.span + 1} legal positions, and it takes every
         <span class="s">${s}</span>-th one → <b>${A.out}</b>.` +
        (A.drop ? ` The division does not come out even: the last <b>${A.drop}</b> row(s)/col(s) are never covered
                    by any window — that is what the ⌊floor⌋ silently throws away.` : ``)
      : `<span class="err">Invalid: the ${A.span}×${A.span} window does not fit inside the padded
         ${A.full}×${A.full} input.</span> Raise <span class="p">padding</span> or lower
         <span class="k">kernel_size</span>.`) + tail;
  }
}

function renderDiag(){
  const {k,s,d,p,opad,n} = S;
  if(isT()){
    const even = (s === 1) || (k % s === 0);
    const uniform = D.valid && D.min === D.max;
    el("diag").innerHTML = `
      <div style="margin-bottom:7px">
        <span class="flag ${even?'ok':'bad'}">${even?'even overlap':'uneven overlap'}</span>
        <span style="color:var(--dim);font-size:11.5px;margin-left:7px">
          <span class="k">k</span> % <span class="s">s</span> = ${k%s}</span>
      </div>
      <div class="note" style="margin-top:0">
        ${even
          ? `Every interior output position receives the same number of stamps, so no periodic pattern is baked in.`
          : `<span class="k">kernel_size</span> is not a multiple of <span class="s">stride</span>: some output
             pixels sum more terms than their neighbours, in a repeating pattern — the classic
             <b>checkerboard artifact</b>. Turn on <em>colour by number of terms</em> to see it.
             Fix with <span class="k">k</span> = ${s*Math.max(1,Math.round(k/s))}, or upsample + plain conv.`}
        <br><br>
        Contributions per output pixel: <b>${D.min}…${D.max}</b>${uniform?" (uniform)":""}.
        Borders always get fewer — that part is normal.
        ${!D.opOk ? `<br><br><span class="err">output_padding must be &lt; max(stride, dilation) = ${Math.max(s,d)};
                     PyTorch would raise here.</span>` : ""}
        ${opad>p ? `<br><br><span style="color:var(--dim2)">output_padding (${opad}) exceeds padding (${p}), so the last
                  ${opad-p} row(s)/col(s) of the output fall outside the canvas and are exactly zero.</span>` : ""}
      </div>`;
    return;
  }

  const same = D.valid && D.out === n && s === 1;
  const skips = D.min === 0;
  const factor = D.valid ? (n/D.out) : 0;
  const notes = [];
  notes.push(same
    ? `Output size = input size: <span class="p">padding</span> ${p} exactly compensates a span of ${D.span}
       (the “same” convolution — it needs p = d·(k−1)/2, so it only works for odd effective kernels).`
    : `${D.valid ? `${n}×${n} → ${D.out}×${D.out}, a factor of <b>${factor % 1 ? factor.toFixed(2) : factor}</b>.` : ""}
       ${s > 1 ? `<span class="s">stride ${s}</span> is doing the downsampling — the layer throws away
                  ${(100*(1-1/(s*s))).toFixed(0)}% of the positions.` : ""}`);
  notes.push(`Each output value is a sum of <b>${k*k}</b> products and sees a
     <b>${D.span}×${D.span}</b> patch of the input${d>1?` — <span class="d">dilation ${d}</span> buys that
     receptive field with the same ${k*k} weights, at the cost of the holes in between`:``}.`);
  notes.push(skips
    ? `<span class="err">Some input pixels are read 0 times.</span> stride ${s} exceeds the window span ${D.span},
       so whole rows and columns never reach the output.`
    : `Each pixel is read <b>${D.min}…${D.max}</b> times. Border pixels are read fewer times than interior ones —
       that asymmetry is exactly what <span class="p">padding</span> softens.`);
  if(p > D.span/2)
    notes.push(`<span style="color:var(--dim2)">padding ${p} is larger than half the window span, so some output
       positions see only zeros. PyTorch allows it; it is rarely what you want.</span>`);

  el("diag").innerHTML = `
    <div style="margin-bottom:7px">
      <span class="flag ${skips?'bad':'ok'}">${skips ? 'pixels skipped' : same ? 'size preserved' : s>1 ? `downsample ×${s}` : 'full coverage'}</span>
      <span style="color:var(--dim);font-size:11.5px;margin-left:7px">receptive field ${D.span}×${D.span}</span>
    </div>
    <div class="note" style="margin-top:0">${notes.join("<br><br>")}</div>`;
}

/* ============================== controls ============================== */
function buildControls(){
  el("ctrls").innerHTML = PARAMS.map(P => `
    <div class="ctrl" id="ctrl_${P.id}">
      <label for="r_${P.id}"><span class="${P.cls||''}">${P.name}</span><b id="v_${P.id}"></b></label>
      <input type="range" id="r_${P.id}" min="${P.min}" max="${P.max}" step="1">
      <div class="hint" id="h_${P.id}"></div>
    </div>`).join("");

  PARAMS.forEach(P => el("r_"+P.id).addEventListener("input", e => {
    S[P.id] = +e.target.value;      // no cross-clamping: invalid combos are explained in Diagnostics
    S.tap = null;
    refresh();
  }));

  ["ch","cw"].forEach(id => el(id).addEventListener("change", () =>
    setCalc(+el("ch").value, +el("cw").value)));
  el("csync").addEventListener("click", () => { relink(); refresh(); });

  document.querySelectorAll("#kernelseg button").forEach(b => b.addEventListener("click", () => {
    S.kernel = b.dataset.kernel; refresh();
  }));
  document.querySelectorAll("#modeseg button").forEach(b =>
    b.addEventListener("click", () => setMode(b.dataset.mode)));

  el("showcounts").addEventListener("change", e => { S.showCounts = e.target.checked; refresh(); });
  el("prog").addEventListener("change", e => { S.progressive = e.target.checked; refresh(); });
  el("prev").addEventListener("click", () => step(-1));
  el("next").addEventListener("click", () => step(+1));
  el("play").addEventListener("click", play);
}

function presetOut(q, hin){
  const span = q.d*(q.k-1)+1;
  return isT() ? (hin-1)*q.s + span - 2*q.p + (q.opad||0)
               : Math.floor((hin + 2*q.p - span)/q.s) + 1;
}
function renderPresets(){
  const list = PRESETS[S.layer];
  el("presethead").textContent = `Presets — real ${isT() ? "generators" : "backbones"}`;
  el("presets").innerHTML = list.map((P,i) => {
    const q = P.q;
    return `<button data-i="${i}"><b>${P.t}</b><i>k${q.k} s${q.s} p${q.p}${q.opad?` op${q.opad}`:""}${q.d>1?` d${q.d}`:""}
            · ${P.in}² → ${presetOut(q, P.in)}²</i></button>`;
  }).join("");
  el("presets").querySelectorAll("button").forEach(b => b.addEventListener("click", () => {
    const P = list[+b.dataset.i];
    Object.assign(S, {opad:0}, P.q);
    if(S.mode === "matrix") S.n = Math.max(1, Math.min(P.n, 12));
    // the calculator carries the architecture's real input size, unless the
    // grid already happens to be that size
    if(P.in === S.n) relink(); else S.calc = {h:P.in, w:P.in};
    S.focus = null; S.tap = null;
    refresh(true);
  }));
}

function relink(){ S.calc = {h:null, w:null}; }

// a size the grids can actually draw is applied to them; anything else stays
// in the calculator alone.
function setCalc(h, w){
  const cl = v => Math.max(1, Math.min(4096, Math.round(v) || 1));
  h = cl(h); w = cl(w);
  const drawable = h === w &&
    (S.mode === "matrix" ? (h >= 1 && h <= 12) : [8,14,28].includes(h));
  if(drawable){
    relink();
    if(S.mode !== "matrix") S.res = h;
    S.n = h; S.focus = null; refresh(true);
  } else {
    S.calc = {h, w}; refresh();
  }
}

function step(dir){
  stop();
  const m = Math.max(stepN(), 1), total = m*m;
  const t = (selIdx() + dir + total) % total;
  S.sel = {i:(t/m)|0, j:t%m};
  S.focus = {stage:stepStage(), r:S.sel.i, c:S.sel.j}; S.tap = null;
  refresh();
}

let timer = null;
function stop(){ clearInterval(timer); timer = null; el("play").textContent = "▶ play"; el("play").classList.remove("on"); }
function play(){
  if(timer){ stop(); return; }
  const m = Math.max(stepN(), 1), total = m*m;
  el("play").textContent = "⏸ pause"; el("play").classList.add("on");
  S.sel = {i:0, j:0}; S.focus = {stage:stepStage(), r:0, c:0}; refresh();
  timer = setInterval(() => {
    const t = selIdx() + 1;
    if(t >= total){ stop(); return; }
    S.sel = {i:(t/m)|0, j:t%m};
    S.focus = {stage:stepStage(), r:S.sel.i, c:S.sel.j};
    refresh();
  }, Math.max(16, Math.round(7000/total)));
}

function setMode(m){
  stop();
  S.mode = m;
  S.n = m === "matrix" ? Math.min(S.n, 12) : S.res;
  S.focus = null; S.tap = null; relink();
  refresh(true);
}

function renderSubbar(){
  const b = el("subbar");
  if(S.mode === "matrix"){
    b.innerHTML = `<span>size</span><div class="seg" id="sizeseg">` +
      [1,2,3,4,5,6,8,10,12].map(v => `<button data-size="${v}" class="${v===S.n?'on':''}">${v}</button>`).join("") +
      `</div><button id="reroll">↻ new values</button><span>random integers 0–5</span>`;
  } else if(S.mode === "mnist"){
    b.innerHTML = `<span>digit</span><div class="seg" id="digseg">` +
      DIGITS.map((dg,i) => `<button data-dig="${i}" class="${i===S.digit?'on':''}">${dg.label}</button>`).join("") +
      `</div><button id="nextdig">next ›</button>` + resHTML();
  } else {
    b.innerHTML = `<button id="pick">upload image…</button>
      <span>or drop one anywhere on this panel</span>` + resHTML() +
      `<input type="file" id="file" accept="image/*" hidden>`;
  }

  if(el("reroll")) el("reroll").addEventListener("click", () => { S.matSeed++; refresh(true); });
  document.querySelectorAll("#sizeseg button").forEach(x =>
    x.addEventListener("click", () => { S.n = +x.dataset.size; S.focus = null; relink(); refresh(true); }));
  const dseg = el("digseg");
  if(dseg) dseg.querySelectorAll("button").forEach(x =>
    x.addEventListener("click", () => { S.digit = +x.dataset.dig; refresh(true); }));
  if(el("nextdig")) el("nextdig").addEventListener("click", () => {
    S.digit = (S.digit+1) % DIGITS.length; refresh(true); });
  if(el("pick")) el("pick").addEventListener("click", () => el("file").click());
  if(el("file")) el("file").addEventListener("change", e => {
    if(e.target.files && e.target.files[0]) loadPhoto(e.target.files[0]); });
  document.querySelectorAll("#resseg button").forEach(x =>
    x.addEventListener("click", () => { S.res = +x.dataset.res; S.n = S.res; relink(); refresh(true); }));

  el("credit").innerHTML = S.mode === "photo" && PHOTO.credit
    ? `Default photo: Yann LeCun by Jérémy Barande / Institut Polytechnique de Paris,
       <a href="https://creativecommons.org/licenses/by-sa/2.0" target="_blank" rel="noopener">CC BY-SA 2.0</a>,
       via Wikimedia Commons — cropped and converted to greyscale.`
    : S.mode === "mnist"
    ? `Digits: 4 samples from the MNIST test set (Yann LeCun &amp; Corinna Cortes), bundled as raw bytes.`
    : "";
}
function resHTML(){
  return `<span style="margin-left:6px">resolution</span><div class="seg" id="resseg">` +
    [8,14,28].map(r => `<button data-res="${r}" class="${r===S.res?'on':''}">${r}</button>`).join("") + `</div>`;
}

function loadPhoto(file){
  const fr = new FileReader();
  fr.onload = () => {
    const img = new Image();
    img.onload = () => {
      const cv = document.createElement("canvas");
      cv.width = cv.height = PHOTO_N;
      const g = cv.getContext("2d");
      const side = Math.min(img.width, img.height);
      g.drawImage(img, (img.width-side)/2, (img.height-side)/2, side, side, 0, 0, PHOTO_N, PHOTO_N);
      const px = g.getImageData(0,0,PHOTO_N,PHOTO_N).data;
      const grey = new Uint8Array(PHOTO_N*PHOTO_N);
      for(let i=0;i<grey.length;i++)
        grey[i] = (0.299*px[i*4] + 0.587*px[i*4+1] + 0.114*px[i*4+2])|0;
      PHOTO = {px:grey, n:PHOTO_N, credit:false};
      S.mode = "photo"; S.n = S.res; S.focus = null; refresh(true);
    };
    img.src = fr.result;
  };
  fr.readAsDataURL(file);
}

function renderHeader(){
  el("cv_title").textContent = isT() ? "ConvTranspose2d, pixel by pixel" : "Conv2d, pixel by pixel";
  el("cv_sub").innerHTML = isT()
    ? `A transposed convolution does <em>not</em> slide a kernel over its input. It does the opposite:
       every <b>input</b> pixel <b>stamps a copy of the kernel</b> into a bigger canvas, at spacing
       <span class="s">stride</span>. Overlapping stamps add up, then <span class="p">padding</span>
       <b>crops</b> the result.
       <b>Hover any number</b> and the page works out that number, greying out everything not involved in it.`
    : `A convolution <b>slides a <span class="k">k×k</span> window</b> over the zero-<span class="p">padded</span>
       input, jumping <span class="s">stride</span> pixels at a time, and writes one number per position: the sum
       of window × kernel. <b>Hover an output cell</b> to see the window it came from, or <b>hover an input pixel</b>
       to see every output cell that reads it — the same arithmetic, read backwards.`;

  el("legend").innerHTML = [
    `<span><span class="sw" style="border:2px solid var(--accent2)"></span>input pixels involved</span>`,
    `<span><span class="sw" style="border:2px solid var(--accent)"></span>kernel weights involved</span>`,
    `<span><span class="sw" style="background:color-mix(in srgb,var(--accent2) 45%,transparent);border:1px solid var(--accent2)"></span>this pixel's contribution</span>`,
    `<span><span class="sw" style="background:var(--panel2);border:1px solid var(--line);opacity:.6"></span>greyed = not part of this sum</span>`,
    isT() ? `<span><span class="sw" style="border:2px solid var(--ok)"></span>kept after <span class="p">padding</span> crop</span>` : "",
    `<span><span class="sw" style="background:#dbe2ea;border:1px solid var(--line)"></span>${isT() ? "outside the canvas — exact zero" : "zero padding"}</span>`,
  ].join("");

  el("countlabel").textContent = isT() ? "colour by number of terms" : "colour by how often each pixel is read";
  el("counthint").innerHTML = isT()
    ? `Cell <em>numbers</em> never change — only the background. Pale = few products summed into that cell,
       strong = many. When it is uneven across the output you get checkerboarding.`
    : `Paints the padded input by how many windows read each pixel. Borders are read less; with a large
       <span class="s">stride</span> some pixels are read zero times and simply vanish from the output.`;
  el("proghint").textContent = isT() ? "only the stamps up to the current pixel"
                                     : "only the window positions up to the current one";
}

function syncControls(){
  PARAMS.forEach(P => {
    el("r_"+P.id).value = S[P.id];
    el("v_"+P.id).textContent = S[P.id];
    el("h_"+P.id).textContent = P.hint[S.layer];
    el("ctrl_"+P.id).style.display = (P.only && P.only !== S.layer) ? "none" : "";
  });
  el("showcounts").checked = S.showCounts;
  renderCountScale();
  el("ch").value = calcH(); el("cw").value = calcW();
  el("csync").style.display = calcLinked() ? "none" : "";
  el("prog").checked = S.progressive;
  const lbl = isT() ? "input" : "output";
  el("sellabel").textContent = `${lbl}[${S.sel.i}][${S.sel.j}]`;
  document.querySelectorAll("#modeseg button").forEach(b =>
    b.classList.toggle("on", b.dataset.mode === S.mode));
  document.querySelectorAll("#kernelseg button").forEach(b =>
    b.classList.toggle("on", b.dataset.kernel === S.kernel));
}

/* ============================== interaction ============================== */
function hit(ev){
  const rect = el("viz").getBoundingClientRect();
  const x = ev.clientX - rect.left, y = ev.clientY - rect.top, cs = L.cs;
  for(const st of L.stages){
    if(st.cols <= 0) continue;
    const c = Math.floor((x - st.x)/cs), r = Math.floor((y - st.y)/cs);
    if(c>=0 && r>=0 && c<st.cols && r<st.cols) return {stage:st.key, r, c};
  }
  return null;
}

function wire(){
  const cv = el("viz");
  cv.addEventListener("mousemove", e => {
    const h = hit(e);
    if(!h) return;                                        // between grids: keep the last reading
    if(h.stage === "ker"){                                // kernel hover only picks out one term
      const t = {a:h.r, b:h.c};
      if(!S.tap || S.tap.a !== t.a || S.tap.b !== t.b){ S.tap = t; paint(); }
      return;
    }
    S.tap = null;
    const f = S.focus;
    if(f && f.stage === h.stage && f.r === h.r && f.c === h.c) return;
    S.focus = h;
    if(h.stage === stepStage()){                          // also drives the progressive build-up
      stop(); S.sel = {i:h.r, j:h.c};
      if(S.progressive){ compute(); syncControls(); renderBase(); }
      else syncControls();
    }
    paint();
  });

  const st = el("stage");
  ["dragenter","dragover"].forEach(t => st.addEventListener(t, e => {
    e.preventDefault(); st.classList.add("drop"); }));
  ["dragleave","drop"].forEach(t => st.addEventListener(t, e => {
    e.preventDefault(); st.classList.remove("drop"); }));
  st.addEventListener("drop", e => {
    const f = e.dataTransfer.files && e.dataTransfer.files[0];
    if(f && f.type.startsWith("image/")) loadPhoto(f);
  });

  addEventListener("keydown", e => {
    if(document.getElementById("view-conv").classList.contains("hidden")) return;
    const m = Math.max(stepN(), 1);
    const map = {ArrowLeft:-1, ArrowRight:1, ArrowUp:-m, ArrowDown:m};
    if(!(e.key in map) || e.target.tagName === "INPUT") return;
    e.preventDefault(); step(map[e.key]);
  });

  let t; addEventListener("resize", () => { clearTimeout(t); t = setTimeout(() => {
    if(!document.getElementById("view-conv").classList.contains("hidden")) refresh();
  }, 120); });
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
    if(!document.getElementById("view-conv").classList.contains("hidden")) refresh();
  });
}

/* ================================ main ================================ */
// the most instructive cell to open on: for convT the output cell with the most
// contributors, for conv a window sitting fully inside the image
function bestExample(){
  if(!D.valid) return null;
  const mid = (D.out-1)/2;
  if(!isT()){
    // prefer a window sitting entirely inside the image, as close to the middle
    // as that allows — a border window full of padding zeros explains nothing
    const lo = Math.ceil(S.p/S.s), hi = Math.floor((S.p + S.n - D.span)/S.s);
    let r = Math.round(mid);
    if(hi >= lo) r = Math.max(lo, Math.min(hi, r));
    r = Math.max(0, Math.min(D.out-1, r));
    return {stage:"out", r, c:r};
  }
  let best = null, bestScore = -1;
  for(let r=0;r<D.out;r++) for(let c=0;c<D.out;c++){
    const R = r+S.p, C = c+S.p;
    if(R>=D.full || C>=D.full) continue;
    const score = CNT[R]*CNT[C]*100 - (Math.abs(r-mid) + Math.abs(c-mid));
    if(score > bestScore){ bestScore = score; best = {stage:"out", r, c}; }
  }
  return best;
}

function clampFocus(){
  const f = S.focus;
  if(!f) return;
  const st = stage(f.stage);
  if(!st || st.cols <= 0 || f.r >= st.cols || f.c >= st.cols) S.focus = null;
}

function refresh(rebuildInput){
  derive();
  buildKernel();
  if(rebuildInput || !IN || IN.length !== S.n*S.n) buildInput();
  const m = Math.max(stepN(), 1);
  S.sel = {i: Math.min(S.sel.i, m-1), j: Math.min(S.sel.j, m-1)};
  compute();
  syncControls();
  renderSubbar();
  renderFormula();
  renderDiag();
  layout();
  clampFocus();
  if(!S.focus && D.valid) S.focus = bestExample();         // open on a real worked example
  renderBase();
  paint();
}

buildControls();
wire();

return {
  show(layer){
    stop();
    if(layer !== S.layer){
      S.layer = layer;
      S.focus = null; S.tap = null; S.sel = {i:0, j:0};
      relink();
    }
    renderHeader();
    renderPresets();
    refresh(true);
  }
};
})();
