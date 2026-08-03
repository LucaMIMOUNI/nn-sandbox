/* ==================================================================
   The menu — Conv2d and ConvTranspose2d are the same page in two
   directions; Normalisation and Backprop are their own.
   ================================================================== */
const TABS = document.getElementById("tabs");
const VIEWS = {conv:"view-conv", convT:"view-conv", norm:"view-norm", backprop:"view-backprop"};
const NAMES = Object.keys(VIEWS);

function setView(v){
  const want = VIEWS[v];
  for(const id of new Set(Object.values(VIEWS)))
    document.getElementById(id).classList.toggle("hidden", id !== want);
  TABS.querySelectorAll("button").forEach(b => b.classList.toggle("on", b.dataset.view === v));

  if(v !== "backprop") BP.hide();               // stop the animations the moment they leave the screen
  if(v !== "norm")     NORM.hide();
  if(v === "norm")          NORM.show();
  else if(v === "backprop") BP.show();
  else                      CONV.show(v);
  try{ history.replaceState(null, "", "#" + v); }catch(e){}
}
TABS.querySelectorAll("button").forEach(b =>
  b.addEventListener("click", () => setView(b.dataset.view)));
addEventListener("hashchange", () => {
  const h = location.hash.slice(1);
  if(NAMES.includes(h)) setView(h);
});

const start = location.hash.slice(1);
setView(NAMES.includes(start) ? start : "conv");
