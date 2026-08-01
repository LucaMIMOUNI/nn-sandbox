/* ==================================================================
   The menu — Conv2d and ConvTranspose2d are the same page in two
   directions, Normalisation is its own.
   ================================================================== */
const TABS = document.getElementById("tabs");
const VIEW_CONV = document.getElementById("view-conv");
const VIEW_NORM = document.getElementById("view-norm");

function setView(v){
  VIEW_CONV.classList.toggle("hidden", v === "norm");
  VIEW_NORM.classList.toggle("hidden", v !== "norm");
  TABS.querySelectorAll("button").forEach(b => b.classList.toggle("on", b.dataset.view === v));
  if(v === "norm") NORM.show(); else CONV.show(v);
  try{ history.replaceState(null, "", "#"+v); }catch(e){}
}
TABS.querySelectorAll("button").forEach(b =>
  b.addEventListener("click", () => setView(b.dataset.view)));
addEventListener("hashchange", () => {
  const h = location.hash.slice(1);
  if(["conv","convT","norm"].includes(h)) setView(h);
});

const start = location.hash.slice(1);
setView(["conv","convT","norm"].includes(start) ? start : "conv");
