# -*- coding: utf-8 -*-
"""Build a self-contained interactive HTML dashboard."""
import os
import json
import numpy as np
import pandas as pd
from lib import ROOT, TAB, load_exe, to_sec, SECTOR

panel = pd.read_parquet(os.path.join(ROOT, "out", "snap_panel.parquet"),
                        columns=["Symbol", "sec", "mid", "rel_spread", "Price"])
panel["minute"] = (panel.sec // 60).astype(int)
last = panel.sort_values("sec").groupby(["Symbol", "minute"]).agg(
    mid=("mid", "last"), spread=("rel_spread", "mean")).reset_index()

exe = load_exe(["Symbol", "Time", "Volume", "Price", "BSFlag"])
exe["minute"] = (to_sec(exe.Time) // 60).astype(int)
vol = exe.groupby(["Symbol", "minute"]).agg(vol=("Volume", "sum"),
                                            amt=("Price", lambda x: 0)).reset_index().drop(columns="amt")
exe["amt"] = exe.Price * exe.Volume
amt = exe.groupby(["Symbol", "minute"]).amt.sum().reset_index()
vol = vol.merge(amt, on=["Symbol", "minute"])


def sess_index(minute):
    if minute <= 11 * 60 + 30:
        return minute - (9 * 60 + 30)
    if minute >= 13 * 60:
        return 120 + (minute - 13 * 60)
    return None


M = pd.read_csv(os.path.join(TAB, "12_master_table.csv"))
data = {}
for _, r in M.iterrows():
    s = int(r.Symbol)
    g = last[last.Symbol == s].copy()
    g["si"] = g.minute.apply(sess_index)
    g = g.dropna(subset=["si"])
    g = g[(g.si >= 0) & (g.si <= 240)]
    v = vol[vol.Symbol == s].copy()
    v["si"] = v.minute.apply(sess_index)
    v = v.dropna(subset=["si"])
    v = v[(v.si >= 0) & (v.si <= 240)]
    data[str(s)] = dict(
        name=str(r.Name), sector=SECTOR.get(s, "其他"),
        metric={k: (None if pd.isna(r[k]) else float(r[k])) for k in
                ["RelSpread_bp", "QuotedSpread_ticks", "EffSpread_bp", "RV_bp",
                 "DepthAmt_wan", "CancelRate", "AggrRate", "FleetingRate",
                 "VPIN", "Top1VolShare", "CloseAucAmtShare", "DayRet"]},
        si=[int(x) for x in g.si], mid=[round(float(x), 4) for x in g.mid],
        spread=[round(float(x) * 1e4, 3) for x in g.spread],
        vsi=[int(x) for x in v.si], vol=[int(x) for x in v.vol],
        amt=[round(float(x) / 1e4, 1) for x in v.amt])

payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))

FIGDIR = "figures"
figs = sorted(f for f in os.listdir(os.path.join(ROOT, "out", "figures")) if f.endswith(".png"))
fig_tags = "\n".join(
    f'<figure><img src="{FIGDIR}/{f}" loading="lazy"><figcaption>{f}</figcaption></figure>'
    for f in figs)

html = """<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>上证50 微观结构 Dashboard</title>
<style>
:root{--bg:#0d1117;--panel:#161b22;--fg:#e6edf3;--mut:#8b949e;--acc:#58a6ff;--red:#f85149;--grn:#3fb950}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font-family:"Microsoft YaHei",system-ui,sans-serif}
header{padding:18px 24px;border-bottom:1px solid #30363d;position:sticky;top:0;background:var(--bg);z-index:5}
h1{margin:0;font-size:19px}
.sub{color:var(--mut);font-size:12px;margin-top:4px}
.wrap{display:grid;grid-template-columns:360px 1fr;gap:16px;padding:16px 24px}
.card{background:var(--panel);border:1px solid #30363d;border-radius:10px;padding:12px}
table{border-collapse:collapse;width:100%;font-size:12px}
th,td{padding:4px 6px;text-align:right;border-bottom:1px solid #21262d;cursor:default;white-space:nowrap}
th{color:var(--mut);font-weight:600;cursor:pointer;position:sticky;top:0;background:var(--panel)}
td:first-child,th:first-child{text-align:left}
tr.sel{background:#1f6feb33}
tbody tr:hover{background:#30363d55;cursor:pointer}
.tblbox{max-height:640px;overflow:auto}
.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-bottom:10px}
.m{background:#0d1117;border:1px solid #30363d;border-radius:8px;padding:8px}
.m .k{color:var(--mut);font-size:11px}.m .v{font-size:16px;margin-top:2px}
canvas{width:100%;display:block}
.rowlab{color:var(--mut);font-size:12px;margin:8px 0 2px}
figure{margin:0 0 16px}
figure img{width:100%;border-radius:8px;border:1px solid #30363d;background:#fff}
figcaption{color:var(--mut);font-size:12px;margin-top:4px}
.gal{padding:0 24px 40px;display:grid;grid-template-columns:1fr 1fr;gap:18px}
select{background:#0d1117;color:var(--fg);border:1px solid #30363d;border-radius:6px;padding:6px;font-size:13px;width:100%}
</style></head><body>
<header><h1>上证50 · 逐笔微观结构 Dashboard</h1>
<div class="sub">2026-09-23 · 50 只 · 6.28M 委托 / 3.79M 成交 / 253K 快照 · 成交额 972.5 亿元</div></header>

<div class="wrap">
  <div class="card">
    <select id="pick"></select>
    <div class="tblbox" style="margin-top:10px">
      <table id="tbl"><thead></thead><tbody></tbody></table>
    </div>
  </div>
  <div class="card">
    <div class="metrics" id="metrics"></div>
    <div class="rowlab">中价（元）</div><canvas id="c1" height="190"></canvas>
    <div class="rowlab">相对价差（bp）</div><canvas id="c2" height="150"></canvas>
    <div class="rowlab">逐分钟成交量（股）</div><canvas id="c3" height="150"></canvas>
  </div>
</div>

<div class="gal">
__FIGTAGS__
</div>

<script>
const DATA=__PAYLOAD__;
const KEYS=["RelSpread_bp","QuotedSpread_ticks","EffSpread_bp","RV_bp","DepthAmt_wan","CancelRate","AggrRate","FleetingRate","VPIN","Top1VolShare","CloseAucAmtShare","DayRet"];
const KLAB={"RelSpread_bp":"相对价差bp","QuotedSpread_ticks":"价差跳数","EffSpread_bp":"有效价差bp","RV_bp":"波动bp","DepthAmt_wan":"十档深度万","CancelRate":"撤单率","AggrRate":"主动单率","FleetingRate":"闪电单率","VPIN":"VPIN","Top1VolShare":"前1%成交占比","CloseAucAmtShare":"收盘竞价占比","DayRet":"当日涨跌%"};
let sortKey="RelSpread_bp", cur=null;
const codes=Object.keys(DATA).sort((a,b)=>DATA[a].metric[sortKey]-DATA[b].metric[sortKey]);

function fmt(k,v){if(v===null)return"-";if(k.endsWith("Rate")||k==="Top1VolShare"||k==="CloseAucAmtShare")return (v*100).toFixed(1)+"%";return v.toFixed(2);}

function buildTable(){
  const thead=document.querySelector("#tbl thead"), tbody=document.querySelector("#tbl tbody");
  thead.innerHTML="<tr><th>股票</th>"+KEYS.map(k=>`<th data-k="${k}">${KLAB[k]}</th>`).join("")+"</tr>";
  thead.querySelectorAll("th[data-k]").forEach(th=>th.onclick=()=>{sortKey=th.dataset.k;render();});
  tbody.innerHTML="";
  codes.forEach(c=>{
    const d=DATA[c], tr=document.createElement("tr"); tr.dataset.c=c;
    tr.innerHTML=`<td>${c} ${d.name}</td>`+KEYS.map(k=>`<td>${fmt(k,d.metric[k])}</td>`).join("");
    tr.onclick=()=>select(c);
    tbody.appendChild(tr);
  });
}

function select(c){
  cur=c; document.getElementById("pick").value=c;
  document.querySelectorAll("#tbl tbody tr").forEach(tr=>tr.classList.toggle("sel",tr.dataset.c===c));
  const d=DATA[c];
  document.getElementById("metrics").innerHTML=KEYS.slice(0,8).map(k=>`<div class="m"><div class="k">${KLAB[k]}</div><div class="v">${fmt(k,d.metric[k])}</div></div>`).join("");
  line("c1",d.si,d.mid,"#58a6ff"); line("c2",d.si,d.spread,"#f0883e"); line("c3",d.vsi,d.vol,"#3fb950",true);
}

function line(id,xs,ys,col,bars){
  const cv=document.getElementById(id), dpr=window.devicePixelRatio||1;
  const w=cv.clientWidth,h=cv.height; cv.width=w*dpr; cv.height=h*dpr;
  const g=cv.getContext("2d"); g.scale(dpr,dpr); g.clearRect(0,0,w,h);
  const pad={l:52,r:8,t:8,b:16};
  const xmin=0,xmax=240,ymin=Math.min(...ys),ymax=Math.max(...ys);
  const X=v=>pad.l+(v-xmin)/(xmax-xmin)*(w-pad.l-pad.r);
  const Y=v=>h-pad.b-(v-ymin)/((ymax-ymin)||1)*(h-pad.t-pad.b);
  g.strokeStyle="#30363d";g.fillStyle="#8b949e";g.font="10px sans-serif";
  g.beginPath();g.moveTo(pad.l,pad.t);g.lineTo(pad.l,h-pad.b);g.stroke();
  g.fillText(ymax.toFixed(2),1,pad.t+8); g.fillText(ymin.toFixed(2),1,h-pad.b);
  g.fillStyle="#8b949e";
  [0,60,120,180,240].forEach(t=>{const x=X(t);g.fillText([\"09:30\",\"10:30\",\"11:30\",\"14:00\",\"15:00\"][t/60],x-14,h-4);});
  if(bars){g.fillStyle=col; xs.forEach((x,i)=>{const xx=X(x),y0=Y(0),y1=Y(ys[i]);g.fillRect(xx-1,Math.min(y0,y1),2,Math.abs(y1-y0)||1);});}
  else{g.strokeStyle=col;g.lineWidth=1.4;g.beginPath();xs.forEach((x,i)=>{const px=X(x),py=Y(ys[i]);i?g.lineTo(px,py):g.moveTo(px,py);});g.stroke();}
}

function render(){
  Object.keys(DATA).sort((a,b)=>DATA[a].metric[sortKey]-DATA[b].metric[sortKey]).forEach((c,i)=>codes[i]=c);
  buildTable(); select(cur||codes[0]);
}
window.addEventListener("resize",()=>{if(cur)select(cur);});
const pick=document.getElementById("pick");
Object.keys(DATA).forEach(c=>{const o=document.createElement("option");o.value=c;o.textContent=c+" "+DATA[c].name;pick.appendChild(o);});
pick.onchange=()=>select(pick.value);
render();
</script>
</body></html>"""

html = html.replace("__PAYLOAD__", payload).replace("__FIGTAGS__", fig_tags)
outp = os.path.join(ROOT, "out", "micro_dashboard.html")
with open(outp, "w", encoding="utf-8") as f:
    f.write(html)
print("wrote", outp, os.path.getsize(outp) // 1024, "KB")
