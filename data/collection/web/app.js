"use strict";
const $ = (id) => document.getElementById(id);
const canvas = $("canvas"), ctx = canvas.getContext("2d");
const layer = document.createElement("canvas"), layerCtx = layer.getContext("2d");
const fragment = new URLSearchParams(location.hash.slice(1));
const token = fragment.get("token") || sessionStorage.getItem("annotationToken") || "";
if (token) sessionStorage.setItem("annotationToken", token);
if (location.hash) history.replaceState(null, "", location.pathname);
let queue = [], current = null, labels = null, photo = null, selected = 255;
const brushSizes = new Map([0, 128, 255].map(value => [value, Number($("size").value)]));
let dirty = false, busy = false, undo = [], drawing = false, previous = null, cursor = null;
let lastExcluded = null;

function message(text, error=false) { $("message").textContent=text; $("message").className=error ? "error" : ""; }
async function api(path, payload) {
  const options = {headers: {"X-Annotation-Token": token}, cache: "no-store"};
  if (payload !== undefined) {
    options.method="POST"; options.headers["Content-Type"]="application/json";
    options.body=JSON.stringify(payload);
  }
  const response = await fetch(path, options);
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || `请求失败 ${response.status}`);
  return data;
}
function toBase64(array) {
  let text="";
  for (let i=0; i<array.length; i+=8192) text+=String.fromCharCode(...array.subarray(i,i+8192));
  return btoa(text);
}
function setDirty(value) { dirty=value; $("dirty").textContent=value ? "● 有未保存修改" : ""; }
function setBusy(value) {
  busy=value;
  for (const id of ["save","skip","prev","next","filter","frames","clear","undo","exclude","restore","excludeReason","undoExclude"]) $(id).disabled=value;
  for (const id of ["save","skip","clear","undo","exclude","excludeReason"]) $(id).disabled=value || !current || current.excluded;
  $("restore").hidden=!current?.excluded; $("restore").disabled=value || !current?.excluded;
  $("exclude").hidden=!!current?.excluded;
  $("undoExclude").hidden=!lastExcluded;
}
function visibleRows() {
  const filter=$("filter").value;
  return queue.filter(r=>filter==="all" || (filter==="excluded" ? r.excluded : !r.excluded && (
    filter==="valid" || (filter==="pending" ? r.status==="ND" :
    filter==="reviewed" ? ["true","test"].includes(r.status.toLowerCase()) : r.status==="pass"))));
}
function renderQueue() {
  $("frames").replaceChildren();
  for (const r of visibleRows()) {
    const option=document.createElement("option"); option.value=r.id;
    option.textContent=`${r.id+1}. [${r.excluded ? "已排除" : r.split || r.status}] ${r.filename}`;
    option.title=r.filename; $("frames").append(option);
  }
  if (current) $("frames").value=String(current.id);
  const active=queue.filter(r=>!r.excluded), done=active.filter(r=>["true","test"].includes(r.status.toLowerCase())).length;
  const pending=active.filter(r=>r.status==="ND").length;
  $("progress").textContent=`总计 ${queue.length} · 有效 ${active.length} · 已确认 ${done} · 待标 ${pending} · 跳过 ${active.filter(r=>r.status==="pass").length} · 已排除 ${queue.length-active.length}`;
}
async function refreshQueue() { const data=await api("/api/queue"); queue=data.frames; $("dataset").textContent=data.dataset; renderQueue(); }
function mayLeave() { return !dirty || confirm("当前修改尚未保存，确定放弃这些修改？"); }
function setBrushSize(value) {
  const input=$("size");
  input.value=Math.max(Number(input.min),Math.min(Number(input.max),value));
  brushSizes.set(selected,Number(input.value)); $("sizeValue").value=input.value; render();
}
function chooseBrush(value) {
  selected=value;
  $("size").value=brushSizes.get(value);
  $("sizeValue").value=$("size").value;
  for (const button of $("brushes").children) button.classList.toggle("active", Number(button.dataset.value)===value);
  $("erase").classList.toggle("active", value===0); render();
}
function resizeView() {
  if (!current) return;
  const zoom=$("zoom").value;
  const scale=zoom==="fit" ? Math.min(($("viewport").clientWidth-4)/current.width, ($("viewport").clientHeight-4)/current.height) : Number(zoom);
  canvas.style.width=`${Math.max(.1,scale)*current.width}px`;
  canvas.style.height=`${Math.max(.1,scale)*current.height}px`;
}
function render() {
  if (!current || !photo || !labels) return;
  ctx.drawImage(photo,0,0);
  if ($("overlay").checked) {
    const rgba=layerCtx.createImageData(current.width,current.height);
    const colors=new Map(current.brushes.map(b=>[b.value,b.color.match(/[a-f0-9]{2}/gi).map(c=>parseInt(c,16))]));
    const alpha=Math.round(Number($("alpha").value)*2.55);
    for(let i=0;i<labels.length;i++) {
      const color=colors.get(labels[i]); if (!color) continue;
      const j=i*4; rgba.data[j]=color[0]; rgba.data[j+1]=color[1]; rgba.data[j+2]=color[2]; rgba.data[j+3]=alpha;
    }
    layerCtx.putImageData(rgba,0,0); ctx.drawImage(layer,0,0);
  }
  if (cursor) {
    ctx.beginPath(); ctx.arc(cursor.x,cursor.y,Number($("size").value)/2,0,2*Math.PI);
    ctx.strokeStyle=selected===0 ? "#fff" : current.brushes.find(b=>b.value===selected).color;
    ctx.lineWidth=1.5; ctx.stroke();
  }
}
function remember() { undo.push(labels.slice()); if(undo.length>20) undo.shift(); }
function undoStroke() { if(busy || current?.excluded || !undo.length) return; labels=undo.pop(); setDirty(true); render(); }
function position(event) {
  const r=canvas.getBoundingClientRect();
  return {x:(event.clientX-r.left)*canvas.width/r.width, y:(event.clientY-r.top)*canvas.height/r.height};
}
function stamp(p) {
  const radius=Number($("size").value)/2, w=current.width, h=current.height;
  for(let y=Math.max(0,Math.floor(p.y-radius));y<Math.min(h,Math.ceil(p.y+radius));y++)
    for(let x=Math.max(0,Math.floor(p.x-radius));x<Math.min(w,Math.ceil(p.x+radius));x++)
      if((x-p.x)**2+(y-p.y)**2<=radius**2) labels[y*w+x]=selected;
}
function stroke(a,b) {
  const steps=Math.max(1,Math.ceil(Math.hypot(b.x-a.x,b.y-a.y)/Math.max(1,Number($("size").value)/4)));
  for(let i=1;i<=steps;i++) stamp({x:a.x+(b.x-a.x)*i/steps,y:a.y+(b.y-a.y)*i/steps});
}
async function loadFrame(id, discard=false) {
  if(busy || (!discard && !mayLeave())) { if(current) $("frames").value=String(current.id); return; }
  setBusy(true);
  try {
    const data=await api(`/api/frame?id=${id}`);
    // Keep the A/V display convention stable, including when an older server
    // remains running while these static assets are updated. Pixel ids are unchanged.
    data.brushes=data.brushes.map(b=>({...b,color:b.name==="静脉" ? "#4d9fff" : b.name==="动脉" ? "#ff6068" : b.color}));
    const img=new Image(); img.src=`data:image/png;base64,${data.image}`; await img.decode();
    const decoded=Uint8Array.from(atob(data.labels),c=>c.charCodeAt(0));
    if(decoded.length!==data.width*data.height) throw new Error("Mask 尺寸错误");
    current=data; photo=img; labels=decoded; undo=[]; cursor=null; drawing=false; previous=null;
    canvas.width=layer.width=data.width; canvas.height=layer.height=data.height;
    canvas.hidden=false; $("empty").hidden=true; $("overlay").checked=true;
    $("brushes").replaceChildren();
    data.brushes.forEach((b,i)=> {const btn=document.createElement("button");btn.textContent=`${b.name} ${i+1}`;
      btn.dataset.value=b.value;btn.style.color=b.color;btn.onclick=()=>chooseBrush(b.value);$("brushes").append(btn);});
    $("legend").replaceChildren();
    data.brushes.forEach(b=> {const line=document.createElement("span");line.textContent=b.name;line.style.color=b.color;
      $("legend").append(line,document.createElement("br"));});
    $("legend").append(document.createTextNode("预标注仅为草稿，请逐帧核对。"));
    chooseBrush(data.brushes[0].value); setDirty(false); resizeView(); render(); renderQueue();
    const source={suggestion:`${data.suggestion_model || "模型"} 预标注 · 必须人工核对`,blank:"空白 · 请人工标注",reviewed:"已保存标注"}[data.source];
    $("frameInfo").textContent=`${data.filename} | ${data.width}×${data.height} | ${data.split} | 受试者 ${data.subject || "—"} | ${source}`;
    const reasons={manual_no_contact:"探头未接触 / 无有效组织回声",manual_non_ultrasound:"非超声画面",manual_unusable_image:"图像损坏或采集失败",no_contact_near_empty_tissue_roi:"此前筛选的无接触帧，可人工复核恢复"};
    $("qualityInfo").textContent=data.excluded ? `已排除：${reasons[data.exclusion_reason] || data.exclusion_reason || "无效帧"}。原图与旧标注仍保留；恢复后需核对并保存。` : "";
    $("save").textContent=data.brushes.length===2 ? "确认动静脉并保存 →" : "确认标注并保存 →";
    $("cleanupInfo").hidden=!data.cleanup_on_save;
    message(data.source==="suggestion" ? "请核对每个血管的类型和边界。点击保存才会成为训练标注。" : "");
  } catch(e) { message(e.message,true); if(current) $("frames").value=String(current.id); }
  finally { setBusy(false); }
}
function emptyQueue() {
  current=null; labels=null; photo=null; undo=[]; canvas.hidden=true; $("empty").hidden=false;
  $("empty").textContent="此队列没有图像。可切换到「全部」或「已确认」复查。";
  $("frameInfo").textContent=""; $("qualityInfo").textContent=""; setDirty(false); setBusy(false);
}
async function save(skip=false) {
  if(busy || drawing || !current || current.excluded) return;
  if(skip && !confirm("将此帧标记为跳过，不用于训练。未保存修改会被放弃，是否继续？")) return;
  if(!skip && !labels.some(v=>v!==0) && !confirm("当前 mask 完全为空，确认这帧没有需要标注的血管？")) return;
  setBusy(true); const id=current.id;
  try {
    const payload={id,revision:current.revision}; if(!skip) payload.labels=toBase64(labels);
    const result=await api(skip ? "/api/skip" : "/api/save",payload); setDirty(false);
    // The write succeeded. Do not retry a save with the old revision if refresh fails.
    current=null;
    await refreshQueue();
    const rows=visibleRows(), next=rows.find(r=>r.id>id) || rows.find(r=>r.id!==id) || rows[0];
    setBusy(false);
    if(next) await loadFrame(next.id,true); else emptyQueue();
    message(skip ? "已跳过。" : result.cleanup ? `第 ${id+1} 张已保存：已补全轮廓并清除孤立区域，调整 ${result.cleanup.changed_pixels} 个像素。可返回上一张查看结果。` : "已保存到服务器。");
  } catch(e) { message(e.message,true); }
  finally { setBusy(false); }
}
async function changeQuality(action, target=current) {
  if(busy || drawing || !target || !mayLeave()) return;
  setBusy(true); const id=target.id;
  try {
    const payload={id,revision:target.revision};
    if(action==="exclude") payload.reason=$("excludeReason").value;
    const response=await api(`/api/${action}`,payload);
    if(action==="exclude") lastExcluded={id,revision:response.revision};
    else if(lastExcluded?.id===id) lastExcluded=null;
    setDirty(false); current=null;
    if(action==="restore") $("filter").value=lastFilter="pending";
    await refreshQueue();
    const rows=visibleRows(), next=action==="restore" ? rows.find(r=>r.id===id) : rows.find(r=>r.id>id) || rows.find(r=>r.id!==id);
    setBusy(false);
    if(next) await loadFrame(next.id,true); else emptyQueue();
    message(action==="exclude" ? `第 ${id+1} 张已排除，原图和标注保留，可撤销。` : `第 ${id+1} 张已恢复到待确认，请核对旧标注后保存。`);
  } catch(e) { message(e.message,true); }
  finally { setBusy(false); }
}
canvas.addEventListener("pointerdown",e=> {
  if(e.button!==0 || busy || !current || current.excluded) return;
  e.preventDefault(); canvas.setPointerCapture(e.pointerId); remember(); drawing=true;
  previous=cursor=position(e); stamp(previous); setDirty(true); render();
});
canvas.addEventListener("pointermove",e=> {
  if(busy || !current) return; cursor=position(e);
  if(drawing) {stroke(previous,cursor); previous=cursor;} render();
});
for(const name of ["pointerup","pointercancel","lostpointercapture"]) canvas.addEventListener(name,()=>{drawing=false;previous=null;});
canvas.addEventListener("pointerleave",()=>{if(!drawing){cursor=null;render();}});
canvas.addEventListener("wheel",e=> {
  if(busy || !current || current.excluded || e.shiftKey || e.ctrlKey || e.metaKey || !e.deltaY) return;
  e.preventDefault();
  cursor=position(e);
  setBrushSize(Number($("size").value)+(e.deltaY<0 ? 2 : -2));
}, {passive:false});
$("frames").onchange=()=>loadFrame(Number($("frames").value));
let lastFilter="valid";
$("filter").onchange=async()=> {
  if(!mayLeave()) {$("filter").value=lastFilter;return;}
  lastFilter=$("filter").value; renderQueue(); const rows=visibleRows();
  if(rows.length) await loadFrame(rows[0].id,true); else emptyQueue();
};
function move(delta) {const rows=visibleRows(),i=rows.findIndex(r=>r.id===current?.id);if(rows[i+delta])loadFrame(rows[i+delta].id);}
$("prev").onclick=()=>move(-1); $("next").onclick=()=>move(1);
$("save").onclick=()=>save(); $("skip").onclick=()=>save(true);
$("exclude").onclick=()=>changeQuality("exclude"); $("restore").onclick=()=>changeQuality("restore");
$("undoExclude").onclick=()=>changeQuality("restore",lastExcluded);
$("erase").onclick=()=>chooseBrush(0); $("undo").onclick=undoStroke;
$("clear").onclick=()=>{if(!current || busy || current.excluded)return;remember();labels.fill(0);setDirty(true);render();};
$("size").oninput=()=>setBrushSize(Number($("size").value));
$("alpha").oninput=render; $("overlay").onchange=render; $("zoom").onchange=resizeView;
window.addEventListener("resize",resizeView);
window.addEventListener("beforeunload",e=>{if(dirty){e.preventDefault();e.returnValue="";}});
document.addEventListener("keydown",e=> {
  if(busy || !current || ["INPUT","SELECT","TEXTAREA"].includes(e.target.tagName)) return;
  if((e.ctrlKey||e.metaKey)&&e.key.toLowerCase()==="s") {e.preventDefault();save();}
  else if((e.ctrlKey||e.metaKey)&&e.key.toLowerCase()==="z") {e.preventDefault();undoStroke();}
  else if(e.key==="Tab") {e.preventDefault();$("overlay").checked=!$("overlay").checked;render();}
  else if(e.key.toLowerCase()==="e") chooseBrush(0);
  else if(["1","2"].includes(e.key) && current.brushes[Number(e.key)-1]) chooseBrush(current.brushes[Number(e.key)-1].value);
});
(async()=> {setBusy(true);try {await refreshQueue();
if (!visibleRows().length && queue.some(r=>["true","test"].includes(r.status.toLowerCase()))) {
  $("filter").value=lastFilter="reviewed"; renderQueue();
}
setBusy(false);const rows=visibleRows();if(rows.length)await loadFrame(rows[0].id,true);else emptyQueue();}
catch(e){setBusy(false);$("empty").textContent="连接失败，请检查 SSH 隧道及服务器输出的完整链接。";message(e.message,true);}})();
