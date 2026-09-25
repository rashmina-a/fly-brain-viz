/* Fly Brain Explorer — real MaleCNS v1.0 connectome (HHMI Janelia, CC-BY-4.0)
 * Soma point-cloud + on-demand skeletons + firing simulation on real synapses. */
"use strict";

/* ---------------- state ---------------- */
let scene, camera, renderer, controls, raycaster;
let somaMesh = null;
let neurons = [];            // per-neuron records (full dataset)
let neuronIndex = new Map(); // bodyId -> record
let pointOfNeuron = null;    // Int32Array neuronIdx -> point index (-1 hidden)
let neuronOfPoint = [];      // point index -> neuronIdx
let edge = null;             // {pre,post,w} lazily loaded
let redge = null;            // {pre,w} + idx (reverse CSR, lazily loaded)
let edgeIdxIds = null, edgeIdxOff = null;
let redgeIdxIds = null, redgeIdxOff = null;
let morphGroup = null;
let connGroup = null;
let selMarker = null;
let selectedId = null;
let canvasPointer = null;
let fireArr = null;          // Float32Array per neuron, decaying excitation
let seedHold = new Set();    // neurons currently receiving the stimulus
let holdUntil = 0;           // ms timestamp: how long the stimulus stays on
let simActiveUntil = 0;      // status text lifetime

const VOXEL_NM = 8;          // somaLocation + skeletons are in 8nm voxels / nm
const SCALE = 1e-5;          // nm -> scene units (CNS spans ~10 scene units)
/* scene mapping: x -> x (L-R), voxel z -> -y (VNC hangs below brain), voxel y -> z */
function toScene(x, y, z) {
  return [x * VOXEL_NM * SCALE, -z * VOXEL_NM * SCALE, y * VOXEL_NM * SCALE];
}
const CENTER = toScene(48686.5, 27515.5, 24721.5);
const CAM_HOME = [CENTER[0] + 6.5, CENTER[1] + 3.2, CENTER[2] + 9.0];

const NT_COLORS = {
  acetylcholine: 0x4fc3f7, gaba: 0xff7043, glutamate: 0xffd54f,
  dopamine: 0xba68c8, octopamine: 0x81c784, serotonin: 0xf06292,
  tyramine: 0x4db6ac, histamine: 0xaed581, unknown: 0x888888,
};

const STIMULI = [
  { id: "olfactory",  label: "🍎 Odor (olfactory)",  cls: ["olfactory", "ALPN"], desc: "Activates olfactory sensory neurons + antennal lobe PNs" },
  { id: "gustatory",  label: "🍬 Taste (gustatory)", cls: ["gustatory"],         desc: "Activates gustatory neurons" },
  { id: "touch",      label: "👆 Touch",             cls: ["mechanosensory_tactile"], desc: "Activates touch mechanosensory neurons" },
  { id: "proprio",    label: "🦵 Proprioception",    cls: ["mechanosensory_proprioceptive"], desc: "Activates proprioceptive neurons" },
  { id: "mechano",    label: "🌀 Mechanical (all)",  cls: ["mechanosensory", "mechanosensory_tactile", "mechanosensory_proprioceptive"], desc: "Activates all mechanosensory neurons" },
  { id: "visual",     label: "👁️ Flash of light",    cls: ["visual"],            desc: "Activates visual + optic-lobe neurons" },
  { id: "humidity",   label: "💧 Humidity",          cls: ["hygrosensory"],      desc: "Activates hygrosensory neurons" },
  { id: "DAN",        label: "🧠 Dopamine burst",    cls: ["DAN"],               desc: "Activates dopaminergic neurons (memory/reinforcement)" },
  { id: "MBON",       label: "🍄 Memory output",     cls: ["MBON"],              desc: "Activates mushroom body output neurons" },
  { id: "KC",         label: "🍄 Kenyon cells",      cls: ["Kenyon_Cell"],       desc: "Activates mushroom body Kenyon cells directly" },
  { id: "CX",         label: "🧭 Heading change",    cls: ["CX"],                desc: "Activates central-complex neurons" },
  { id: "unknown",    label: "❓ Unknown sensory",   cls: ["unknown_sensory"],   desc: "Activates unclassified sensory neurons" },
];

const REGIONS = {
  "":   { label: "Whole CNS", test: () => true },
  "ol": { label: "Optic lobes", test: (n) => /^ol_/.test(n.superclass) },
  "cb": { label: "Central brain", test: (n) => /^cb_/.test(n.superclass) ||
      ["descending_neuron", "ascending_neuron", "visual_projection",
       "visual_centrifugal"].includes(n.superclass) },
  "vnc": { label: "VNC (nerve cord)", test: (n) => /^vnc_/.test(n.superclass) },
};

/* ---------------- helpers ---------------- */
const $ = (id) => document.getElementById(id);
function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
async function fetchBuffer(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(url + " -> " + r.status);
  return r.arrayBuffer();
}
function fmt(n) { return n.toLocaleString("en-US"); }

/* ---------------- init ---------------- */
init();
loadData();

function init() {
  const canvas = $("canvas3d");
  renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0b0e14);
  scene.fog = new THREE.FogExp2(0x0b0e14, 0.010);

  camera = new THREE.PerspectiveCamera(50, 1, 0.01, 200);
  camera.position.set(...CAM_HOME);

  controls = new THREE.OrbitControls(camera, canvas);
  controls.target.set(...CENTER);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.mouseButtons = {
    LEFT: THREE.MOUSE.ROTATE, MIDDLE: THREE.MOUSE.DOLLY, RIGHT: THREE.MOUSE.PAN,
  };

  raycaster = new THREE.Raycaster();
  raycaster.params.Points.threshold = 0.05;

  window.addEventListener("resize", onResize);
  onResize();
  canvas.addEventListener("pointerdown", onPointerDown);
  canvas.addEventListener("pointerup", onPointerUp);

  // stimulus dropdown
  const sel = $("stimSelect");
  for (const s of STIMULI) {
    const o = document.createElement("option");
    o.value = s.id; o.textContent = s.label;
    sel.appendChild(o);
  }
  sel.onchange = () => {
    $("stimDesc").textContent = STIMULI.find((x) => x.id === sel.value).desc;
  };
  sel.onchange();

  $("fireBtn").onclick = fireStimulus;
  $("searchBtn").onclick = doSearch;
  $("search").addEventListener("keydown", (e) => { if (e.key === "Enter") doSearch(); });
  $("countRange").oninput = () => {
    $("countVal").textContent = ($("countRange").value / 1000) + "k";
    buildPointCloud();
  };
  for (const id of ["sideL", "sideR", "sideM"]) $(id).onchange = buildPointCloud;
  const rs = $("regionSel");
  for (const k of Object.keys(REGIONS)) {
    const o = document.createElement("option");
    o.value = k; o.textContent = REGIONS[k].label;
    rs.appendChild(o);
  }
  rs.onchange = buildPointCloud;
  $("resetBtn").onclick = () => {
    camera.position.set(...CAM_HOME);
    controls.target.set(...CENTER);
  };
  $("morphChk").onchange = () => {
    if (!$("morphChk").checked && morphGroup) { scene.remove(morphGroup); morphGroup = null; }
    else if (selectedId) loadSkeleton(selectedId);
  };

  const legend = ["acetylcholine", "gaba", "glutamate", "dopamine", "octopamine", "serotonin"];
  $("legend").innerHTML = legend.map((n) => {
    const c = "#" + NT_COLORS[n].toString(16).padStart(6, "0");
    return `<span class="lg" style="background:${c}"></span>${n}&nbsp; `;
  }).join("") + `<span class="lg" style="background:#888"></span>other`;

  animate();
}

function onResize() {
  const v = $("viewport");
  camera.aspect = v.clientWidth / v.clientHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(v.clientWidth, v.clientHeight);
}

/* ---------------- data loading ---------------- */
async function loadData() {
  const bar = $("loadBar"), msg = $("loadMsg");
  const set = (p, m) => { bar.style.width = p + "%"; if (m) msg.textContent = m; };
  try {
    set(5, "fetching neuron table…");
    const r = await fetch("data/neurons.tsv.gz");
    if (!r.ok) throw new Error("neurons.tsv.gz " + r.status);
    const rows = await tsvLines(new Uint8Array(await r.arrayBuffer()));
    set(40, "parsing " + fmt(rows.length - 1) + " neurons…");
    await new Promise((res) => setTimeout(res, 30));
    parseNeurons(rows);

    set(55, "loading synaptic index…");
    [edgeIdxIds, edgeIdxOff] = await Promise.all([
      fetchBuffer("data/edges_idx_ids.u32").then((b) => new Uint32Array(b)),
      fetchBuffer("data/edges_idx_off.u32").then((b) => new Uint32Array(b)),
    ]);
    [redgeIdxIds, redgeIdxOff] = await Promise.all([
      fetchBuffer("data/redges_idx_ids.u32").then((b) => new Uint32Array(b)),
      fetchBuffer("data/redges_idx_off.u32").then((b) => new Uint32Array(b)),
    ]);

    set(70, "building 3D point cloud…");
    await new Promise((res) => setTimeout(res, 30));
    fireArr = new Float32Array(neurons.length);
    buildPointCloud();

    let syn = "26.0M";
    try {
      const meta = await (await fetch("data/meta.json")).json();
      syn = (meta.counts.edges / 1e6).toFixed(1) + "M";
    } catch (_) { /* cosmetic only */ }
    $("statLine").textContent = fmt(neurons.length) + " neurons · " + syn + " synapses";
    set(100, "done");
    setTimeout(() => {
      const l = $("loading");
      l.style.opacity = "0";
      setTimeout(() => (l.style.display = "none"), 450);
    }, 400);
  } catch (e) {
    msg.textContent = "Error: " + e.message;
    console.error(e);
  }
}

async function tsvLines(bytes) {
  if (!("DecompressionStream" in window)) {
    throw new Error("This app needs a browser with DecompressionStream (any recent Chrome/Edge/Firefox).");
  }
  const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream("gzip"));
  const text = await new Response(stream).text();
  return text.split("\n").map((s) => s.replace(/\r$/, ""));
}

function parseNeurons(rows) {
  const header = rows[0].split("\t");
  const col = {};
  header.forEach((h, i) => (col[h] = i));
  neurons = [];
  neuronIndex = new Map();
  for (let r = 1; r < rows.length; r++) {
    const f = rows[r].split("\t");
    if (f.length < header.length || !f[col.bodyId]) continue;
    const x = +f[col.x], y = +f[col.y], z = +f[col.z];
    if (!isFinite(x)) continue;
    const rec = {
      bodyId: +f[col.bodyId],
      instance: f[col.instance] || "",
      type: f[col.type] || "",
      cls: f[col.class] || "",
      superclass: f[col.superclass] || "",
      side: f[col.somaSide] || "",
      nt: f[col.nt] || "unknown",
      x, y, z,                                        // 8nm voxel coords
      scene: toScene(x, y, z),
    };
    const idx = neurons.length;
    neurons.push(rec);
    neuronIndex.set(rec.bodyId, idx);
  }
  pointOfNeuron = new Int32Array(neurons.length).fill(-1);
}

/* ---------------- point cloud ---------------- */
function buildPointCloud() {
  if (!neurons.length) return;
  const maxCount = +$("countRange").value;
  const region = $("regionSel").value;
  const sideL = $("sideL").checked, sideR = $("sideR").checked, sideM = $("sideM").checked;
  const test = REGIONS[region].test;

  const keep = [];
  for (let i = 0; i < neurons.length; i++) {
    const n = neurons[i];
    if (!sideL && n.side === "L") continue;
    if (!sideR && n.side === "R") continue;
    if (!sideM && n.side === "M") continue;
    if (region && !test(n)) continue;
    keep.push(i);
  }
  if (keep.length > maxCount) {
    const step = keep.length / maxCount;
    for (let j = 0; j < maxCount; j++) keep[j] = keep[Math.floor(j * step)];
    keep.length = maxCount;
  }

  const n = keep.length;
  const pos = new Float32Array(n * 3);
  const col = new Float32Array(n * 3);
  const siz = new Float32Array(n);
  const pId = new Uint32Array(n);
  pointOfNeuron.fill(-1);
  neuronOfPoint = new Array(n);

  for (let j = 0; j < n; j++) {
    const i = keep[j], nr = neurons[i];
    pos[j * 3] = nr.scene[0]; pos[j * 3 + 1] = nr.scene[1]; pos[j * 3 + 2] = nr.scene[2];
    const c = new THREE.Color(NT_COLORS[nr.nt] !== undefined ? NT_COLORS[nr.nt] : 0x888888);
    col[j * 3] = c.r; col[j * 3 + 1] = c.g; col[j * 3 + 2] = c.b;
    siz[j] = Math.max(0.14, Math.min(0.6, 0.34 * Math.sqrt(60000 / n)));
    pId[j] = nr.bodyId;
    pointOfNeuron[i] = j;
    neuronOfPoint[j] = i;
  }

  if (somaMesh) { scene.remove(somaMesh); somaMesh.geometry.dispose(); somaMesh.material.dispose(); }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  geo.setAttribute("aColor", new THREE.BufferAttribute(col, 3));
  geo.setAttribute("psize", new THREE.BufferAttribute(siz, 1));
  geo.setAttribute("aFire", new THREE.BufferAttribute(new Float32Array(n), 1));
  geo.setAttribute("aSel", new THREE.BufferAttribute(new Float32Array(n), 1));
  somaMesh = new THREE.Points(geo, somaMaterial());
  somaMesh.userData = { count: n };
  scene.add(somaMesh);
  updateHud();
  if (selectedId !== null) highlightSelected();
}

function somaMaterial() {
  return new THREE.ShaderMaterial({
    uniforms: { uTime: { value: 0 } },
    vertexShader: `
      attribute vec3 aColor;
      attribute float psize;
      attribute float aFire;
      attribute float aSel;
      varying vec3 vColor;
      varying float vFire;
      varying float vSel;
      varying float vPhase;
      void main() {
        vColor = aColor; vFire = aFire; vSel = aSel;
        vPhase = position.x * 13.7 + position.y * 7.3 + position.z * 5.1;
        vec4 mv = modelViewMatrix * vec4(position, 1.0);
        gl_PointSize = psize * (1.0 + 2.2 * max(aFire, aSel)) * (140.0 / -mv.z);
        gl_Position = projectionMatrix * mv;
      }`,
    fragmentShader: `
      uniform float uTime;
      varying vec3 vColor;
      varying float vFire;
      varying float vSel;
      varying float vPhase;
      void main() {
        vec2 c = gl_PointCoord - 0.5;
        float d = length(c);
        if (d > 0.5) discard;
        float f = clamp(vFire, 0.0, 1.0);
        float pulse = 0.75 + 0.25 * sin(uTime * 7.0 + vPhase);
        vec3 fireCol = vec3(1.0, 0.55, 0.16) * pulse;
        vec3 col = mix(vColor, fireCol, f);
        col = mix(col, vec3(1.0), clamp(vSel, 0.0, 1.0) * 0.85);
        float a = smoothstep(0.5, 0.25, d);
        a = min(1.0, a + f * 0.55);
        gl_FragColor = vec4(col, a);
      }`,
    transparent: true,
    depthWrite: false,
  });
}

function updateHud() {
  $("hud").innerHTML = "<b>Fly Brain Explorer</b> — MaleCNS v1.0<br>" +
    fmt(somaMesh ? somaMesh.userData.count : 0) + " somas · color = neurotransmitter";
}

/* ---------------- picking & selection ---------------- */
function onPointerDown(e) { canvasPointer = { x: e.clientX, y: e.clientY, t: Date.now(), b: e.button }; }
function onPointerUp(e) {
  if (!canvasPointer || canvasPointer.b !== 0) { canvasPointer = null; return; }
  const d = Math.hypot(e.clientX - canvasPointer.x, e.clientY - canvasPointer.y);
  const dt = Date.now() - canvasPointer.t;
  canvasPointer = null;
  if (d < 4 && dt < 350) pickAt(e);
}
function pickAt(e) {
  if (!somaMesh) return;
  const rect = $("viewport").getBoundingClientRect();
  raycaster.setFromCamera(new THREE.Vector2(
    ((e.clientX - rect.left) / rect.width) * 2 - 1,
    -((e.clientY - rect.top) / rect.height) * 2 + 1), camera);
  const hits = raycaster.intersectObject(somaMesh);
  if (hits.length) selectNeuron(neuronOfPoint[hits[0].index]);
  else deselect();
}

function selectNeuron(neuronIdx) {
  const n = neurons[neuronIdx];
  if (!n) return;
  selectedId = n.bodyId;
  showInfo(n, neuronIdx);
  highlightSelected();
  if ($("morphChk").checked) loadSkeleton(n.bodyId);
}
function deselect() {
  selectedId = null;
  document.body.classList.remove("has-sel");
  $("info").innerHTML = "Select a neuron (dot) to see details, shape and connections.";
  clearASel();
  if (morphGroup) { scene.remove(morphGroup); morphGroup = null; }
  if (connGroup) { scene.remove(connGroup); connGroup = null; }
  if (selMarker) { scene.remove(selMarker); selMarker = null; }
}
function clearASel() {
  if (!somaMesh) return;
  const a = somaMesh.geometry.getAttribute("aSel");
  a.array.fill(0); a.needsUpdate = true;
}

function highlightSelected() {
  if (!somaMesh) return;
  const aSel = somaMesh.geometry.getAttribute("aSel");
  aSel.array.fill(0);
  const pi = selectedId === null ? -1 : pointOfNeuron[neuronIndex.get(selectedId)];
  if (pi >= 0) { aSel.array[pi] = 1; aSel.needsUpdate = true; }
  drawSelMarker(pi);
}

function drawSelMarker(pi) {
  if (selMarker) { scene.remove(selMarker); selMarker = null; }
  if (pi < 0) return;
  const pos = somaMesh.geometry.getAttribute("position");
  const geo = new THREE.SphereGeometry(0.04, 10, 10);
  const mat = new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.9 });
  selMarker = new THREE.Mesh(geo, mat);
  selMarker.position.set(pos.getX(pi), pos.getY(pi), pos.getZ(pi));
  scene.add(selMarker);
}

/* ---------------- info panel + connectivity ---------------- */
function showInfo(n, neuronIdx) {
  document.body.classList.add("has-sel");
  const c = "#" + (NT_COLORS[n.nt] !== undefined ? NT_COLORS[n.nt] : 0x888888).toString(16).padStart(6, "0");
  $("info").innerHTML = `
    <h3>${esc(n.instance || n.type || "body " + n.bodyId)}</h3>
    <div class="typ">${esc(n.cls || "unclassified")} · ${esc(n.superclass)} · ${esc(n.side || "–")} side</div>
    <table>
      <tr><td>bodyId</td><td>${n.bodyId}</td></tr>
      <tr><td>type</td><td>${esc(n.type || "–")}</td></tr>
      <tr><td>neurotransmitter</td><td style="color:${c}">${esc(n.nt)}</td></tr>
      <tr><td>soma (voxel)</td><td>${n.x}, ${n.y}, ${n.z}</td></tr>
    </table>
    <div style="margin-top:10px"><b>Outputs</b> <span class="hint">(postsynaptic, top 15)</span></div>
    <div id="outList" class="hint">loading…</div>
    <div style="margin-top:10px"><b>Inputs</b> <span class="hint">(presynaptic, top 15)</span></div>
    <div id="inList" class="hint">loading…</div>
    <div style="margin-top:10px"><span class="tag" onclick="focusCamera(${n.bodyId})">🎯 focus</span>
    <span class="tag" onclick="loadSkeleton(${n.bodyId})">🧵 shape</span></div>`;
  renderConnections(n.bodyId);
}

function csrRange(ids, off, bodyId) {
  let lo = 0, hi = ids.length;
  while (lo < hi) { const m = (lo + hi) >> 1; if (ids[m] < bodyId) lo = m + 1; else hi = m; }
  if (lo >= ids.length || ids[lo] !== bodyId) return null;
  return [off[lo], off[lo + 1]];
}

async function ensureEdgeArrays() {
  if (edge) return;
  const [preB, postB, wB, rpreB, rwB] = await Promise.all([
    fetchBuffer("data/edges_pre.u32"), fetchBuffer("data/edges_post.u32"),
    fetchBuffer("data/edges_w.u16"), fetchBuffer("data/redges_pre.u32"),
    fetchBuffer("data/redges_w.u16"),
  ]);
  edge = { pre: new Uint32Array(preB), post: new Uint32Array(postB), w: new Uint16Array(wB) };
  redge = { pre: new Uint32Array(rpreB), w: new Uint16Array(rwB) };
}

async function renderConnections(bodyId) {
  await ensureEdgeArrays();
  const outs = [], ins = [];
  let r = csrRange(edgeIdxIds, edgeIdxOff, bodyId);
  if (r) for (let k = r[0]; k < r[1]; k++) outs.push([edge.post[k], edge.w[k]]);
  r = csrRange(redgeIdxIds, redgeIdxOff, bodyId);
  if (r) for (let k = r[0]; k < r[1]; k++) ins.push([redge.pre[k], redge.w[k]]);
  outs.sort((a, b) => b[1] - a[1]);
  ins.sort((a, b) => b[1] - a[1]);

  const row = (id, w) => {
    const n = neurons[neuronIndex.get(id)];
    const nm = n ? (n.instance || n.type || id) : id;
    return `<div class="conn" onclick="selectById(${id})"><span>${esc(nm)}</span>` +
      `<span class="w">${fmt(w)}</span></div>`;
  };
  $("outList").innerHTML = outs.length
    ? outs.slice(0, 15).map((p) => row(p[0], p[1])).join("") +
      (outs.length > 15 ? `<div class="hint">… ${fmt(outs.length - 15)} more partners</div>` : "")
    : "<div class='hint'>no output synapses found</div>";
  $("inList").innerHTML = ins.length
    ? ins.slice(0, 15).map((p) => row(p[0], p[1])).join("") +
      (ins.length > 15 ? `<div class="hint">… ${fmt(ins.length - 15)} more partners</div>` : "")
    : "<div class='hint'>no input synapses found</div>";

  drawConnLines(bodyId, outs.slice(0, 60), ins.slice(0, 60));
}

function drawConnLines(bodyId, outs, ins) {
  if (connGroup) { scene.remove(connGroup); connGroup = null; }
  connGroup = new THREE.Group();
  const from = neurons[neuronIndex.get(bodyId)];
  const mkMat = (col, op) => new THREE.LineBasicMaterial({ color: col, transparent: true, opacity: op });
  const addLines = (partners, col, op) => {
    const pts = [];
    for (const [id] of partners) {
      const idx = neuronIndex.get(id);
      if (idx === undefined) continue;
      const pi = pointOfNeuron[idx];
      if (pi < 0) continue;                       // partner not visible: skip
      const p = somaMesh.geometry.getAttribute("position");
      pts.push(new THREE.Vector3(from.scene[0], from.scene[1], from.scene[2]));
      pts.push(new THREE.Vector3(p.getX(pi), p.getY(pi), p.getZ(pi)));
    }
    if (!pts.length) return;
    const g = new THREE.BufferGeometry().setFromPoints(pts);
    connGroup.add(new THREE.LineSegments(g, mkMat(col, op)));
  };
  addLines(outs, 0x4fc3f7, 0.55);   // outputs: cyan
  addLines(ins, 0xff7043, 0.55);    // inputs: orange
  scene.add(connGroup);
}

window.selectById = function (bodyId) {
  const idx = neuronIndex.get(bodyId);
  if (idx === undefined) return;
  const pi = pointOfNeuron[idx];
  if (pi < 0) {
    // hidden by current filters — show info + marker anyway
    selectedId = bodyId;
    showInfo(neurons[idx], idx);
    clearASel();
    if (connGroup) { scene.remove(connGroup); connGroup = null; }
    if (morphGroup) { scene.remove(morphGroup); morphGroup = null; }
    if (selMarker) { scene.remove(selMarker); selMarker = null; }
    return;
  }
  selectNeuron(idx);
};

window.focusCamera = function (bodyId) {
  const idx = neuronIndex.get(bodyId);
  if (idx === undefined) return;
  const s = neurons[idx].scene;
  controls.target.set(s[0], s[1], s[2]);
  camera.position.set(s[0] + 1.6, s[1] + 1.0, s[2] + 2.2);
};

/* ---------------- skeleton (neuron shape) ---------------- */
async function loadSkeleton(bodyId) {
  try {
    const buf = await fetchBuffer("/skeleton/" + bodyId);
    const u32 = new Uint32Array(buf);
    const f32 = new Float32Array(buf);
    const nv = u32[0], ns = u32[1];
    const verts = f32.subarray(2, 2 + nv * 3);      // 8-byte header = 2 float words
    const edgeIdx = u32.subarray(2 + nv * 3, 2 + nv * 3 + ns * 2);
    const pts = [];
    for (let e = 0; e < ns; e++) {
      for (let k = 0; k < 2; k++) {
        const vi = edgeIdx[e * 2 + k];
        const p = toScene(verts[vi * 3], verts[vi * 3 + 1], verts[vi * 3 + 2]);
        pts.push(p[0], p[1], p[2]);
      }
    }
    if (morphGroup) { scene.remove(morphGroup); morphGroup = null; }
    morphGroup = new THREE.Group();
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(new Float32Array(pts), 3));
    morphGroup.add(new THREE.LineSegments(g,
      new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.35 })));
    scene.add(morphGroup);
  } catch (e) {
    console.warn("skeleton load failed", e);
  }
}

/* ---------------- firing simulation ---------------- */
async function fireStimulus() {
  if (!somaMesh) return;
  await ensureEdgeArrays();
  const stim = STIMULI.find((s) => s.id === $("stimSelect").value);
  const classes = new Set(stim.cls);
  const visible = somaMesh.userData.count;

  const seeds = [];
  for (let j = 0; j < visible; j++) {
    const n = neurons[neuronOfPoint[j]];
    if (classes.has(n.cls)) seeds.push(neuronOfPoint[j]);
  }
  if (!seeds.length) {
    $("simStatus").textContent = "no matching neurons visible — widen filters";
    return;
  }

  fireArr.fill(0);
  for (const i of seeds) fireArr[i] = 1.0;
  seedHold = new Set(seeds);
  holdUntil = performance.now() + 4000;   // stimulus stays on 4 s

  // first-hop: real postsynaptic partners weighted by synapse count
  const hop = new Map();                          // neuronIdx -> weight sum
  for (const i of seeds) {
    const bodyId = neurons[i].bodyId;
    const r = csrRange(edgeIdxIds, edgeIdxOff, bodyId);
    if (!r) continue;
    for (let k = r[0]; k < r[1]; k++) {
      const idx = neuronIndex.get(edge.post[k]);
      if (idx === undefined) continue;
      hop.set(idx, (hop.get(idx) || 0) + edge.w[k]);
    }
  }
  const MAX = 400;                                // synapse count for full activation
  for (const [idx, w] of hop) {
    const f = Math.min(1, w / MAX);
    if (f > fireArr[idx]) fireArr[idx] = f;
  }
  // second hop: weak downstream excitation from the strongest partners
  const top = [...hop.entries()].filter((p) => p[1] >= MAX * 0.25).slice(0, 60);
  for (const [idx] of top) {
    const r = csrRange(edgeIdxIds, edgeIdxOff, neurons[idx].bodyId);
    if (!r) continue;
    for (let k = r[0]; k < r[1]; k++) {
      const i2 = neuronIndex.get(edge.post[k]);
      if (i2 === undefined) continue;
      const f = Math.min(0.35, (edge.w[k] / MAX) * 0.5);
      if (f > fireArr[i2]) fireArr[i2] = f;
    }
  }

  simActiveUntil = performance.now() + 5000;
  $("simStatus").textContent =
    `⚡ ${fmt(seeds.length)} activated · ${fmt(hop.size)} connected partners respond`;
  setTimeout(() => { if (performance.now() > simActiveUntil) $("simStatus").textContent = ""; }, 5000);
}

function decayAndPaint() {
  if (!somaMesh) return false;
  const fire = somaMesh.geometry.getAttribute("aFire");
  const arr = fire.array;
  let any = false;
  const now = performance.now();
  const n = somaMesh.userData.count;
  for (let j = 0; j < n; j++) {
    const i = neuronOfPoint[j];
    let f = fireArr[i];
    if (now < holdUntil && seedHold.has(i)) {
      f = 1.0;                                   // stimulus still on
      any = true;
    } else if (f > 0.004) {
      f *= 0.992; fireArr[i] = f; any = true;    // slow decay afterwards
    } else if (f !== 0) {
      f = 0; fireArr[i] = 0; any = true;
    }
    arr[j] = f;
  }
  if (any) fire.needsUpdate = true;
  return any || now < simActiveUntil;
}

/* ---------------- search ---------------- */
function doSearch() {
  const q = $("search").value.trim().toLowerCase();
  const res = $("results");
  if (q.length < 2) { res.innerHTML = ""; return; }
  const hits = [];
  for (let i = 0; i < neurons.length && hits.length < 40; i++) {
    const n = neurons[i];
    if ((n.instance && n.instance.toLowerCase().includes(q)) ||
        (n.type && n.type.toLowerCase().includes(q)) ||
        (n.cls && n.cls.toLowerCase().includes(q))) hits.push(i);
  }
  res.innerHTML = hits.length
    ? hits.map((i) => {
        const n = neurons[i];
        return `<div class="res" onclick="selectById(${n.bodyId})">` +
          `<span class="nm">${esc(n.instance || n.type)}</span>` +
          `<span class="cls">${esc(n.cls || n.superclass)}</span></div>`;
      }).join("")
    : "<div class='res'><span class='cls'>no matches</span></div>";
}

/* ---------------- render loop ---------------- */
function animate() {
  requestAnimationFrame(animate);
  controls.update();
  if (somaMesh) somaMesh.material.uniforms.uTime.value = performance.now() / 1000;
  decayAndPaint();
  renderer.render(scene, camera);
}
