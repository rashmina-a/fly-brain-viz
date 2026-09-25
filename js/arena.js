/* Fly Foraging Arena — a 2D world where the fly runs the REAL MaleCNS v1.0
 * olfactory connectome (ORN->PN->KC->MBON + DAN learning).
 *
 * Two modes:
 *   🧬 TRAIN     — a genetic algorithm evolves the fly's sensorimotor policy.
 *                  Episodes run headless (no rendering), food is placed
 *                  automatically and respawns when eaten, at up to 500× speed.
 *                  The best genome autosaves to localStorage after each gen.
 *   🔬 EXPERIMENT — the live arena runs your trained genome. Place food (or
 *                  let it auto-place) and watch the trained fly forage; the
 *                  mushroom body keeps learning DAN-gated as it eats.
 *
 * The live fly's policy is byte-for-byte the policy the GA evolves, and
 * sensing (odor = 1/(1+d/150) from each food) is identical in both, so the
 * trained genome transfers exactly. The diffusion grid is the plume
 * visualization the fly is drawn over.
 */
"use strict";

/* ---------------- config ---------------- */
const CFG = {
  worldW: 1400, worldH: 900,
  grid: 28,
  flySpeed: 105,
  turnRate: 3.6,
  eatDist: 16,
  wind: [14, -6],
  diffusion: 0.10,
  evaporation: 0.994,
  trailEvery: 3,
  maxTrail: 400,
};

const SPEEDS = [1, 2, 5, 10, 25, 50, 100, 250, 500];
const STORE_KEY = "flybrain.ga.v1";

const ODOR_COLORS = {
  apple: "#e57373", banana: "#ffd54f", mango: "#ffb74d",
  yeast: "#aed581", mushroom: "#bcaaa4", vinegar: "#4fc3f7",
};

/* ---------------- global state ---------------- */
let circ = null;
let mode = "train";            // "train" | "exp"
let training = false;
let trainer = null;
let expGenome = null;          // genome the live fly runs in experiment mode
let expMeta = null;            // {gen, score} of the trained genome
let sim = null;                // the one visual Sim (demo during train, live in exp)
let gaPool = null;             // worker pool for parallel GA evaluation
let genRate = 0, lastEvolveT = 0, lastPanelT = 0;
let speedIdx = 0;
let paused = false;
let selectedOdor = "apple";
let frame = 0;
let last = 0;

const $ = (id) => document.getElementById(id);
function toast(msg) {
  const t = $("toast");
  t.textContent = msg;
  t.style.opacity = 1;
  clearTimeout(t._h);
  t._h = setTimeout(() => (t.style.opacity = 0), 2200);
}
function rnd(a, b) { return a + Math.random() * (b - a); }
function clamp(v, a, b) { return v < a ? a : (v > b ? b : v); }
function angDiff(a, b) {
  let d = a - b;
  while (d > Math.PI) d -= 2 * Math.PI;
  while (d < -Math.PI) d += 2 * Math.PI;
  return d;
}

/* ---------------- Sim: one visual world ---------------- */
class Sim {
  constructor(genome) {
    this.genome = genome;
    this.foods = [];
    this.fly = null;
    this.trail = [];
    this.time = 0;
    this.frame = 0;
    this.autoFood = true;
    this.respawnT = 0;
    this.brain = null;
    this.stats = { found: 0, trials: 0, ttfHistory: [], lastFindT: 0, circles: 0, circFlash: 0 };
    this.learnCurve = [];
    this._initGrid();
  }

  _initGrid() {
    const gw = Math.ceil(CFG.worldW / CFG.grid), gh = Math.ceil(CFG.worldH / CFG.grid);
    this.grid = { gw, gh, a: new Float32Array(gw * gh), b: new Float32Array(gw * gh) };
  }

  setBrain(b) { this.brain = b; }

  spawnFly(x, y) {
    this.fly = {
      x: x ?? rnd(80, CFG.worldW - 80),
      y: y ?? rnd(80, CFG.worldH - 80),
      heading: rnd(0, Math.PI * 2),
      angVel: 0, state: "wander",
      wanderT: 0, wanderTurn: 0, castPhase: 0,
      lastOdor: 0, legPhase: 0,
    };
    GA.initCirc(this.fly);
    this.trail.length = 0;
    this.stats.lastFindT = this.time;
  }

  placeFood(x, y, odor) {
    this.foods.push({ x: clamp(x, 10, CFG.worldW - 10), y: clamp(y, 10, CFG.worldH - 10), odor });
    this._foodCount();
  }

  autoPlace() {
    const names = Object.keys(ODOR_COLORS);
    const odor = names[Math.floor(Math.random() * names.length)];
    let x = 0, y = 0;
    for (let t = 0; t < 20; t++) {
      x = rnd(90, CFG.worldW - 90); y = rnd(90, CFG.worldH - 90);
      if (!this.fly || Math.hypot(x - this.fly.x, y - this.fly.y) > 300) break;
    }
    this.placeFood(x, y, odor);
  }

  clearFood() { this.foods = []; this._foodCount(); }

  step(dt) {
    this.time += dt;
    this.frame++;
    this._emit(dt);
    this._diffuse(dt);
    if (this.fly) this._flyStep(dt);
    // auto food
    if (this.autoFood && this.foods.length === 0) {
      this.respawnT -= dt;
      if (this.respawnT <= 0) { this.autoPlace(); this.respawnT = 1.0; }
    }
    if (this.stats.circFlash > 0) this.stats.circFlash = Math.max(0, this.stats.circFlash - dt * 2);
  }

  /* ---- odor grid (visualization only) ---- */
  _emit(dt) {
    for (const f of this.foods) {
      const gx = Math.floor(f.x / CFG.grid), gy = Math.floor(f.y / CFG.grid);
      if (gx >= 0 && gy >= 0 && gx < this.grid.gw && gy < this.grid.gh) {
        const i = gy * this.grid.gw + gx;
        this.grid.a[i] = Math.min(2.5, this.grid.a[i] + dt * 1.4);
      }
    }
  }
  _diffuse(dt) {
    const { gw, gh, a, b } = this.grid;
    const D = CFG.diffusion, evap = CFG.evaporation;
    const wx = CFG.wind[0] * dt / CFG.grid, wy = CFG.wind[1] * dt / CFG.grid;
    for (let y = 1; y < gh - 1; y++) {
      for (let x = 1; x < gw - 1; x++) {
        const i = y * gw + x;
        const lap = a[i - 1] + a[i + 1] + a[i - gw] + a[i + gw] - 4 * a[i];
        let v = a[i] + D * lap * 4;
        const sx = clampI(x - wx, 1, gw - 2), sy = clampI(y - wy, 1, gh - 2);
        v = v * 0.82 + a[clampI(Math.floor(sy), 1, gh - 2) * gw + clampI(Math.floor(sx), 1, gw - 2)] * 0.18;
        b[i] = v * evap;
      }
    }
    this.grid.a.set(b);
  }

  /* ---- the fly: runs GA.policyStep — byte-identical to training ---- */
  _flyStep(dt) {
    const f = this.fly, g = this.genome, b = this.brain;
    f.genome = g;

    const sns = GA.sense(null, f, this.foods);

    // brain sniff (real circuit) — cached per odor, same as training
    let val = b.val;
    if (this.frame % 2 === 0) {
      const tmpl = sns.nearF ? env.odorGlom[sns.nearF.odor] : {};
      val = b.sniffCached(sns.nearF ? sns.nearF.odor : "", tmpl, sns.odor);
    }
    sns.val = val;

    GA.policyStep(f, sns, g, dt, CFG.flySpeed, CFG.turnRate, Math.random);

    // orbiting food is punished: DANs + counter (identical to training)
    const doLearn = $("learnChk").checked;
    if (GA.updateCircling(f, sns, dt, doLearn ? b : null, parseFloat($("lrRange").value) / 1000)) {
      if (doLearn) b.invalidateAllValence();
      this.stats.circles++;
      this.stats.circFlash = 1;
      if (mode === "exp") toast("🌀 circling — punishment DANs fired");
    }

    if (sns.nearD < CFG.eatDist) this._eat(sns.nearF);

    f.legPhase += (f.speed || CFG.flySpeed) * dt * 0.35;
    if (this.frame % CFG.trailEvery === 0) {
      this.trail.push([f.x, f.y]);
      if (this.trail.length > CFG.maxTrail) this.trail.shift();
    }
  }

  _eat(fd) {
    this.foods.splice(this.foods.indexOf(fd), 1);
    GA.resetCirc(this.fly);
    const ttf = this.time - this.stats.lastFindT;
    this.stats.lastFindT = this.time;
    this.stats.found++; this.stats.trials++;
    this.stats.ttfHistory.push(ttf);
    if (this.stats.ttfHistory.length > 40) this.stats.ttfHistory.shift();
    const avg = this.stats.ttfHistory.slice(-10);
    this.learnCurve.push(avg.reduce((a, c) => a + c, 0) / avg.length);
    if (this.learnCurve.length > 120) this.learnCurve.shift();

    if (this.respawnT <= 0) this.respawnT = 1.0;   // auto food replaces eaten item

    // DAN burst -> mushroom-body plasticity (toggleable in the UI)
    this.brain.reward(1.0);
    if ($("learnChk").checked) {
      const lr = parseFloat($("lrRange").value) / 1000;
      let changed = 0;
      for (let i = 0; i < 10; i++) {
        changed += this.brain.learn(lr);
        this.brain.danDecay(0.8);
      }
      this.brain.invalidateOdor(fd.odor);
      if (mode === "exp") {
        const v = this.brain.sniff((circ.odors.find((o) => o.name === fd.odor) || { glom: {} }).glom, 1);
        toast(`🍒 ate ${fd.odor}! DAN burst · ${changed.toLocaleString()} synapses · valence ${v.toFixed(2)}`);
      }
    }
    this._foodCount();
  }

  _foodCount() {
    const el = $("foodCount");
    if (el) el.textContent = this.foods.length + " food item" + (this.foods.length === 1 ? "" : "s") +
      (this.autoFood ? " · auto-respawn ON" : "");
  }
}
function clampI(v, a, b) { return v < a ? a : (v > b ? b : v); }

/* ---------------- GA environment hookup ---------------- */
const env = {
  W: CFG.worldW, H: CFG.worldH,
  flySpeed: CFG.flySpeed, turnRate: CFG.turnRate, eatDist: CFG.eatDist,
  odorNames: Object.keys(ODOR_COLORS),
  odorGlom: {},                 // filled after circuit loads
  rnd: Math.random,
  // one shared brain, plasticity reset between headless episodes (fast)
  makeBrain: () => {
    if (!env._brain) env._brain = new FlyBrain(circ);
    env._brain.resetPlasticity();
    return env._brain;
  },
  setSeed(s) { this._rng = mulberry32(s); this.rnd = this._rng; },
};

/* ---------------- persistence ---------------- */
function saveBest(force) {
  if (!trainer || !trainer.bestGenome) return;
  let cur = null;
  try { cur = JSON.parse(localStorage.getItem(STORE_KEY) || "null"); } catch (e) { cur = null; }
  if (force || !cur || trainer.bestScore > cur.score) {
    localStorage.setItem(STORE_KEY, JSON.stringify({
      genome: trainer.bestGenome,
      score: trainer.bestScore,
      gen: trainer.history.length ? trainer.history[trainer.history.length - 1].gen : 0,
      saved: Date.now(),
    }));
  }
}
function loadSaved() {
  try { return JSON.parse(localStorage.getItem(STORE_KEY) || "null"); } catch (e) { return null; }
}

/* ---------------- training pump ----------------
 * A setTimeout(0) chain that keeps one near-full core on GA evaluation
 * (the rAF loop only time-boxed it to ~10ms/frame). Uses the worker pool
 * when available — nWorkers × single-core speed — else sync slices. */
function gaPump() {
  setTimeout(gaPump, 0);
  if (!training || !trainer || mode !== "train") { lastEvolveT = 0; return; }
  if (trainer.busy) return;
  if (gaPool && gaPool.ready) {
    trainer.evaluateParallel(finishGeneration);
  } else if (trainer.evaluateSlice(30)) {
    finishGeneration();
  }
}
function finishGeneration() {
  const row = trainer.evolve();
  if (row.gen % 5 === 0) saveBest(false);
  sim.genome = GA.cloneGenome(trainer.bestGenome);   // demo fly improves live
  const now = performance.now();
  if (lastEvolveT > 0) {
    const r = 1000 / Math.max(1, now - lastEvolveT);
    genRate = genRate ? genRate * 0.85 + r * 0.15 : r;
  }
  lastEvolveT = now;
  if (now - lastPanelT > 150) {
    lastPanelT = now;
    updateTrainingPanel();
    updateTrainingProgress();
    $("genRate").textContent = genRate ? genRate.toFixed(0) + " gen/s" : "…";
  }
}

/* ---------------- rendering ---------------- */
let ctx, canvas;
function resize() {
  canvas.width = $("world").clientWidth;
  canvas.height = $("world").clientHeight;
}
function worldToScreen(x, y) {
  const s = Math.min(canvas.width / CFG.worldW, canvas.height / CFG.worldH);
  const ox = (canvas.width - CFG.worldW * s) / 2;
  const oy = (canvas.height - CFG.worldH * s) / 2;
  return [ox + x * s, oy + y * s, s];
}

function draw() {
  ctx.fillStyle = "#0b0e14";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  if (!sim || !sim.fly) return;

  const s = Math.min(canvas.width / CFG.worldW, canvas.height / CFG.worldH);
  // plume (visualization grid)
  for (let gy = 0; gy < sim.grid.gh; gy++) {
    for (let gx = 0; gx < sim.grid.gw; gx++) {
      const v = sim.grid.a[gy * sim.grid.gw + gx];
      if (v < 0.02) continue;
      ctx.fillStyle = `rgba(79,195,247,${Math.min(0.16, v * 0.07)})`;
      const [sx, sy] = worldToScreen(gx * CFG.grid, gy * CFG.grid);
      ctx.fillRect(sx, sy, CFG.grid * s + 1, CFG.grid * s + 1);
    }
  }

  // trail
  if ($("showTrail").checked && sim.trail.length > 1) {
    ctx.strokeStyle = "rgba(215,222,233,0.25)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    const [tx, ty] = worldToScreen(sim.trail[0][0], sim.trail[0][1]);
    ctx.moveTo(tx, ty);
    for (const [x, y] of sim.trail) {
      const [sx2, sy2] = worldToScreen(x, y);
      ctx.lineTo(sx2, sy2);
    }
    ctx.stroke();
  }

  // food
  for (const f of sim.foods) {
    const [sx, sy, sc] = worldToScreen(f.x, f.y);
    const col = ODOR_COLORS[f.odor] || "#fff";
    const pulse = 1 + 0.15 * Math.sin(sim.time * 4 + f.x);
    ctx.beginPath(); ctx.arc(sx, sy, 9 * sc * pulse, 0, 7);
    ctx.fillStyle = col; ctx.globalAlpha = 0.25; ctx.fill();
    ctx.globalAlpha = 1;
    ctx.beginPath(); ctx.arc(sx, sy, 5 * sc * pulse, 0, 7);
    ctx.fillStyle = col; ctx.fill();
    ctx.fillStyle = "#0b0e14";
    ctx.font = `${Math.round(8 * sc)}px sans-serif`;
    ctx.textAlign = "center";
    ctx.fillText(f.odor[0].toUpperCase(), sx, sy + 3 * sc);
  }

  // fly
  const f = sim.fly;
  const [fx, fy, fsc] = worldToScreen(f.x, f.y);
  ctx.save();
  ctx.translate(fx, fy);
  ctx.rotate(f.heading);
  ctx.fillStyle = "#d7dee9";
  ctx.beginPath(); ctx.ellipse(0, 0, 7 * fsc, 3.2 * fsc, 0, 0, 7); ctx.fill();
  const wing = f.state === "surge" ? 0.5 : f.state === "cast" ? 0.85 : 0.65;
  ctx.fillStyle = "rgba(215,222,233,0.5)";
  ctx.beginPath();
  ctx.ellipse(-2 * fsc, -5 * fsc * wing, 4.5 * fsc, 1.8 * fsc, -0.5, 0, 7);
  ctx.ellipse(-2 * fsc, 5 * fsc * wing, 4.5 * fsc, 1.8 * fsc, 0.5, 0, 7);
  ctx.fill();
  ctx.strokeStyle = f.state === "surge" ? "#ff7043" : f.state === "cast" ? "#4fc3f7" : "#8b95a7";
  ctx.lineWidth = 1.2;
  ctx.beginPath();
  ctx.moveTo(5 * fsc, -1 * fsc); ctx.lineTo(10 * fsc, -3.5 * fsc);
  ctx.moveTo(5 * fsc, 1 * fsc); ctx.lineTo(10 * fsc, 3.5 * fsc);
  ctx.stroke();
  ctx.restore();

  // circling punishment flash: red dashed ring around the fly
  if (sim.stats.circFlash > 0) {
    ctx.strokeStyle = `rgba(229,115,115,${(sim.stats.circFlash * 0.9).toFixed(2)})`;
    ctx.lineWidth = 2;
    ctx.setLineDash([4, 3]);
    ctx.beginPath();
    ctx.arc(fx, fy, 16 * fsc, 0, 7);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = "#e57373";
    ctx.font = "bold 11px sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("CIRCLING ✕", fx, fy - 20 * fsc);
  }

  // status line
  ctx.fillStyle = "#8b95a7";
  ctx.font = "11px sans-serif";
  ctx.textAlign = "left";
  const speedTxt = SPEEDS[speedIdx] + "×";
  if (mode === "train") {
    const g = trainer;
    ctx.fillText(
      `🧬 TRAINING demo fly · best genome ${g && g.bestGenome ? "gen " + (g.history.length ? g.history[g.history.length - 1].gen : 0) : "—"}`
      + `   state: ${f.state}   odor: ${f.lastOdor.toFixed(2)}   speed ${speedTxt}`,
      12, canvas.height - 12);
    // top-center progress banner
    ctx.textAlign = "center";
    ctx.fillStyle = "#4fc3f7";
    ctx.font = "bold 12px sans-serif";
    const prog = g ? (g.evalIdx / g.popSize * 100).toFixed(0) : 0;
    ctx.fillText(
      g ? `GENERATION ${g.gen} · evaluating fly ${g.evalIdx + 1}/${g.popSize} (${prog}%) · best score ${g.bestScore === -1e9 ? "—" : g.bestScore.toFixed(1)}`
        : "press ▶ Start training to evolve the sensorimotor policy",
      canvas.width / 2, 22);
  } else {
    ctx.fillText(
      `🔬 experiment · ${expMeta ? `trained genome (gen ${expMeta.gen}, score ${expMeta.score})` : "naive genome"}`
      + `   state: ${f.state}   odor: ${f.lastOdor.toFixed(2)}   speed ${speedTxt}`,
      12, canvas.height - 12);
  }
}

/* ---------------- brain monitor ---------------- */
function updateBrainPanel() {
  if (!sim || !sim.brain || !sim.fly) return;
  const b = sim.brain;
  const top = (arr, chipsId, barId, k) => {
    const idx = Array.from(arr.keys());
    idx.sort((a, c) => arr[c] - arr[a]);
    let html = "";
    for (let i = 0; i < Math.min(k, idx.length); i++) {
      const v = arr[idx[i]];
      if (v < 0.02) break;
      html += `<span class="chip ${v > 0.5 ? "on" : ""}">${esc(nameOf(arr, idx[i]))}</span>`;
    }
    $(chipsId).innerHTML = html || "<span class='hint'>quiet</span>";
    let sum = 0;
    for (let i = 0; i < arr.length; i++) sum += arr[i];
    $(barId).style.width = (Math.min(1, sum / (0.25 * arr.length)) * 100).toFixed(0) + "%";
  };
  top(b.pn, "pnChips", "pnBar", 6);
  top(b.kc, "kcChips", "kcBar", 8);
  top(b.mbon, "mbChips", "mbBar", 5);
  top(b.dan, "danChips", "danBar", 5);

  const rows = [];
  for (const f of sim.foods) {
    const d = Math.hypot(f.x - sim.fly.x, f.y - sim.fly.y);
    const c = Math.max(0, 1 / (1 + d / 150));
    rows.push(`<div class="od"><span style="color:${ODOR_COLORS[f.odor]}">${f.odor}</span><span>${(c * 100).toFixed(0)}%</span></div>`);
  }
  $("sniffRows").innerHTML = rows.join("") || "<span class='hint'>no food in world</span>";
  $("valVal").textContent = b.valence.toFixed(2);
  $("modeVal").textContent = sim.fly.state;
}
function nameOf(arr, i) {
  const b = sim.brain;
  if (arr === b.pn) return b.c.pn[i].name;
  if (arr === b.kc) return "KC" + i;
  if (arr === b.mbon) return b.c.mbon[i].name.replace(/\(.*\)/, "");
  if (arr === b.dan) return b.c.dan[i].name.replace(/\(.*\)/, "");
  return i;
}
function esc(s) { return String(s); }

function updateLearningPanel() {
  if (!sim) return;
  $("foundCount").textContent = sim.stats.found;
  $("trials").textContent = sim.stats.trials;
  $("circCount").textContent = sim.stats.circles;
  const h = sim.stats.ttfHistory.slice(-10);
  $("ttf").textContent = h.length ? (h.reduce((a, b) => a + b, 0) / h.length).toFixed(1) + "s" : "–";
  const c = $("lc");
  const g = c.getContext("2d");
  g.clearRect(0, 0, c.width, c.height);
  const curve = sim.learnCurve;
  if (curve.length < 2) return;
  const max = Math.max(...curve, 5);
  g.strokeStyle = "#81c784";
  g.lineWidth = 1.5;
  g.beginPath();
  curve.forEach((v, i) => {
    const x = (i / (curve.length - 1)) * (c.width - 4) + 2;
    const y = c.height - 4 - (v / max) * (c.height - 8);
    i ? g.lineTo(x, y) : g.moveTo(x, y);
  });
  g.stroke();
}

/* ---------------- training panel ---------------- */
function updateTrainingPanel() {
  if (!trainer) return;
  const h = trainer.history;
  const lastRow = h[h.length - 1];
  $("genVal").textContent = trainer.gen;
  $("bestVal").textContent = lastRow ? lastRow.best : "–";
  $("avgVal").textContent = lastRow ? lastRow.avg : "–";
  $("foundGenVal").textContent = lastRow ? lastRow.bestFound + " / " + lastRow.avgFound.toFixed(1) : "–";
  $("circGenVal").textContent = lastRow ? lastRow.bestCirc + " / " + lastRow.avgCirc.toFixed(1) : "–";
  $("succVal").textContent = lastRow ? lastRow.succ.toFixed(0) + "%" : "–";
  drawGenChart();
  drawSuccChart();
}
function _histMax(h, key) {
  let m = 10;
  for (let i = 0; i < h.length; i++) if (h[i][key] > m) m = h[i][key];
  return m;
}
function drawGenChart() {
  const c = $("genChart");
  const g = c.getContext("2d");
  g.clearRect(0, 0, c.width, c.height);
  const h = trainer ? trainer.history : [];
  if (h.length < 2) return;
  const max = _histMax(h, "best");
  const step = Math.max(1, Math.floor(h.length / 240));   // downsample long runs
  const plot = (key, color) => {
    g.strokeStyle = color;
    g.lineWidth = 1.5;
    g.beginPath();
    let first = true;
    for (let i = 0; i < h.length; i += step) {
      const r = h[Math.min(i, h.length - 1)];
      const x = (i / (h.length - 1)) * (c.width - 4) + 2;
      const y = c.height - 4 - (r[key] / max) * (c.height - 8);
      first ? (g.moveTo(x, y), first = false) : g.lineTo(x, y);
    }
    g.stroke();
  };
  plot("avg", "#8b95a7");
  plot("best", "#81c784");
  g.fillStyle = "#8b95a7";
  g.font = "9px sans-serif";
  g.textAlign = "left";
  g.fillText("best", 4, 10);
  g.fillStyle = "#4fc3f7";
  g.fillText("avg", 30, 10);
}
function drawSuccChart() {
  const c = $("succChart");
  if (!c) return;
  const g = c.getContext("2d");
  g.clearRect(0, 0, c.width, c.height);
  const h = trainer ? trainer.history : [];
  if (h.length < 2) return;
  const step = Math.max(1, Math.floor(h.length / 240));
  g.strokeStyle = "rgba(139,149,167,0.25)";   // 50% gridline
  g.lineWidth = 1;
  g.beginPath(); g.moveTo(0, c.height / 2); g.lineTo(c.width, c.height / 2); g.stroke();
  g.strokeStyle = "#81c784";
  g.lineWidth = 1.5;
  g.beginPath();
  let first = true;
  for (let i = 0; i < h.length; i += step) {
    const r = h[Math.min(i, h.length - 1)];
    const x = (i / (h.length - 1)) * (c.width - 4) + 2;
    const y = c.height - 4 - (r.succ / 100) * (c.height - 8);
    first ? (g.moveTo(x, y), first = false) : g.lineTo(x, y);
  }
  g.stroke();
  g.fillStyle = "#8b95a7";
  g.font = "9px sans-serif";
  g.textAlign = "left";
  g.fillText("success rate % — flies that found food", 4, 10);
}
function updateTrainingProgress() {
  if (!trainer) return;
  $("memberVal").textContent = `${Math.min(trainer.evalIdx + 1, trainer.popSize)}/${trainer.popSize}`;
}

/* ---------------- mode / genome ---------------- */
function setMode(m) {
  mode = m;
  $("tabTrain").classList.toggle("active", m === "train");
  $("tabExp").classList.toggle("active", m === "exp");
  $("trainSec").style.display = m === "train" ? "" : "none";
  $("expSec").style.display = m === "exp" ? "" : "none";
  sim.genome = m === "exp"
    ? expGenome
    : (trainer && trainer.bestGenome ? GA.cloneGenome(trainer.bestGenome) : expGenome);
  if (m === "exp") {
    updateGenomeBadge();
    updateLearningPanel();
  }
}
function updateGenomeBadge() {
  const el = $("genomeBadge");
  if (expMeta) {
    el.textContent = `🧬 trained genome loaded — gen ${expMeta.gen}, GA score ${Number(expMeta.score).toFixed(1)}`;
    el.style.color = "var(--good)";
  } else {
    el.textContent = "fly runs the naive (hand-tuned) genome";
    el.style.color = "var(--dim)";
  }
}
function useGenome(g, meta) {
  expGenome = GA.cloneGenome(g);
  expMeta = meta || null;
  if (mode === "exp") { sim.genome = expGenome; updateGenomeBadge(); }
}

/* ---------------- UI wiring ---------------- */
function initUI() {
  const sel = $("odorSel");
  for (const o of Object.keys(ODOR_COLORS)) {
    const opt = document.createElement("option");
    opt.value = o; opt.textContent = o;
    sel.appendChild(opt);
  }
  sel.onchange = () => (selectedOdor = sel.value);
  $("placeBtn").onclick = () => toast("click the arena to place " + selectedOdor);
  $("randomBtn").onclick = () => sim.placeFood(rnd(100, CFG.worldW - 100), rnd(100, CFG.worldH - 100), selectedOdor);
  $("clearFoodBtn").onclick = () => sim.clearFood();
  $("autoFoodChk").onchange = (e) => { sim.autoFood = e.target.checked; sim._foodCount(); };
  $("spawnBtn").onclick = () => sim.spawnFly();
  $("resetBrainBtn").onclick = () => {
    sim.brain.resetPlasticity();
    sim.brain.invalidateAllValence();
    sim.stats = { found: 0, trials: 0, ttfHistory: [], lastFindT: sim.time, circles: 0, circFlash: 0 };
    sim.learnCurve = [];
    updateLearningPanel();
    toast("brain reset to naive connectome");
  };
  $("testBtn").onclick = () => {
    const html = circ.odors.map((o) => `${o.name}: ${sim.brain.sniff(o.glom, 1).toFixed(2)}`);
    toast("learned valence → " + html.join(" · "));
  };
  $("lrRange").oninput = () => ($("lrVal").textContent = ($("lrRange").value / 1000).toFixed(3));

  // tabs
  $("tabTrain").onclick = () => setMode("train");
  $("tabExp").onclick = () => setMode("exp");

  // training controls
  $("trainBtn").onclick = () => {
    if (!trainer) {
      trainer = new GA.GATrainer(env, {
        popSize: 16, elite: 3, episodeT: 15, nFood: 3, lr: 0.018,
      });
      if (gaPool) trainer.attachPool(gaPool);
    }
    training = !training;
    $("trainBtn").textContent = training ? "⏸ Pause training" : "▶ Start training";
    $("trainBtn").classList.toggle("primary", !training);
    if (training) toast("GA training — episodes run headless; watch the demo fly improve");
  };
  $("resetGABtn").onclick = () => {
    if (trainer) trainer.reset();
    genRate = 0; lastEvolveT = 0;
    $("genRate").textContent = "–";
    updateTrainingPanel();
    toast("population reset — evolution restarts from random genomes");
  };
  $("saveBtn").onclick = () => { saveBest(true); toast("best genome saved to this browser"); };
  $("applyBtn").onclick = () => {
    if (trainer && trainer.bestGenome) {
      const h = trainer.history;
      useGenome(trainer.bestGenome, { gen: h.length ? h[h.length - 1].gen : 0, score: trainer.bestScore });
      saveBest(true);
      setMode("exp");
      toast("best genome loaded — you are now experimenting with the trained fly");
    } else toast("no genome yet — run training first");
  };
  $("loadGenBtn").onclick = () => {
    const s = loadSaved();
    if (s) { useGenome(s.genome, { gen: s.gen, score: s.score }); toast(`trained genome loaded (gen ${s.gen}, score ${Number(s.score).toFixed(1)})`); }
    else toast("no saved genome in this browser yet");
  };
  $("naiveGenBtn").onclick = () => { useGenome(GA.DEFAULT_GENOME, null); toast("switched to the naive genome"); };
  $("importBtn").onclick = () => $("importFile").click();

  // export / import genome
  $("exportBtn").onclick = () => {
    const s = loadSaved();
    if (!s) { toast("nothing saved yet"); return; }
    const blob = new Blob([JSON.stringify(s, null, 2)], { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "trained-fly-genome.json";
    a.click();
    URL.revokeObjectURL(a.href);
  };
  $("importFile").onchange = (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const rd = new FileReader();
    rd.onload = () => {
      try {
        const s = JSON.parse(rd.result);
        if (!s.genome) throw new Error("no genome");
        localStorage.setItem(STORE_KEY, JSON.stringify(s));
        useGenome(s.genome, { gen: s.gen || 0, score: s.score || 0 });
        toast("genome imported ✓");
      } catch (err) { toast("import failed: not a genome file"); }
    };
    rd.readAsText(file);
    e.target.value = "";
  };

  // sim controls
  $("pauseBtn").onclick = () => {
    paused = !paused;
    $("pauseBtn").textContent = paused ? "▶ Play" : "⏸ Pause";
  };
  const sp = $("speedRange");
  sp.oninput = () => {
    speedIdx = parseInt(sp.value, 10);
    $("speedVal").textContent = SPEEDS[speedIdx] + "×";
  };

  // arena clicks
  canvas.addEventListener("pointerdown", (e) => {
    const rect = canvas.getBoundingClientRect();
    const s = Math.min(canvas.width / CFG.worldW, canvas.height / CFG.worldH);
    const ox = (canvas.width - CFG.worldW * s) / 2;
    const oy = (canvas.height - CFG.worldH * s) / 2;
    const wx = (e.clientX - rect.left - ox) / s;
    const wy = (e.clientY - rect.top - oy) / s;
    if (mode === "train") { toast("training runs headless — switch to 🔬 Experiment to place food"); return; }
    if (e.shiftKey) { sim.spawnFly(wx, wy); return; }
    sim.autoFood = $("autoFoodChk").checked;
    sim.placeFood(wx, wy, selectedOdor);
  });
  window.addEventListener("resize", resize);
}

/* ---------------- boot + loop ---------------- */
async function boot() {
  canvas = $("arena");
  ctx = canvas.getContext("2d");
  resize();
  initUI();

  const r = await fetch("data/olfactory_circuit.json");
  circ = prepCircuit(await r.json());
  for (const o of circ.odors) env.odorGlom[o.name] = o.glom;

  // worker pool for parallel GA evaluation (sync slice fallback otherwise)
  if (GA.createGAPool) {
    const nW = Math.min(6, Math.max(2, (navigator.hardwareConcurrency || 4) - 2));
    gaPool = GA.createGAPool(circ, {
      W: CFG.worldW, H: CFG.worldH, flySpeed: CFG.flySpeed,
      turnRate: CFG.turnRate, eatDist: CFG.eatDist,
    }, nW);
  }

  // genome from storage or naive default
  const saved = loadSaved();
  expGenome = saved ? GA.cloneGenome(saved.genome) : GA.cloneGenome(GA.DEFAULT_GENOME);
  expMeta = saved ? { gen: saved.gen, score: saved.score } : null;

  sim = new Sim(GA.cloneGenome(expGenome));
  sim.setBrain(new FlyBrain(circ));
  sim.spawnFly(CFG.worldW * 0.5, CFG.worldH * 0.5);
  sim.autoPlace();
  sim._foodCount();
  updateGenomeBadge();
  setMode("train");

  function loop(now) {
    requestAnimationFrame(loop);
    const dtReal = Math.min(0.05, (now - last) / 1000) || 1 / 60;
    last = now;
    const S = SPEEDS[speedIdx];
    if (!paused) {
      const steps = Math.max(1, Math.round(S * dtReal * 60));
      const sub = dtReal * S / steps;
      for (let i = 0; i < steps; i++) sim.step(sub);
      frame += steps;
    }
    draw();
    if (frame % 8 === 0) { updateBrainPanel(); updateLearningPanel(); }
  }
  requestAnimationFrame(loop);
  gaPump();
}

boot();
