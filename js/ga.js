/* Genetic Algorithm for the Fly Foraging Arena.
 *
 * A genome encodes the fly's sensorimotor policy: how it steers using two
 * antenna samples (left/right odor — real flies compare bilateral input),
 * the odor gradient, and the learned MBON valence from the REAL MaleCNS
 * connectome brain. Fitness = food eaten, weighted by speed of finding.
 * Food is placed automatically and respawns when eaten.
 *
 * The locomotor policy lives in ONE shared function (policyStep) used
 * byte-for-byte by both headless GA episodes and the live arena, so the
 * trained genome transfers exactly.
 *
 * Worlds are seeded: every genome in a generation faces identical food
 * placements (fair selection); the world rotates every 8 generations so
 * winners must generalize. Best genome autosaves after every generation.
 */
"use strict";

/* ---------------- seeded RNG (fair, repeatable episodes) ---------------- */
function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/* ---------------- genome ---------------- */
const GENE_KEYS = [
  "surgeGain",      // turn strength from odor trend (getting warmer)
  "bilatGain",      // turn strength from left/right antenna contrast
  "bilatReach",     // antenna sample point distance ahead (world units)
  "gradGain",       // extra weight on raw d(odor)/dt
  "castGain",       // cross-plume sweep amplitude after losing odor
  "castFreq",       // sweep frequency
  "castHold",       // how long (s) to cast after losing the plume
  "wanderGain",     // random-walk turn amplitude
  "wanderHold",     // how long a wander turn is held (s)
  "surgeSpeed",     // speed multiplier while tracking odor
  "cruiseSpeed",    // speed multiplier while casting / wandering
  "odorThresh",     // odor level that counts as "smelling food"
  "valGain",        // how much learned valence boosts surge persistence
  "closeBias",      // pull toward food direction when very close
  "turnSmooth",     // angular inertia (0 = twitchy, 1 = heavy)
];

const GENE_DEF = {
  surgeGain:  [0.0, 4.0],
  bilatGain:  [0.5, 20.0],
  bilatReach: [6.0, 40.0],
  gradGain:   [0.0, 4.0],
  castGain:   [0.3, 3.0],
  castFreq:   [0.6, 5.0],
  castHold:   [0.2, 3.0],
  wanderGain: [0.05, 2.0],
  wanderHold: [0.15, 3.0],
  surgeSpeed: [0.8, 2.2],
  cruiseSpeed:[0.5, 1.6],
  odorThresh: [0.004, 0.12],
  valGain:    [0.0, 3.0],
  closeBias:  [0.0, 1.5],
  turnSmooth: [0.05, 0.6],
};

function randomGenome(rng = Math.random) {
  const g = {};
  for (const k of GENE_KEYS) {
    const [lo, hi] = GENE_DEF[k];
    g[k] = lo + rng() * (hi - lo);
  }
  return g;
}

function mutate(genome, rate, amount, rng = Math.random) {
  const g = {};
  for (const k of GENE_KEYS) {
    const [lo, hi] = GENE_DEF[k];
    let v = genome[k];
    if (rng() < rate) {
      const u = rng() + rng() - 1;            // triangular noise
      v = v + u * amount * (hi - lo);
    }
    g[k] = Math.min(hi, Math.max(lo, v));
  }
  return g;
}

function crossover(a, b, rng = Math.random) {
  const g = {};
  for (const k of GENE_KEYS) g[k] = rng() < 0.5 ? a[k] : b[k];
  return g;
}

function cloneGenome(g) {
  const c = {};
  for (const k of GENE_KEYS) c[k] = g[k];
  return c;
}

/* sensible starting genome — pre-evolved offline for 40 generations on the
 * real circuit (scripts/evolve_default.js); ~12× better foraging than the
 * original hand-tuned constants. The GA improves from here. */
const DEFAULT_GENOME = {
  surgeGain: 0.326, bilatGain: 20.0, bilatReach: 40, gradGain: 0.36,
  castGain: 0.3, castFreq: 1.175, castHold: 2.776,
  wanderGain: 0.25, wanderHold: 0.505,
  surgeSpeed: 2.2, cruiseSpeed: 1.549,
  odorThresh: 0.0633, valGain: 2.645, closeBias: 0.76, turnSmooth: 0.6,
};

/* ---------------- shared senses & policy ---------------- */
/* odor concentration at a point: same falloff everywhere */
function senseAt(env, x, y, foods) {
  let v = 0;
  for (const fd of foods) {
    const d = Math.hypot(fd.x - x, fd.y - y);
    const c = Math.max(0, 1 / (1 + d / 120) - 0.06) * 1.4;
    if (c > v) v = c;
  }
  return Math.min(1, v);
}

/* bilateral + trend senses for the fly. Returns a SHARED scratch object
 * (GC-free hot path) — consume it before the next sense() call. */
const _SENSE = { odor: 0, bilat: 0, nearD: 0, nearF: null };
function sense(env, f, foods) {
  const reach = f.genome ? f.genome.bilatReach : DEFAULT_GENOME.bilatReach;
  const lx = f.x + Math.cos(f.heading - 0.45) * reach;
  const ly = f.y + Math.sin(f.heading - 0.45) * reach;
  const rx = f.x + Math.cos(f.heading + 0.45) * reach;
  const ry = f.y + Math.sin(f.heading + 0.45) * reach;
  let odor = 0, nearF = null, nearD = 1e9;
  for (const fd of foods) {
    const dx = fd.x - f.x, dy = fd.y - f.y;
    const d = Math.sqrt(dx * dx + dy * dy);
    if (d < nearD) { nearD = d; nearF = fd; }
    const c = Math.max(0, 1 / (1 + d / 120) - 0.06) * 1.4;
    if (c > odor) odor = c;
  }
  _SENSE.odor = Math.min(1, odor);
  _SENSE.bilat = senseAt(env, rx, ry, foods) - senseAt(env, lx, ly, foods);
  _SENSE.nearD = nearD; _SENSE.nearF = nearF;
  return _SENSE;
}

/* THE locomotor policy — identical for GA episodes and the live arena.
 * f: stateful fly {x,y,heading,angVel,state,...}; s: senses; g: genome;
 * dt: physics step; baseSpeed: world units/s; rng: seeded random fn.
 * valence (learned MBON drive) arrives in s.val. */
function policyStep(f, s, g, dt, baseSpeed, turnRateMax, rng) {
  let desiredTurn = 0;
  const grad = s.odor - (f.lastOdor || 0);
  f.lastOdor = s.odor;

  if (s.odor > g.odorThresh) {
    f.lostT = 0;
    f.state = "surge";
    // steer by bilateral contrast + odor trend; learned valence sharpens persistence
    desiredTurn = g.bilatGain * s.bilat + (g.surgeGain + g.gradGain) * grad
                * (1.1 + Math.max(0, s.val || 0) * g.valGain);
    if (s.nearD < 90 && s.nearF) {
      const ang = Math.atan2(s.nearF.y - f.y, s.nearF.x - f.x);
      desiredTurn += angDiffS(ang, f.heading) * g.closeBias;
    }
  } else if ((f.lostT || 0) < g.castHold) {
    // just lost the plume: sweep side to side across where it was
    f.lostT = (f.lostT || 0) + dt;
    f.state = "cast";
    f.castPhase = (f.castPhase || 0) + dt * g.castFreq * 2.4;
    desiredTurn = Math.sin(f.castPhase) * g.castGain;
  } else {
    // nothing to smell: Levy-ish wandering
    f.state = "wander";
    f.wanderT = (f.wanderT || 0) - dt;
    if (f.wanderT <= 0) {
      f.wanderT = g.wanderHold * (0.5 + rng());
      f.wanderTurn = (rng() * 2 - 1) * g.wanderGain;
    }
    desiredTurn = f.wanderTurn || 0;
  }

  f.angVel += (desiredTurn - f.angVel) * Math.min(1, dt * g.turnSmooth * 12);
  f.angVel = Math.max(-turnRateMax, Math.min(turnRateMax, f.angVel));
  f.heading += f.angVel * dt;

  const sp = baseSpeed * (f.state === "surge" ? g.surgeSpeed : g.cruiseSpeed);
  f.x += Math.cos(f.heading) * sp * dt;
  f.y += Math.sin(f.heading) * sp * dt;
  f.speed = sp;

  // walls: reflect
  if (f.x < 20 || f.x > env_W(g) - 20) { f.heading = Math.PI - f.heading; f.x = Math.max(20, Math.min(env_W(g) - 20, f.x)); }
  if (f.y < 20 || f.y > env_H(g) - 20) { f.heading = -f.heading; f.y = Math.max(20, Math.min(env_H(g) - 20, f.y)); }
  return sp;
}

/* world bounds ride on the genome object slot _W/_H set per environment */
function env_W(g) { return g._W || 1400; }
function env_H(g) { return g._H || 900; }

/* ---------------- circling detector (shared, punished) ----------------
 * Orbiting food instead of landing on it is wasteful dithering: the fly
 * accumulates the angle swept AROUND the nearest food while inside CIRC.R.
 * A full revolution (2π) inside that ring = one circling event, which costs
 * GA fitness AND fires punishment DANs (PPL1-style) that depress the
 * approach MBONs for the odor currently being sniffed — the mushroom body
 * learns "orbiting this smell gets me nowhere". */
const CIRC = {
  R: 70,        // orbit radius that counts as "circling around food"
  EXIT: 1.6,    // accumulator only wipes when the fly leaves R*EXIT (wobble grace)
  TURNS: 1.0,   // full revolutions required to flag an event
  COOL: 2.0,    // s of cooldown after an event
  PEN: 2.0,     // GA fitness points lost per event
  PUNISH: 0.7,  // punishment DAN burst strength (vs +1.0 reward on eating)
};

function initCirc(f) { f.circ = { ang: 0, lastA: null, cool: 0, events: 0 }; }

/* call once per physics step AFTER the move; returns 1 on a circling event */
function updateCircling(f, sns, dt, brain, lr) {
  if (!f.circ) initCirc(f);
  const c = f.circ;
  c.cool = Math.max(0, c.cool - dt);
  const fd = sns.nearF;
  if (!fd || sns.nearD > CIRC.R * CIRC.EXIT) { c.ang = 0; c.lastA = null; return 0; }
  if (sns.nearD > CIRC.R) return 0;   // grace ring: hold the sweep, don't add
  if (c.cool > 0) { c.lastA = null; return 0; }
  const a = Math.atan2(f.y - fd.y, f.x - fd.x);
  if (c.lastA === null) { c.lastA = a; return 0; }
  c.ang += angDiffS(a, c.lastA);
  c.lastA = a;
  if (Math.abs(c.ang) < 2 * Math.PI * CIRC.TURNS) return 0;
  c.events++; c.ang = 0; c.lastA = null; c.cool = CIRC.COOL;
  if (brain) {
    brain.reward(-CIRC.PUNISH);
    for (let i = 0; i < 4; i++) { brain.learn(lr || 0.018); brain.danDecay(0.8); }
  }
  return 1;
}

/* clear the accumulator after a successful approach (eating) */
function resetCirc(f) { if (f.circ) { f.circ.ang = 0; f.circ.lastA = null; } }

/* ---------------- headless episode ----------------
 * Tuned for thousands of generations per minute:
 *   - coarse physics (dt = 1/10) with a midpoint eat check so fast flies
 *     can't tunnel through food between samples;
 *   - valence CACHE: sniffing the real circuit (272 PN -> 4064 KC -> 97 MBON)
 *     is the hot path, but learned valence only changes when plasticity
 *     fires — so each odor's valence is sniffed once per weight-epoch and
 *     reused. Brain._epoch bumps on every weight change (learn/reset).
 * One foraging episode with a given genome. A fresh (plasticity-reset) brain
 * is used each episode — GA evolves the policy, while MBON learning still
 * runs inside episodes so evolution can also discover faster chemistry.
 * Food auto-respawns when eaten.
 * Returns { fitness, found, avgTTF, steps, circles }. */
function runEpisode(env, genome, opts = {}) {
  const brain = env.makeBrain();
  const episodeT = opts.episodeT || 15;         // short episodes = many gens
  const dt = opts.dt || 1 / 10;                 // coarse headless physics
  const maxSteps = Math.round(episodeT / dt);
  const nFood = opts.nFood || 3;
  const odorNames = env.odorNames;
  const eatDist = env.eatDist;
  const g = Object.assign({ _W: env.W, _H: env.H }, genome);

  const spawnFood = () => {
    let x = 0, y = 0;
    for (let tries = 0; tries < 20; tries++) {
      x = 90 + env.rnd() * (env.W - 180);
      y = 90 + env.rnd() * (env.H - 180);
      if (Math.hypot(x - f.x, y - f.y) > 260) break;
    }
    foods.push({ x, y, odor: odorNames[Math.floor(env.rnd() * odorNames.length)] });
  };

  const foods = [];
  const f = {
    x: 60 + env.rnd() * (env.W - 120), y: 60 + env.rnd() * (env.H - 120),
    heading: env.rnd() * Math.PI * 2, angVel: 0,
    state: "wander", lastOdor: 0,
  };
  for (let i = 0; i < nFood; i++) spawnFood();

  let found = 0, ttfSum = 0, lastFind = 0, steps = 0, circEvents = 0;

  for (let s = 0; s < maxSteps; s++) {
    steps++;
    const t = s * dt;
    f.genome = g;
    const sns = sense(env, f, foods);

    // learned valence from the real circuit — persistent per-brain cache,
    // only the eaten/punished odor ever re-sniffs
    let val = brain.val;
    if (sns.odor > g.odorThresh && s % 3 === 0) {
      const tmpl = sns.nearF ? env.odorGlom[sns.nearF.odor] : {};
      val = brain.sniffCached(sns.nearF ? sns.nearF.odor : "", tmpl, sns.odor);
    }

    sns.val = val;
    const px = f.x, py = f.y;
    policyStep(f, sns, g, dt, env.flySpeed, env.turnRate, env.rnd);

    // orbiting food is punished: fitness penalty + punishment DANs
    const circled = updateCircling(f, sns, dt, brain, opts.lr || 0.018);
    if (circled) { circEvents++; brain.invalidateAllValence(); }

    // eat? (midpoint test closes the tunneling gap at coarse dt) -> respawn
    let ate = sns.nearD < eatDist;
    if (!ate) {
      const mx = (px + f.x) / 2 - sns.nearF.x, my = (py + f.y) / 2 - sns.nearF.y;
      ate = mx * mx + my * my < eatDist * eatDist;
    }
    if (ate) {
      resetCirc(f);
      found++;
      ttfSum += t - lastFind;
      lastFind = t;
      foods.splice(foods.indexOf(sns.nearF), 1);
      const ateOdor = sns.nearF.odor;
      spawnFood();
      brain.reward(1.0);
      // 4 gated learn iterations at 1.5× lr ≈ the old 10 @ lr (gate decays 0.8)
      const lr = (opts.lr || 0.018) * 1.5;
      for (let i = 0; i < 4; i++) { brain.learn(lr); brain.danDecay(0.8); }
      brain.invalidateOdor(ateOdor);   // only the eaten odor's valence moved
    }
  }

  const avgTTF = found ? ttfSum / found : episodeT;
  const fitness = found * 10 + Math.max(0, episodeT - avgTTF) * 0.2
                - circEvents * CIRC.PEN;
  return { fitness, found, avgTTF, steps, circles: circEvents };
}

function angDiffS(a, b) {
  let d = a - b;
  while (d > Math.PI) d -= 2 * Math.PI;
  while (d < -Math.PI) d += 2 * Math.PI;
  return d;
}

/* ---------------- island-mode local evolution ----------------
 * Runs entirely inside one worker (or test harness): evolves the local
 * subpopulation for `steps` generations with zero IPC, then reports the
 * per-generation rows + batch champion. Called via GATrainer.startIslandRun
 * on the main thread; every worker runs its own copy in lockstep, so the
 * global generation counter advances `steps` per batch at near full CPU. */
function runIslandLocally(env, popIn, cfg, opts, startGen, steps, seedBase, w, nP) {
  let pop = popIn.map((p) => ({ genome: p.genome, fitness: p.fitness || 0 }));
  const n = pop.length;
  const rows = [];
  let champ = null, champScore = -1e9;
  const tSel = (arr) => {
    let best = null;
    for (let i = 0; i < cfg.tournament; i++) {
      const c = arr[Math.floor(Math.random() * arr.length)];
      if (!best || c.fitness > best.fitness) best = c;
    }
    return best;
  };
  for (let s = 0; s < steps; s++) {
    const gen = startGen + s;
    const sb = seedBase ^ ((gen >> 3) * 7919);
    for (let j = 0; j < n; j++) {
      env.setSeed(sb ^ ((w + j * nP) * 104729));
      const r = runEpisode(env, pop[j].genome, opts);
      pop[j].fitness = r.fitness; pop[j].found = r.found;
      pop[j].avgTTF = r.avgTTF; pop[j].circles = r.circles || 0;
    }
    pop.sort((a, b) => b.fitness - a.fitness);
    if (pop[0].fitness > champScore) { champScore = pop[0].fitness; champ = pop[0].genome; }
    let sumF = 0, sumFnd = 0, sumC = 0, nSucc = 0;
    for (const p of pop) {
      sumF += p.fitness; sumFnd += p.found || 0; sumC += p.circles || 0;
      if ((p.found || 0) > 0) nSucc++;
    }
    rows.push({
      gen, best: +pop[0].fitness.toFixed(2), avg: +(sumF / n).toFixed(2),
      bestFound: pop[0].found, avgFound: +(sumFnd / n).toFixed(2),
      bestCirc: pop[0].circles || 0, avgCirc: +(sumC / n).toFixed(2),
      succ: +(100 * nSucc / n).toFixed(1),
    });
    const next = [];
    for (let i = 0; i < Math.min(cfg.elite, n); i++)
      next.push({ genome: cloneGenome(pop[i].genome), fitness: 0 });
    while (next.length < n) {
      const p1 = tSel(pop), p2 = tSel(pop);
      let g = crossover(p1.genome, p2.genome);
      g = mutate(g, cfg.mutRate, cfg.mutAmt);
      next.push({ genome: g, fitness: 0 });
    }
    pop = next;
  }
  return { rows, champion: champ, championScore: champScore, gen: startGen + steps };
}

/* ---------------- GA trainer (budget-aware, resumable) ---------------- */
class GATrainer {
  constructor(env, opts = {}) {
    this.env = env;
    this.popSize = opts.popSize || 24;
    this.elite = opts.elite || 3;
    this.mutRate = opts.mutRate || 0.3;
    this.mutAmt = opts.mutAmt || 0.35;
    this.tournament = opts.tournament || 3;
    this.episodeT = opts.episodeT || 15;
    this.nFood = opts.nFood || 3;
    this.lr = opts.lr || 0.018;
    this.gen = 0;
    this.evalIdx = 0;                 // next population member to evaluate
    this.busy = false;                // an async (parallel) evaluation is in flight
    this.evalToken = 0;               // invalidates in-flight results on reset
    this.pool = null;                 // optional worker pool (see attachPool)
    this.pop = [];
    this._seedPop();
    this.history = [];                // {gen, best, avg, bestFound, avgFound}
    this.bestGenome = null;
    this.bestScore = -1e9;
  }

  /* a third of the population starts as mutants of the pre-evolved default
   * genome so training improves fast instead of rediscovering chemotaxis */
  _seedPop() {
    this.pop = [];
    const nSeed = Math.floor(this.popSize / 3);
    for (let i = 0; i < nSeed; i++)
      this.pop.push({ genome: mutate(DEFAULT_GENOME, 0.6, 0.25), fitness: 0 });
    while (this.pop.length < this.popSize)
      this.pop.push({ genome: randomGenome(), fitness: 0 });
  }

  /* Evaluate population members until budget exhausted or all done.
   * Returns true when the whole population is evaluated. */
  evaluateSlice(budgetMs = 10) {
    const t0 = performance.now();
    while (this.evalIdx < this.popSize) {
      if (performance.now() - t0 > budgetMs) return false;
      const ind = this.pop[this.evalIdx];
      const world = (this.gen >> 3) * 7919;   // same world for 8 generations
      this.env.setSeed(0xBADA55 ^ world ^ (this.evalIdx * 104729));
      const r = runEpisode(this.env, ind.genome, {
        episodeT: this.episodeT, nFood: this.nFood, lr: this.lr,
      });
      ind.fitness = r.fitness; ind.found = r.found; ind.avgTTF = r.avgTTF;
      ind.circles = r.circles || 0;
      this.evalIdx++;
    }
    return true;
  }

  /* ---- parallel evaluation over a worker pool (browser) ---- */
  attachPool(pool) { this.pool = pool; }

  /* Fully pipelined dispatch: each worker holds a BATCH of population
   * members; the instant a worker returns, the onmessage handler feeds it
   * the next batch. No setTimeout, no 4 ms timer clamp, no all-worker
   * barrier — every core stays busy end-to-end, and the next generation
   * dispatches directly from the final result handler (hundreds of
   * generations/sec). Deterministic: per-member seeds match the sync path.
   * In-flight results are dropped if gen/token changed. */
  evaluateParallel(onGenDone) {
    if (this.busy || !this.pool || !this.pool.ready) return false;
    const P = this.pool.workers.length;
    this._batch = Math.max(1, Math.ceil(this.popSize / P / 2));
    const world = (this.gen >> 3) * 7919;      // same world for 8 generations
    const seedBase = 0xBADA55 ^ world;
    const opts = { episodeT: this.episodeT, nFood: this.nFood, lr: this.lr };
    const tk = ++this.evalToken;
    const gen = this.gen;
    this.busy = true;
    this.evalIdx = 0;
    this._nextJob = 0;
    this._doneJobs = 0;
    this._lastResultT = performance.now();
    const self = this;

    const dispatchNext = (w) => {
      if (self._nextJob >= self.popSize) return;
      const jobs = [];
      for (let n = 0; n < self._batch && self._nextJob < self.popSize; n++) {
        const i = self._nextJob++;
        jobs.push({ i, genome: self.pop[i].genome });
      }
      self.pool.workers[w].postMessage({
        cmd: "evalOne", gen, tk, seedBase, opts, w, jobs,
      });
    };

    this.pool.onResults = (res) => {
      if (res.tk !== tk || res.gen !== gen) return;   // stale — drop
      self._lastResultT = performance.now();
      for (const r of res.results) {
        const ind = self.pop[r.i];
        if (ind) {
          ind.fitness = r.fitness; ind.found = r.found;
          ind.avgTTF = r.avgTTF; ind.circles = r.circles || 0;
          self.evalIdx++;
        }
        self._doneJobs++;
      }
      if (self._doneJobs < self.popSize) {
        dispatchNext(res.w);          // refill this worker immediately
      } else {
        self.busy = false;            // clear BEFORE the callback so the
        onGenDone();                  // next generation can chain instantly
      }
    };

    for (let w = 0; w < P; w++) dispatchNext(w);
    return true;
  }

  /* Selection + mutation → next generation. Returns stats row. */
  evolve() {
    this.pop.sort((a, b) => b.fitness - a.fitness);
    const best = this.pop[0];
    if (best.fitness > this.bestScore) {
      this.bestGenome = cloneGenome(best.genome);
      this.bestScore = best.fitness;
    }
    const avg = this.pop.reduce((a, b) => a + b.fitness, 0) / this.popSize;
    const avgFound = this.pop.reduce((a, b) => a + (b.found || 0), 0) / this.popSize;
    const avgCirc = this.pop.reduce((a, b) => a + (b.circles || 0), 0) / this.popSize;
    const nSucc = this.pop.reduce((a, b) => a + ((b.found || 0) > 0 ? 1 : 0), 0);
    const row = {
      gen: this.gen, best: +best.fitness.toFixed(2), avg: +avg.toFixed(2),
      bestFound: best.found, avgFound: +avgFound.toFixed(2),
      bestCirc: best.circles || 0, avgCirc: +avgCirc.toFixed(2),
      succ: +(100 * nSucc / this.popSize).toFixed(1),
      bestGenome: cloneGenome(best.genome),
    };
    this.history.push(row);

    const rng = Math.random;
    const next = [];
    for (let i = 0; i < this.elite; i++) next.push({ genome: cloneGenome(this.pop[i].genome), fitness: 0 });
    while (next.length < this.popSize) {
      const p1 = this._tournament(), p2 = this._tournament();
      let g = crossover(p1.genome, p2.genome, rng);
      g = mutate(g, this.mutRate, this.mutAmt, rng);
      next.push({ genome: g, fitness: 0 });
    }
    this.pop = next;
    this.gen++;
    this.evalIdx = 0;
    return row;
  }

  _tournament() {
    let best = null;
    for (let i = 0; i < this.tournament; i++) {
      const c = this.pop[Math.floor(Math.random() * this.popSize)];
      if (!best || c.fitness > best.fitness) best = c;
    }
    return best;
  }

  reset() {
    this.gen = 0; this.evalIdx = 0; this.history = [];
    this.bestGenome = null; this.bestScore = -1e9;
    this.evalToken++;                 // drop any in-flight worker results
    this.busy = false;
    this._seedPop();
  }

  /* ---- island mode: each worker evolves its own subpopulation locally for
   * `steps` generations (zero IPC during the batch), then reports history
   * rows + its champion and receives new migrants. This amortizes message
   * overhead ~batch-size×, which is what pushes past 200 generations/sec.
   * World seed advances with absolute generation so islands stay comparable. */
  startIslandRun(steps, onRow, onIdle) {
    if (this.busy || !this.pool || !this.pool.ready) return false;
    const P = this.pool.workers.length;
    const tk = ++this.evalToken;
    const startGen = this.gen;
    const world = (startGen >> 3) * 7919;
    const seedBase = 0xBADA55 ^ world;
    const opts = { episodeT: this.episodeT, nFood: this.nFood, lr: this.lr };
    this.busy = true;
    this._islandMode = true;
    const self = this;
    let done = 0;

    // split the global population into per-island chunks (spread elites)
    const islands = [];
    for (let w = 0; w < P; w++) islands.push([]);
    for (let i = 0; i < this.popSize; i++) {
      islands[i % P].push({ genome: this.pop[i].genome, fitness: this.pop[i].fitness || 0 });
    }

    this._islandChampions = [];
    const batchChamps = [];          // w -> champion genome
    const batchRows = new Map();     // gen -> [{row, w}]
    this.pool.onResults = (res) => {
      if (res.tk !== tk) return;
      for (const row of res.rows) {
        if (!batchRows.has(row.gen)) batchRows.set(row.gen, []);
        batchRows.get(row.gen).push({ row, w: res.w });
      }
      batchChamps[res.w] = res.champion ? cloneGenome(res.champion) : null;
      if (res.champion) {
        self._islandChampions.push(cloneGenome(res.champion));
        if (self.bestGenome === null || res.championScore > self.bestScore) {
          self.bestGenome = cloneGenome(res.champion);
          self.bestScore = res.championScore;
        }
      }
      if (++done === P) {
        self.busy = false;
        self._islandMode = false;
        self.gen = Math.max(self.gen, res.gen);   // batch complete: advance
        // merge the islands' per-generation rows into ONE global row per
        // generation (best across islands, means for the rest) so history
        // length always equals the generation count
        const gensSorted = [...batchRows.keys()].sort((a, b) => a - b);
        for (const g of gensSorted) {
          const rs = batchRows.get(g);
          let bi = 0;
          for (let k = 1; k < rs.length; k++) if (rs[k].row.best > rs[bi].row.best) bi = k;
          const m = rs[bi].row;
          const mean = (f) => rs.reduce((s, r) => s + (r.row[f] || 0), 0) / rs.length;
          self.history.push({
            gen: m.gen, best: m.best, avg: +mean("avg").toFixed(2),
            bestFound: m.bestFound, avgFound: +mean("avgFound").toFixed(2),
            bestCirc: m.bestCirc, avgCirc: +mean("avgCirc").toFixed(2),
            succ: +mean("succ").toFixed(1),
            bestGenome: batchChamps[rs[bi].w] || null,
          });
        }
        // cross-pollinate: rebuild the global pop from island champions +
        // mutants, then re-split — keeps islands from drifting apart
        if (self._islandChampions.length) {
          const champs = self._islandChampions;
          const next = [];
          for (let i = 0; i < self.popSize; i++) {
            const src = champs[i % champs.length];
            next.push({
              genome: i < champs.length ? cloneGenome(src)
                                        : mutate(src, self.mutRate, self.mutAmt),
              fitness: 0,
            });
          }
          self.pop = next;
        }
        self._islandChampions = [];
        onIdle();
      }
    };

    for (let w = 0; w < P; w++) {
      this.pool.workers[w].postMessage({
        cmd: "islandRun", tk, startGen, steps, seedBase, opts, w, nP: P,
        pop: islands[w],
        cfg: { elite: this.elite, mutRate: this.mutRate, mutAmt: this.mutAmt, tournament: this.tournament },
      });
    }
    return true;
  }
}

/* ---------------- worker pool (browser only) ----------------
 * Each worker runs the same brain.js + ga.js and evaluates a strided chunk
 * of the population per generation. Pool throughput ≈ nWorkers × single-
 * core speed, which is what pushes training past 200 generations/sec. */
function createGAPool(circ, envSpec, n, onReady) {
  if (typeof Worker === "undefined") return null;
  const workers = [];
  let readyCount = 0;
  const pool = { workers, n, ready: false, onResults: null,
    dispose() { for (const w of workers) w.terminate(); } };
  for (let i = 0; i < n; i++) {
    const w = new Worker("js/ga-worker.js");
    w.onmessage = (e) => {
      const m = e.data;
      if (m.cmd === "ready") {
        if (++readyCount === n) { pool.ready = true; if (onReady) onReady(); }
      } else if (m.cmd === "results" && pool.onResults) {
        pool.onResults(m);
      }
    };
    w.onerror = (e) => console.error("GA worker error:", e.message || e);
    w.postMessage({ cmd: "init", circ, envSpec });
    workers.push(w);
  }
  return pool;
}

if (typeof window !== "undefined") {
  window.GA = {
    GENE_KEYS, GENE_DEF, DEFAULT_GENOME,
    randomGenome, mutate, crossover, cloneGenome,
    senseAt, sense, policyStep, runEpisode, GATrainer, createGAPool,
    runIslandLocally,
    CIRC, initCirc, updateCircling, resetCirc,
  };
}
