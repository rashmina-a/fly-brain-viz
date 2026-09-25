/* Pipeline test: exercise the REAL GATrainer contracts with a fake async pool
 * (setImmediate "workers"), verifying (a) no lost jobs, (b) generation
 * chaining without timers, (c) island-mode batching, (d) generations/sec.
 * Run:  node scripts/test_ga_pipeline.js [gens] [workers] */
"use strict";
const fs = require("fs");
const path = require("path");

const root = path.join(__dirname, "..");
const brainSrc = fs.readFileSync(path.join(root, "js/brain.js"), "utf8");
const gaSrc = fs.readFileSync(path.join(root, "js/ga.js"), "utf8");
const circ = JSON.parse(fs.readFileSync(path.join(root, "data/olfactory_circuit.json"), "utf8"));

const harness = `
${brainSrc}
${gaSrc}

prepCircuit(circ);

/* fake pool: same message contract as js/ga-worker.js + createGAPool */
function makeFakePool(n, env) {
  const pool = { workers: [], ready: true, onResults: null, dispose() {} };
  for (let w = 0; w < n; w++) {
    pool.workers.push({ postMessage(msg) {
      setImmediate(() => {
        if (msg.cmd === "evalOne") {
          const results = msg.jobs.map((job) => {
            env.setSeed(msg.seedBase ^ (job.i * 104729));
            const ep = runEpisode(env, job.genome, msg.opts);
            return { i: job.i, fitness: ep.fitness, found: ep.found, avgTTF: ep.avgTTF, circles: ep.circles };
          });
          if (pool.onResults) pool.onResults({ cmd: "results", gen: msg.gen, tk: msg.tk, w, results });
        } else if (msg.cmd === "islandRun") {
          const out = runIslandLocally(env, msg.pop, msg.cfg, msg.opts, msg.startGen, msg.steps, msg.seedBase, msg.w, msg.nP);
          if (pool.onResults) pool.onResults({ cmd: "results", tk: msg.tk, w, ...out });
        }
      });
    }});
  }
  return pool;
}

const env = {
  W: 1400, H: 900, flySpeed: 105, turnRate: 3.6, eatDist: 16,
  odorNames: circ.odors.map(o => o.name), odorGlom: {}, rnd: Math.random,
  makeBrain: () => {
    if (!env._brain) env._brain = new FlyBrain(circ);
    env._brain.resetPlasticity();
    return env._brain;
  },
  setSeed(s) { this._rng = mulberry32(s); this.rnd = this._rng; },
};
for (const o of circ.odors) env.odorGlom[o.name] = o.glom;

const targetGens = parseInt(${JSON.stringify(String(process.argv[2] || "200"))}, 10);
const nWorkers = parseInt(${JSON.stringify(String(process.argv[3] || "4"))}, 10);

const pool = makeFakePool(nWorkers, env);

/* ---- phase 1: pipelined per-member evaluation ---- */
const t = new GATrainer(env, { popSize: 16, elite: 3, episodeT: 15, nFood: 3, lr: 0.018 });
t.attachPool(pool);
let gens = 0;
const t0 = Date.now();
function pump() {
  if (t.busy) return;
  if (!t.evaluateParallel(done)) {
    console.error("FAIL: evaluateParallel returned false while pool ready & idle");
    process.exit(1);
  }
}
function done() {
  t.evolve(); gens++;
  if (gens < targetGens) { pump(); return; }
  const el = (Date.now() - t0) / 1000;
  const last = t.history[t.history.length - 1];
  console.log("pipelined trainer:", gens, "gens in", el.toFixed(2) + "s =", (gens / el).toFixed(1), "gen/s (fake pool,", nWorkers, "workers, single process)");
  console.log("  last row:", JSON.stringify({ gen: last.gen, best: last.best, avg: last.avg, succ: last.succ, avgCirc: last.avgCirc }));
  if (t.history.length !== gens) { console.error("FAIL: history/gen mismatch"); process.exit(1); }
  islandPhase();
}

/* ---- phase 2: island-mode batched evaluation ---- */
function islandPhase() {
  const STEPS = 10, BATCHES = 5;
  const t2 = new GATrainer(env, { popSize: 8, elite: 2, episodeT: 10, nFood: 3, lr: 0.018 });
  t2.attachPool(pool);
  let batches = 0;
  const i0 = Date.now();
  function islandDone() {
    batches++;
    if (batches < BATCHES) { t2.startIslandRun(STEPS, null, islandDone); return; }
    const el2 = (Date.now() - i0) / 1000;
    const totalGens = t2.gen;
    const lastRow = t2.history[t2.history.length - 1];
    console.log("island trainer:", totalGens, "gens in", el2.toFixed(2) + "s =", (totalGens / el2).toFixed(1), "gen/s (single proc,", nWorkers, "islands)");
    console.log("  last row:", JSON.stringify({ gen: lastRow.gen, best: lastRow.best, succ: lastRow.succ }));
    console.log("  bestScore:", t2.bestScore.toFixed(1), "| history rows:", t2.history.length);
    if (t2.history.length !== totalGens) { console.error("FAIL: island history/gen mismatch"); process.exit(1); }
    if (!(t2.bestScore > 0)) { console.error("FAIL: island bestScore is 0"); process.exit(1); }
    console.log("ISLAND-OK");
    process.exit(0);
  }
  t2.startIslandRun(STEPS, null, islandDone);
}
pump();
`;
eval(harness);
