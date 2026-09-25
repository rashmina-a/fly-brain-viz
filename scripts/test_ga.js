/* Headless GA smoke test (node): evolve a few generations on the real circuit
 * and print per-generation stats. Run:  node scripts/test_ga.js [gens]
 * Loads only brain.js + ga.js — no DOM needed. */
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

const env = {
  W: 1400, H: 900, flySpeed: 105, turnRate: 3.6, eatDist: 16,
  odorNames: circ.odors.map(o => o.name),
  odorGlom: {},
  rnd: Math.random,
  makeBrain: () => {
    if (!env._brain) env._brain = new FlyBrain(circ);
    env._brain.resetPlasticity();
    return env._brain;
  },
  setSeed(s) { this._rng = mulberry32(s); this.rnd = this._rng; },
};
for (const o of circ.odors) env.odorGlom[o.name] = o.glom;

const gens = parseInt(process.argv[2] || "6", 10);
const t = new GATrainer(env, { popSize: 16, elite: 3, episodeT: 15, nFood: 3, lr: 0.018 });

// naive baseline for reference
env.setSeed(12345);
const base = runEpisode(env, DEFAULT_GENOME, { episodeT: 15, nFood: 3, lr: 0.018 });
console.log("naive default genome:", JSON.stringify(base));

const t0 = Date.now();
for (let g = 0; g < gens; g++) {
  const gt = Date.now();
  const done = t.evaluateSlice(Infinity);
  if (!done) { console.error("evaluateSlice did not finish!"); process.exit(1); }
  const row = t.evolve();
  console.log(
    "gen " + String(row.gen).padStart(4) +
    "  best " + String(row.best).padStart(7) +
    "  avg " + String(row.avg).padStart(7) +
    "  found(best/avg) " + row.bestFound + "/" + row.avgFound.toFixed(1) +
    "  succ " + row.succ.toFixed(0) + "%" +
    "  circ " + row.bestCirc + "/" + row.avgCirc.toFixed(1) +
    "  (" + (Date.now() - gt) + " ms)");
}
console.log("total: " + (Date.now() - t0) + " ms for " + gens + " generations = " +
  (gens / ((Date.now() - t0) / 1000)).toFixed(0) + " gens/sec (node, single core)");
console.log("best genome:", JSON.stringify(
  Object.fromEntries(Object.entries(t.bestGenome).map(([k, v]) => [k, +v.toFixed(3)]))));
console.log("OK");
`;

eval(harness);
