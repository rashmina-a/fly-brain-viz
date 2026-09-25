/* Offline: evolve a strong DEFAULT_GENOME and validate on fresh worlds.
 * Run:  node scripts/evolve_default.js [gens]
 * Writes scripts/default_genome.json and prints stats. */
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

const gens = parseInt(process.argv[2] || "40", 10);
const t = new GATrainer(env, { popSize: 24, elite: 3, episodeT: 45, nFood: 3, lr: 0.018 });
for (let g = 0; g < gens; g++) {
  t.evaluateSlice(Infinity);
  const row = t.evolve();
  if (g % 5 === 0 || g === gens - 1)
    console.log("gen " + row.gen + "  best " + row.best + "  avg " + row.avg + "  found " + row.bestFound + "/" + row.avgFound.toFixed(1));
}

// validate best vs hand default on 12 FRESH worlds (seeds never used in training)
function evalGenome(genome, label) {
  let f = 0, found = 0;
  for (let k = 0; k < 12; k++) {
    env.setSeed(0xFEED + k * 7919);
    const r = runEpisode(env, genome, { episodeT: 45, nFood: 3, lr: 0.018 });
    f += r.fitness; found += r.found;
  }
  console.log(label + ": avg fitness " + (f / 12).toFixed(1) + "  avg found " + (found / 12).toFixed(1));
}
evalGenome(DEFAULT_GENOME, "hand default ");
evalGenome(t.bestGenome, "evolved best ");

fs.writeFileSync(process.argv[1].replace(/evolve_default\\.js$/, "default_genome.json"),
  JSON.stringify({ genome: t.bestGenome, score: t.bestScore, gens }, null, 2));
console.log("saved default_genome.json");
`;

eval(harness);
