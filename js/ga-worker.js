/* GA worker — headless episode evaluation for the Fly Foraging Arena.
 * Runs the SAME brain.js + ga.js as the main thread; the trainer pipelines
 * small BATCHES of population members per worker (the trainer immediately
 * feeds the next batch when a result arrives), so all cores stay busy
 * end-to-end. Deterministic: per-member seeds match the sync path exactly. */
"use strict";

importScripts("brain.js", "ga.js");

let circ = null, env = null;

self.onmessage = (e) => {
  const m = e.data;
  if (m.cmd === "init") {
    circ = m.circ;
    prepCircuit(circ);
    env = {
      W: m.envSpec.W, H: m.envSpec.H,
      flySpeed: m.envSpec.flySpeed, turnRate: m.envSpec.turnRate,
      eatDist: m.envSpec.eatDist,
      odorNames: circ.odors.map((o) => o.name),
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
    postMessage({ cmd: "ready" });
  } else if (m.cmd === "evalOne") {
    const results = [];
    for (const job of m.jobs) {
      env.setSeed(m.seedBase ^ (job.i * 104729));
      const r = runEpisode(env, job.genome, m.opts);
      results.push({ i: job.i, fitness: r.fitness, found: r.found, avgTTF: r.avgTTF, circles: r.circles });
    }
    postMessage({ cmd: "results", gen: m.gen, tk: m.tk, w: m.w, results });
  } else if (m.cmd === "islandRun") {
    const out = runIslandLocally(env, m.pop, m.cfg, m.opts, m.startGen, m.steps, m.seedBase, m.w, m.nP);
    postMessage({ cmd: "results", tk: m.tk, w: m.w, ...out });
  }
};
