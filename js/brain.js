/* FlyBrain — runs the REAL MaleCNS v1.0 olfactory circuit (HHMI Janelia, CC-BY-4.0)
 *
 *   ORN (51 glomeruli, odor-driven)  = concentration per glomerulus
 *   -> PN  (272 real PNs, real ORN->PN synapse weights, lateral inhibition)
 *   -> KC  (4064 real Kenyon cells, real PN->KC weights, sparse code)
 *   -> MBON (97 real MBONs, real KC->MBON weights; innate valence from NT)
 *   DAN  (337 real DANs) fire on reward and gate plasticity at KC->MBON
 *   synapses (dopamine = teaching signal, like the real mushroom body).
 *
 * Learning: KC->MBON weights decay toward baseline when their MBON's DANs fire
 * (reward), i.e. approach + food shifts the KC->MBON readout — the classic
 * mushroom-body learning rule, on the real connectome's synapses.
 */
"use strict";

/* ---------------- tiny math ---------------- */
function zeros(n) { return new Float32Array(n); }
function norm2(x, y) { return Math.hypot(x, y); }
function clamp(v, a, b) { return v < a ? a : (v > b ? b : v); }

/* sparse edge storage: [src, dst, w] triples -> per-src adjacency lists */
function indexEdges(edges, nSrc, nDst) {
  const off = new Int32Array(nSrc + 1);
  const cnt = new Int32Array(nSrc);
  for (const e of edges) cnt[e[0]]++;
  let acc = 0;
  for (let i = 0; i < nSrc; i++) { off[i] = acc; acc += cnt[i]; }
  off[nSrc] = acc;
  const dst = new Int32Array(acc);
  const w = new Float32Array(acc);
  const cur = Int32Array.from(off.subarray(0, nSrc));
  for (const [s, d, wt] of edges) {
    const k = cur[s]++;
    dst[k] = d; w[k] = wt;
  }
  return { off, dst, w, nDst };
}

function spread(index, input, out, gain) {
  out.fill(0);
  const { off, dst, w } = index;
  for (let s = 0; s < input.length; s++) {
    const v = input[s];
    if (v <= 0) continue;
    for (let k = off[s]; k < off[s + 1]; k++) out[dst[k]] += v * w[k];
  }
  if (gain !== undefined) {
    for (let i = 0; i < out.length; i++) out[i] *= gain;
  }
  return out;
}

/* like spread() but with an external (plastic) weight array */
function spreadW(index, W, input, out) {
  out.fill(0);
  const { off, dst } = index;
  for (let s = 0; s < input.length; s++) {
    const v = input[s];
    if (v <= 0) continue;
    for (let k = off[s]; k < off[s + 1]; k++) out[dst[k]] += v * W[k];
  }
  return out;
}

/* ---------------- FlyBrain ---------------- */
class FlyBrain {
  constructor(circ) {
    this.c = circ;
    const nG = circ.glomeruli.length;
    const nPN = circ.pn.length, nKC = circ.kc.length;
    const nMBON = circ.mbon.length, nDAN = circ.dan.length;
    this.n = { g: nG, pn: nPN, kc: nKC, mbon: nMBON, dan: nDAN };

    this.orn = zeros(nG);
    this.pn = zeros(nPN);
    this.kc = zeros(nKC);
    this.mbon = zeros(nMBON);
    this.mbonBaseline = zeros(nMBON);
    this.dan = zeros(nDAN);
    this.gate = zeros(nMBON);        // per-compartment plasticity gate
    this.val = 0;                     // instantaneous valence (approach drive)

    // real connectivity
    this.ornPn = indexEdges(circ.ornPn, nG, nPN);
    this.pnKc = indexEdges(circ.pnKc, nPN, nKC);
    this.kcMbon = indexEdges(circ.kcMbon, nKC, nMBON);
    this.danMbon = indexEdges(circ.danMbon, nDAN, nMBON);

    // plastic KC->MBON efficacy (starts at real synapse weight)
    this.kcMbonW = Float32Array.from(this.kcMbon.w);
    this.wBase = Float32Array.from(this.kcMbon.w);

    // per-MBON innervating DAN lists (for gating which MBONs learn)
    this.dansOfMbon = Array.from({ length: nMBON }, () => []);
    for (const [d, m] of circ.danMbon) this.dansOfMbon[m].push(d);

    // inverse KC->MBON index: synapses grouped by target MBON. Gated
    // plasticity (learn) then touches only the ~2k synapses feeding the few
    // gated compartments instead of scanning all KC axons every call.
    {
      const off = this.kcMbon.off, dst = this.kcMbon.dst;
      const nSyn = off[nKC];
      const invOff = new Int32Array(nMBON + 1);
      for (let s = 0; s < nKC; s++)
        for (let k = off[s]; k < off[s + 1]; k++) invOff[dst[k] + 1]++;
      for (let m = 0; m < nMBON; m++) invOff[m + 1] += invOff[m];
      const invKc = new Int32Array(nSyn), invSyn = new Int32Array(nSyn);
      const cur = Int32Array.from(invOff.subarray(0, nMBON));
      for (let s = 0; s < nKC; s++)
        for (let k = off[s]; k < off[s + 1]; k++) {
          const m = dst[k], j = cur[m]++;
          invKc[j] = s; invSyn[j] = k;
        }
      this.invOff = invOff; this.invKc = invKc; this.invSyn = invSyn;
    }
    this._epoch = 0;                  // bumped whenever plastic weights change

    // persistent valence cache (per odor name). KC sparseness means an
    // eaten odor's learned valence shift barely moves OTHER odors', so only
    // the eaten odor is invalidated — the rest survive plasticity AND
    // plasticity resets. This is what makes headless GA evaluation ~4×
    // cheaper: ~2 circuit sniffs per episode instead of ~15.
    this._valCache = new Map();

    // baseline MBON activity from a reference odor mixture (homeostasis)
    this._computeBaseline();

    // preallocated scratch buffers (sniff runs thousands of times per second
    // during GA training — allocation-free hot path matters)
    this._bufPn = zeros(nPN);
    this._bufKc = zeros(nKC);
    this._bufMb = zeros(nMBON);

    this.meta = { nG, nPN, nKC, nMBON, nDAN };
  }

  /* back to naive weights, empty gate/DAN (fresh subject for a new episode) */
  resetPlasticity() {
    this.kcMbonW.set(this.wBase);
    this.dan.fill(0);
    this.gate.fill(0);
    this.val = 0;
    this.driveP = 0; this.driveN = 0;
    this._lastMBON = null;
    this._epoch++;                    // invalidate any external sniff caches
  }

  _computeBaseline() {
    // typical ambient mixture = mild random template
    const nG = this.n.g;
    const ref = zeros(nG);
    for (let i = 0; i < nG; i++) ref[i] = 0.18;
    const pn = zeros(this.n.pn), kc = zeros(this.n.kc), mb = zeros(this.n.mbon);
    spread(this.ornPn, ref, pn);
    this._pnStep(pn);
    spread(this.pnKc, this.pn, kc, 1);
    this._kcStep(kc);
    spreadW(this.kcMbon, this.kcMbonW, this.kc, mb);
    this.mbonBaseline.set(mb);
  }

  _pnStep(drive) {
    // PN response = excitatory ORN drive + global inhibition (olfactory AL style)
    const pn = this.pn;
    let sum = 0;
    for (let i = 0; i < drive.length; i++) sum += drive[i];
    const inh = this._inhGain !== undefined ? this._inhGain : 0.35;
    for (let i = 0; i < pn.length; i++) {
      // PN's glomerular drive: use drive through ornPn weights (already spread)
      pn[i] = drive[i] / (1 + inh * sum / (this.n.pn / 40));
      if (pn[i] < 0) pn[i] = 0;
    }
  }

  _kcStep(kcDrive) {
    // KCs: high threshold -> sparse code, like the real MB calyx
    const kc = this.kc;
    let max = 0;
    for (let i = 0; i < kcDrive.length; i++) if (kcDrive[i] > max) max = kcDrive[i];
    const thr = max * 0.4;
    for (let i = 0; i < kc.length; i++) kc[i] = kcDrive[i] > thr ? (kcDrive[i] - thr) / (max - thr + 1e-9) : 0;
  }

  /* cached valence lookup: full circuit sniff only on cache miss */
  sniffCached(odorName, odorTemplate, conc) {
    let v = this._valCache.get(odorName);
    if (v === undefined) {
      v = this.sniff(odorTemplate, conc);
      this._valCache.set(odorName, v);
    } else {
      this.val = v;
    }
    return v;
  }
  invalidateOdor(odorName) { this._valCache.delete(odorName); }
  invalidateAllValence() { this._valCache.clear(); }

  /* odor: template dict glom->amp; conc: 0..1 local concentration */
  sniff(odorTemplate, conc) {
    const orn = this.orn;
    orn.fill(0);
    if (conc <= 0) return this.val;
    for (const g in odorTemplate) {
      const gi = this.c.glomIndex[g];
      if (gi !== undefined) orn[gi] = odorTemplate[g] * conc;
    }
    const drive = this._bufPn;
    spread(this.ornPn, orn, drive);
    this._pnStep(drive);
    const kcDrive = this._bufKc;
    spread(this.pnKc, this.pn, kcDrive, 1);
    this._kcStep(kcDrive);
    const mb = this._bufMb;
    spreadW(this.kcMbon, this.kcMbonW, this.kc, mb);
    // valence = approach vs avoidance drive from valence-carrying MBONs
    // (raw KC->MBON drive; learned depression shifts the balance directly)
    let p = 0, n = 0;
    for (let i = 0; i < this.n.mbon; i++) {
      const v = this.c.mbon[i].v;
      if (v === 1) p += mb[i];
      else if (v === -1) n += mb[i];
    }
    this.driveP = p; this.driveN = n;
    this.val = (p - n) / (p + n + 1);
    this._lastMBON = mb;
    return this.val;
  }

  /* Reward (rate>0): DANs fire onto AVOIDANCE compartments of currently-
   * active MBONs (appetitive teaching, like real PAM DANs). Punishment
   * (rate<0): DANs onto APPROACH compartments (like PPL1 DANs). The gate
   * makes plasticity compartment-specific even though real DANs ramify
   * across several compartments. */
  reward(rate = 1.0) {
    const mb = this._lastMBON;
    if (!mb) return 0;
    const want = rate > 0 ? -1 : 1;   // MBON sign to gate
    let touched = 0;
    for (let m = 0; m < this.n.mbon; m++) {
      if (mb[m] <= 0.01) continue;
      if (this.c.mbon[m].v !== want) continue;
      this.gate[m] = Math.max(this.gate[m], Math.abs(rate));
      for (const d of this.dansOfMbon[m]) {
        this.dan[d] = Math.min(1, this.dan[d] + Math.abs(rate) * 0.6);
      }
      touched++;
    }
    return touched;
  }

  /* Mushroom-body plasticity: in gated compartments, active KC->MBON
   * synapses weaken (dopaminergic depression). KC sparseness makes the
   * change specific to the rewarded odor's ensemble. Uses the inverse
   * synapse index: cost ~ gated synapses, not the whole calyx. */
  learn(lrate = 0.02) {
    const mb = this._lastMBON;
    if (!mb) return 0;
    let changed = 0;
    const kc = this.kc, gate = this.gate, W = this.kcMbonW;
    for (let m = 0; m < this.n.mbon; m++) {
      const gt = gate[m];
      if (gt <= 0.01) continue;
      for (let j = this.invOff[m]; j < this.invOff[m + 1]; j++) {
        const s = this.invKc[j];
        if (kc[s] <= 0.01) continue;
        const k = this.invSyn[j];
        W[k] = Math.max(0, W[k] - lrate * gt * kc[s] * W[k]);
        changed++;
      }
    }
    if (changed) this._epoch++;       // invalidate external sniff caches
    return changed;
  }

  danDecay(rate = 0.9) {
    for (let i = 0; i < this.dan.length; i++) this.dan[i] *= rate;
    for (let i = 0; i < this.gate.length; i++) this.gate[i] *= rate;
  }

  /* approach value used by the agent's steering */
  get valence() { return this.val; }

  /* names of the most active units (for the brain monitor) */
  topUnits(arr, k) {
    const idx = Array.from(arr.keys());
    idx.sort((a, b) => arr[b] - arr[a]);
    return idx.slice(0, k).map((i) => ({
      name: this._nameFor(arr, i), act: arr[i], i,
    }));
  }
  _nameFor(arr, i) {
    if (arr === this.orn) return this.c.glomeruli[i];
    if (arr === this.pn) return this.c.pn[i].name;
    if (arr === this.kc) return "KC#" + i;
    if (arr === this.mbon) return this.c.mbon[i].name;
    if (arr === this.dan) return this.c.dan[i].name;
    return String(i);
  }
}

/* build glomIndex helper from circuit json */
function prepCircuit(circ) {
  circ.glomIndex = {};
  circ.glomeruli.forEach((g, i) => (circ.glomIndex[g] = i));
  return circ;
}

if (typeof window !== "undefined") {
  window.FlyBrain = FlyBrain;
  window.prepCircuit = prepCircuit;
}
