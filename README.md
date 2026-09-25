# 🪰 Fly Brain Explorer — MaleCNS v1.0

An interactive web interface for the **connectome of the adult male fruit fly
(*Drosophila melanogaster*) CNS** — cloned from the open-source dataset
published by HHMI Janelia ([MaleCNS v1.0](http://male-cns.janelia.org),
CC-BY-4.0). Every neuron and every chemical synapse is real data.

Two apps:

1. **3D brain explorer** (`/`) — 200k neurons in 3D, neurotransmitter colors,
   stimulus firing simulation, per-neuron skeletons and real connectivity.
2. **2D foraging arena** (`/arena.html`) — a fly agent runs the *real*
   olfactory circuit (ORN → PN → KC → MBON + DAN learning) inside a 2D world.
   A **genetic algorithm** evolves its sensorimotor policy at **hundreds of
   generations per second**; you train first, then experiment with the
   trained fly.

## Quick start

```bash
cd fly-brain-viz
python server.py            # default port 8000
```

Open `http://127.0.0.1:8000/` (3D explorer) or
`http://127.0.0.1:8000/arena.html` (foraging arena).

**First run only:** the server automatically downloads the raw connectome
dataset (~1.1 GB, public GCS bucket, resumable) and builds the web-optimized
files with `scripts/prepare_data.py` + `scripts/prepare_olfactory.py`
(needs `pip install pandas pyarrow numpy`). Later runs start instantly.
Use `python server.py --no-download` to serve without ever downloading.

## The foraging arena (train → experiment)

- **🧬 Train (GA)** — episodes run headless across a worker pool using an
  **island model** (each core evolves a subpopulation locally for 10
  generations, then champions migrate). Food is placed automatically;
  worlds are seeded so selection is fair. Typical throughput: **300–400
  generations/sec** — 2000 generations in ~5–6 s. The **success-rate chart**
  shows the share of flies that find food each generation; best/avg score
  chart sits above it. The best genome autosaves to the browser.
- **🔬 Experiment** — press *use best genome in experiment* and the trained
  fly forages live while its mushroom body keeps learning (DAN-gated
  KC→MBON plasticity on real synapses). Click to place food, watch the
  brain monitor (real PN/KC/MBON/DAN names) fire.
- **Circling punishment** — orbiting food instead of landing costs GA
  fitness *and* fires punishment DANs that devalue the currently sniffed
  odor (PPL1-style), so the mushroom body learns "orbiting gets me nowhere".

### Speed & performance notes

- Training is deliberately throttled in the UI (demo sim frozen, panels
  refreshed at 2.5 Hz) so nearly all CPU goes to the workers.
- The real circuit sniff is cached per odor (KC sparseness makes valence
  shift only for the rewarded odor), so episodes cost ~2 sniffs instead of ~15.
- Genuinely offline knobs: `ISLAND_STEPS` (batch size), `popSize`/`episodeT`
  (in `arena.js`), mutation settings (in `GATrainer`).

## Files

```
index.html, js/app.js        3D brain explorer (three.js, vendored r147)
arena.html, js/arena.js      2D foraging arena (Sim, UI, training pump)
js/brain.js                  REAL olfactory circuit (ORN→PN→KC→MBON+DAN)
js/ga.js                     genomes, policy, episodes, GATrainer, islands
js/ga-worker.js              worker entry (same code, per-core)
server.py                    static server + skeleton proxy + FIRST-RUN
                             dataset auto-download (resumable) + auto-derive
scripts/prepare_data.py      1 GB connectome → compact web binaries
scripts/prepare_olfactory.py extracts ORN→PN→KC→MBON/DAN circuit JSON
scripts/evolve_default.js    offline bake of the shipped default genome
scripts/test_ga.js           headless GA smoke test (node)
scripts/test_ga_pipeline.js  pipelined + island trainer test (node)
data/                        downloaded + generated (see .gitignore)
```

## Credits & license

- Connectome data: **MaleCNS v1.0**, HHMI Janelia —
  `https://storage.googleapis.com/flyem-male-cns/` (public bucket),
  licensed **CC-BY-4.0**. Citation: see http://male-cns.janelia.org.
- Skeletons proxied from the same bucket's neuroglancer precomputed volume.
- This app is an independent visualization/teaching tool, not affiliated
  with Janelia.
