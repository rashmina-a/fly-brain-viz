#!/usr/bin/env python3
"""Extract the real olfactory pathway (ORN -> PN -> KC -> MBON, + DANs) from the
MaleCNS v1.0 connectome into a compact JSON for the 2D foraging arena.

Pathway (all real synaptic weights from the connectome):
  ORN (class=olfactory, type ORN_<glomerulus>)
  PN  (class=ALPN, <glomerulus>_lPN / _adPN / ...)
  KC  (class=Kenyon_Cell)
  MBON (class=MBON)   DAN (class=DAN)

Outputs data/olfactory_circuit.json (~ a few MB).
"""
import json
import os
import re

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
CW = os.path.join(DATA, "connectome-weights.feather")

# deterministic odor templates: each food = sparse activation over glomeruli
ODOR_SEED = 42
ODOR_NAMES = ["apple", "banana", "mango", "yeast", "mushroom", "vinegar"]
ODOR_GLoms = 10          # glomeruli per odor


def glom_from_orn_type(t: str) -> str:
    m = re.match(r"^ORN_([A-Za-z]+\d+[a-z]*)", t or "")
    return m.group(1) if m else ""


def glom_from_pn_type(t: str) -> str:
    # types like DA1_lPN, VA7m_adPN, VL2a_vPN ...
    m = re.match(r"^([A-Za-z]+\d+[a-z]*)_.*PN$", t or "")
    return m.group(1) if m else ""


def main():
    print("reading annotations ...")
    ann = pd.read_feather(os.path.join(DATA, "body-annotations.feather"))
    ann = ann[ann["status"] != "Glia"].copy()
    ann["type"] = ann["type"].fillna("")
    ann["instance"] = ann["instance"].fillna("")

    orn = ann[ann["class"] == "olfactory"].copy()
    orn["glom"] = orn["type"].map(glom_from_orn_type)
    orn = orn[orn["glom"] != ""]

    pn = ann[ann["class"] == "ALPN"].copy()
    pn["glom"] = pn["type"].map(glom_from_pn_type)
    pn = pn[pn["glom"] != ""]

    kc = ann[ann["class"] == "Kenyon_Cell"].copy()
    mbon = ann[ann["class"] == "MBON"].copy()
    dan = ann[ann["class"] == "DAN"].copy()

    orn_ids = set(orn["bodyId"]); pn_ids = set(pn["bodyId"])
    kc_ids = set(kc["bodyId"]); mbon_ids = set(mbon["bodyId"]); dan_ids = set(dan["bodyId"])
    print(f"ORN {len(orn_ids)} | PN {len(pn_ids)} | KC {len(kc_ids)} | "
          f"MBON {len(mbon_ids)} | DAN {len(dan_ids)}")

    print("scanning 152M edges for pathway subsets ...")
    cw = pd.read_feather(CW)
    pre = cw["body_pre"].to_numpy()
    post = cw["body_post"].to_numpy()
    w = cw["weight"].to_numpy()

    def slice_edges(pre_set, post_set):
        m = pd.Series(pre).isin(pre_set) & pd.Series(post).isin(post_set)
        return pre[m], post[m], w[m]

    # ORN -> PN (aggregate per glomerulus x PN)
    pre_p, post_p, w_p = slice_edges(orn_ids, pn_ids)
    glom_of_orn = dict(zip(orn["bodyId"], orn["glom"]))
    orn_gloms = pd.Series([glom_of_orn[b] for b in pre_p])
    orn_pn = pd.DataFrame({"glom": orn_gloms, "pn": post_p, "w": w_p})
    orn_pn = orn_pn.groupby(["glom", "pn"], as_index=False)["w"].sum()
    print("ORN->PN glom x PN rows:", len(orn_pn))

    # PN -> KC
    pre_k, post_k, w_k = slice_edges(pn_ids, kc_ids)
    pn_kc = pd.DataFrame({"pn": pre_k, "kc": post_k, "w": w_k})
    pn_kc = pn_kc.groupby(["pn", "kc"], as_index=False)["w"].sum()
    print("PN->KC rows:", len(pn_kc))

    # KC -> MBON
    pre_m, post_m, w_m = slice_edges(kc_ids, mbon_ids)
    kc_mbon = pd.DataFrame({"kc": pre_m, "mbon": post_m, "w": w_m})
    kc_mbon = kc_mbon.groupby(["kc", "mbon"], as_index=False)["w"].sum()
    print("KC->MBON rows:", len(kc_mbon))

    # DAN -> MBON (reward gating, real projections)
    pre_d, post_d, w_d = slice_edges(dan_ids, mbon_ids)
    dan_mbon = pd.DataFrame({"dan": pre_d, "mbon": post_d, "w": w_d})
    dan_mbon = dan_mbon.groupby(["dan", "mbon"], as_index=False)["w"].sum()
    print("DAN->MBON rows:", len(dan_mbon))

    # ---- compact index space -------------------------------------------
    # index over the UNION of neurons appearing anywhere in the pathway
    pn_all = set(orn_pn["pn"]) | set(pn_kc["pn"])
    kc_all = set(pn_kc["kc"]) | set(kc_mbon["kc"])
    mbon_all = set(kc_mbon["mbon"]) | set(dan_mbon["mbon"])
    dan_all = set(dan_mbon["dan"])

    gloms = sorted(orn_pn["glom"].unique())
    glom_ix = {g: i for i, g in enumerate(gloms)}

    pns = pn[pn["bodyId"].isin(pn_all)].copy()
    pns = pns.sort_values("bodyId")
    pn_ix = {b: i for i, b in enumerate(pns["bodyId"])}
    # each PN's glomerulus = its strongest real ORN input glomerulus
    prof = orn_pn.groupby(["pn", "glom"], as_index=False)["w"].sum()
    prof = prof.sort_values("w", ascending=False).drop_duplicates("pn")
    top_glom = dict(zip(prof["pn"], prof["glom"]))
    pn_list = [{"name": r["instance"] or r["type"],
                "glom": glom_ix[top_glom[r["bodyId"]]] if r["bodyId"] in top_glom else 0}
               for _, r in pns.iterrows()]

    kcs = kc[kc["bodyId"].isin(kc_all)].copy().sort_values("bodyId")
    kc_ix = {b: i for i, b in enumerate(kcs["bodyId"])}
    kc_list = [{"name": (r["type"] or "KC")} for _, r in kcs.iterrows()]

    mbons = mbon[mbon["bodyId"].isin(mbon_all)].copy().sort_values("bodyId")
    mbon_ix = {b: i for i, b in enumerate(mbons["bodyId"])}
    # innate-ish valence prior from predicted neurotransmitter (documented
    # simplification): ACh MBONs = approach-mediating, GABA/GLU = avoidance
    ntpath = os.path.join(DATA, "body-neurotransmitters.feather")
    ntdf = pd.read_feather(ntpath)
    ntdf = ntdf.sort_values("predicted_nt_confidence", ascending=False)
    ntdf = ntdf.drop_duplicates("body", keep="first")
    nt_of = dict(zip(ntdf["body"], ntdf["predicted_nt"].fillna("")))
    valence = []
    for b in mbons["bodyId"]:
        v = nt_of.get(b, "")
        if v == "acetylcholine": valence.append(1)
        elif v == "gaba": valence.append(-1)
        elif v == "glutamate": valence.append(-0.5)
        else: valence.append(0)
    mbon_list = [{"name": r["instance"] or r["type"], "v": valence[i]}
                 for i, (_, r) in enumerate(mbons.iterrows())]

    dans = dan[dan["bodyId"].isin(dan_all)].copy().sort_values("bodyId")
    dan_ix = {b: i for i, b in enumerate(dans["bodyId"])}
    dan_list = [{"name": r["instance"] or r["type"]} for _, r in dans.iterrows()]

    # drop edges whose endpoint is missing from the index (shouldn't happen
    # after union, but keeps the file consistent)
    pn_edges = [[glom_ix[g], pn_ix[b], int(min(w, 2000))]
                for g, b, w in orn_pn.itertuples(index=False) if b in pn_ix]
    kc_edges = [[pn_ix[a], kc_ix[b], int(min(w, 2000))]
                for a, b, w in pn_kc.itertuples(index=False)
                if a in pn_ix and b in kc_ix]
    mbon_edges = [[kc_ix[a], mbon_ix[b], int(min(w, 2000))]
                  for a, b, w in kc_mbon.itertuples(index=False)
                  if a in kc_ix and b in mbon_ix]
    dan_edges = [[dan_ix[a], mbon_ix[b], int(min(w, 2000))]
                 for a, b, w in dan_mbon.itertuples(index=False)
                 if a in dan_ix and b in mbon_ix]

    # ---- odor templates (fixed, sparse, deterministic) -------------------
    rng = np.random.default_rng(ODOR_SEED)
    odors = []
    for name in ODOR_NAMES:
        picks = rng.choice(len(gloms), size=ODOR_GLoms, replace=False)
        amps = np.round(rng.uniform(0.35, 1.0, size=ODOR_GLoms), 2)
        # vinegar overlaps apple (harder discrimination demo)
        tpl = {gloms[int(p)]: float(a) for p, a in zip(picks, amps)}
        odors.append({"name": name, "glom": tpl})
    if "vinegar" in ODOR_NAMES and "apple" in ODOR_NAMES:
        g_apple = odors[ODOR_NAMES.index("apple")]["glom"]
        t = odors[ODOR_NAMES.index("vinegar")]["glom"]
        shared = list(g_apple.items())[:5]
        for k, v in shared:
            t[k] = float(round(min(1.0, v * 0.9 + 0.1), 2))

    out = {
        "source": "MaleCNS v1.0 (HHMI Janelia, CC-BY-4.0) - real synapse weights",
        "glomeruli": gloms,
        "pn": pn_list, "kc": kc_list, "mbon": mbon_list, "dan": dan_list,
        "ornPn": pn_edges, "pnKc": kc_edges, "kcMbon": mbon_edges, "danMbon": dan_edges,
        "odors": odors,
    }
    path = os.path.join(DATA, "olfactory_circuit.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, separators=(",", ":"))
    print("wrote", path, f"({os.path.getsize(path)/1e6:.1f} MB)")
    print("glomeruli:", len(gloms), "| PNs:", len(pn_list), "| KCs:", len(kc_list),
          "| MBONs:", len(mbon_list), "| DANs:", len(dan_list))


if __name__ == "__main__":
    main()
