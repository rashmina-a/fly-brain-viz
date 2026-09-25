#!/usr/bin/env python3
"""Distill the public MaleCNS v1.0 connectome (HHMI Janelia, CC-BY-4.0) into
compact web-friendly data files for the 3D fly-brain viewer.

Outputs (into ../data):
  neurons.tsv.gz   one row per traced neuron: bodyId, name, type, class,
                   superclass, side, soma xyz (8nm voxel coords), neurotransmitter
  edges_pre.u32    synaptic edges sorted by body_pre (uint32)
  edges_post.u32   ... target bodyIds (uint32)
  edges_w.u16      ... synapse weights (uint16)
  edges_idx_*.u32  sorted unique pre bodyIds + row offsets (CSR-style lookup)
  meta.json        dataset provenance + counts for the UI
"""
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, "..", "data"))

ANN = os.path.join(DATA, "body-annotations.feather")
NT = os.path.join(DATA, "body-neurotransmitters.feather")
CW = os.path.join(DATA, "connectome-weights.feather")

SUBSET_CLASSES = [
    "olfactory", "gustatory", "mechanosensory", "mechanosensory_tactile",
    "mechanosensory_proprioceptive", "hygrosensory", "unknown_sensory",
    "ALPN", "ALLN", "DAN", "MBON", "Kenyon_Cell", "CX", "visual",
]


def main():
    print("reading annotations ...")
    ann = pd.read_feather(ANN)
    ann = ann[ann["status"] != "Glia"].copy()

    for col in ("instance", "type", "class", "superclass", "somaSide"):
        ann[col] = ann[col].fillna("")

    # soma xyz as int columns
    sl = ann["somaLocation"]
    has_xy = sl.notna()
    xyz = np.array(sl[has_xy].tolist(), dtype=np.int64)
    ann.loc[has_xy, "x"] = xyz[:, 0]
    ann.loc[has_xy, "y"] = xyz[:, 1]
    ann.loc[has_xy, "z"] = xyz[:, 2]

    print("reading neurotransmitters ...")
    nt = pd.read_feather(NT)
    nt = nt[nt["body"].notna()].copy()
    nt["body"] = nt["body"].astype(np.int64)
    nt = nt.sort_values("predicted_nt_confidence", ascending=False)
    nt = nt.drop_duplicates("body", keep="first")
    nt_map = dict(zip(nt["body"], nt["predicted_nt"].fillna("")))

    ann["nt"] = ann["bodyId"].map(nt_map).fillna("")

    cols = ["bodyId", "instance", "type", "class", "superclass",
            "somaSide", "x", "y", "z", "nt"]
    neurons = ann[cols].copy()
    neurons.to_csv(os.path.join(DATA, "neurons.tsv.gz"), sep="\t",
                   index=False, compression="gzip")
    print("neurons.tsv.gz:", len(neurons), "rows")

    print("reading connectome weights (1 GB, ~1 min) ...")
    cw = pd.read_feather(CW)
    print("edges:", len(cw))

    # edges whose endpoints are in the annotated-neuron table
    ok = cw["body_pre"].isin(neurons["bodyId"]) & \
        cw["body_post"].isin(neurons["bodyId"])
    edges = cw[ok].copy()
    print("edges with annotated endpoints:", len(edges))

    pre = edges["body_pre"].to_numpy(dtype=np.uint32)
    post = edges["body_post"].to_numpy(dtype=np.uint32)
    w = edges["weight"].to_numpy(dtype=np.uint32)
    w = np.minimum(w, 65535).astype(np.uint16)

    order = np.argsort(pre, kind="stable")
    pre, post, w = pre[order], post[order], w[order]

    uniq_pre, start_idx = np.unique(pre, return_index=True)
    offsets = np.append(start_idx, len(pre)).astype(np.uint32)

    def raw(arr, name):
        arr.tofile(os.path.join(DATA, name))

    raw(pre, "edges_pre.u32")
    raw(post, "edges_post.u32")
    raw(w, "edges_w.u16")
    raw(uniq_pre, "edges_idx_ids.u32")
    raw(offsets, "edges_idx_off.u32")

    # reverse CSR (sorted by body_post) so the UI can list a neuron's inputs
    order2 = np.argsort(post, kind="stable")
    pre2 = pre[order2]
    w2 = w[order2]
    uniq_post, start2 = np.unique(post[order2], return_index=True)
    off2 = np.append(start2, len(post)).astype(np.uint32)
    raw(pre2, "redges_pre.u32")
    raw(w2, "redges_w.u16")
    raw(uniq_post, "redges_idx_ids.u32")
    raw(off2, "redges_idx_off.u32")
    print("edges_*.u32 + reverse index written")

    # roi / compartment list for the UI (from class vocabulary)
    meta = {
        "source": "MaleCNS v1.0 (Drosophila melanogaster male CNS connectome)",
        "publisher": "HHMI Janelia Research Campus, Fly EM project team",
        "license": "CC-BY-4.0",
        "urls": {
            "project": "http://male-cns.janelia.org/",
            "neuprint": "https://neuprint.janelia.org/?dataset=male-cns:v1.0",
            "data": ("https://storage.googleapis.com/flyem-male-cns/"
                     "v1.0/connectome-data/flat-connectome/"),
        },
        "citation": ("Takeuchi, Bates, Rfiy et al. Sexual dimorphism in the "
                     "complete Drosophila male central nervous system. Cell (2026)."),
        "counts": {
            "neurons": int(len(neurons)),
            "edges": int(len(edges)),
        },
        "classes": sorted(c for c in neurons["class"].unique() if c),
        "superclasses": sorted(c for c in neurons["superclass"].unique() if c),
        "subset_classes": SUBSET_CLASSES,
    }
    with open(os.path.join(DATA, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
    print("meta.json written")


if __name__ == "__main__":
    main()
