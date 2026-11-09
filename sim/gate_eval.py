#!/usr/bin/env python
"""Offline threshold eval for the addressee gate. gpu-run -n k-gate --ram 8 ~/.venvs/mlxvlm/bin/python -I sim/gate_eval.py
data: sim/data/gate_eval.json (label 1 = addressed). Prints per-item p, ROC, best thresholds, latency."""
import sys, json, pathlib, time
HERE = pathlib.Path(__file__).parent; sys.path[:0] = [str(HERE.parent)]
import numpy as np
import addressee as A

data = json.loads((HERE / "data/gate_eval.json").read_text())
a = A.Addressee(); a.p_addr("Hallo.")
ps, lat = [], []
for d in data:
    p = a.p_addr(d["q"], [tuple(c) for c in d.get("ctx", [])]); ps.append(p); lat.append(a.ms)
ps = np.array(ps); y = np.array([d["label"] for d in data])
for d, p in sorted(zip(data, ps), key=lambda z: z[1]): print(f"{p:.3f} {d['label']} {d['q'][:70]}")
print(f"\nn={len(y)} pos={y.sum()} neg={(1 - y).sum()}  latency ms median={np.median(lat):.0f} p90={np.percentile(lat, 90):.0f} max={max(lat)}")
print("thr   TPR(addressed passed)  FPR(non-addressed passed)")
for t in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95):
    print(f"{t:.2f}  {(ps[y == 1] >= t).mean():.2f}  {(ps[y == 0] >= t).mean():.2f}")
th = np.sort(np.unique(ps)); tpr = [(ps[y == 1] >= t).mean() for t in th]; fpr = [(ps[y == 0] >= t).mean() for t in th]
auc = -np.trapezoid(tpr, fpr); print("AUC", round(float(auc), 3))
