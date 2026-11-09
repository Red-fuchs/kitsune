#!/usr/bin/env python
"""barge-in verify: old (rules + addressee p>=0.9) vs new d1 question 'will der Nutzer unterbrechen?' with Kitsune's current sentence.
gpu-run -n k-barge --ram 8 ~/.venvs/mlxvlm/bin/python -I sim/barge_eval.py   (data: sim/data/barge_eval.json)"""
import sys, json, pathlib
HERE = pathlib.Path(__file__).parent; sys.path[:0] = [str(HERE.parent)]
import numpy as np, addressee as A, rules as R

D = json.loads((HERE / "data/barge_eval.json").read_text())["items"]
a = A.Addressee(); a.p_addr("Hallo.")


def rule(q, sp):
    if R.self_echo(q, sp): return 0
    if R.stop_word(q) or R.barge_stop(q) or R.barge_no(q) or R.NAME.search(q) or R.is_question_or_cmd(q): return 1
    return None


def auc(ps, y):
    th = np.sort(np.unique(ps)); return round(float(-np.trapezoid([(ps[y == 1] >= t).mean() for t in th], [(ps[y == 0] >= t).mean() for t in th])), 3) if len(set(y)) > 1 else None


rows = []
for d in D:
    pa = a.p_addr(d["q"]); ma = a.ms; pb = a.p_barge(d["q"], d["speaking"]); mb = a.ms
    rows.append({**d, "rule": rule(d["q"], d["speaking"]), "p_addr": pa, "p_barge": pb, "ms_a": ma, "ms_b": mb})
y = np.array([r["label"] for r in rows]); pa = np.array([r["p_addr"] for r in rows]); pb = np.array([r["p_barge"] for r in rows])
res = [r for r in rows if r["rule"] is None]; yr = np.array([r["label"] for r in res])
print(f"n={len(y)} pos={y.sum()} | residual after rules n={len(res)} pos={yr.sum()}")
print("AUC all:      addr", auc(pa, y), " barge", auc(pb, y))
print("AUC residual: addr", auc(np.array([r["p_addr"] for r in res]), yr), " barge", auc(np.array([r["p_barge"] for r in res]), yr))
print("latency ms med addr", int(np.median([r["ms_a"] for r in rows])), "barge", int(np.median([r["ms_b"] for r in rows])))


def pipe(key, t, rules=True):
    pr = np.array([(r["rule"] if rules and r["rule"] is not None else int(r[key] >= t)) for r in rows])
    return (pr[y == 1] == 1).mean(), (pr[y == 0] == 1).mean()


print("\npipeline  hit-rate(real interrupts)  false-barge(TV/3rd/echo/backchannel)")
print("old rules+addr>=0.9      %.2f  %.2f" % pipe("p_addr", .9))
for t in (.3, .4, .5, .6, .7, .8): print(f"rules+barge>={t}        %.2f  %.2f" % pipe("p_barge", t))
for t in (.4, .5, .6, .7): print(f"barge only>={t} (no rules) %.2f  %.2f" % pipe("p_barge", t, False))
print("\nper item (label rule p_addr p_barge):")
for r in sorted(rows, key=lambda r: r["p_barge"]): print(f"{r['label']} {str(r['rule']):4s} {r['p_addr']:.2f} {r['p_barge']:.2f}  {r['q']}")
(HERE / "reports/barge_eval.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1))
