import sys, json, pathlib, re
HERE = pathlib.Path(__file__).parent; sys.path[:0] = [str(HERE.parent)]
import numpy as np, addressee as A, rules as R
data = json.loads((HERE / "data/gate_eval.json").read_text()); y = np.array([d["label"] for d in data])
a = A.Addressee()
p = np.array([a.p_addr(d["q"], [tuple(c) for c in d.get("ctx", [])]) for d in data])
f = np.array([bool(R.is_question_or_cmd(d["q"]) or re.search(r"\b(du|dir|dich|bitte|mal|noch mal|nochmal)\b", d["q"].lower())) for d in data])
for k in (1.0, .9, .8, .7):
    q = np.where(f, p, p * k)
    print("k", k, " ".join(f"{t}:{(q[y==1]>=t).mean():.2f}/{(q[y==0]>=t).mean():.2f}" for t in (.5, .6, .7, .75, .8)))
print("feat pos", f[y==1].mean(), "neg", f[y==0].mean())
for d, pp, ff in zip(data, p, f):
    if d["label"] == 1 and not ff: print("pos no feat", d["q"], round(pp, 2))
