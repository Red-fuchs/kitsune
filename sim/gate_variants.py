import sys, json, pathlib
HERE = pathlib.Path(__file__).parent; sys.path[:0] = [str(HERE.parent)]
import numpy as np, addressee as A
data = json.loads((HERE / "data/gate_eval.json").read_text()); y = np.array([d["label"] for d in data])
a = A.Addressee()
QS = {
 "A": {"type": "noul", "instructions": A.Q["addr"]["instructions"]},
 "B": {"type": "noul", "instructions": "Richtet der Sprecher einen Befehl oder eine Frage an einen digitalen Sprachassistenten, die dieser beantworten kann (Wetter, Kalender, Wissen, Rechnen, Smalltalk mit dem Assistenten)?"},
 "tv": {"type": "noul", "instructions": "Stammt der Satz aus Nachrichten, Film, Radio oder Fernsehen (Erzählstil, Sprecher spricht über Dritte)?"},
 "human": {"type": "noul", "instructions": "Spricht der Sprecher mit einer anderen anwesenden Person (Familie, Partner, Kind, Telefon) statt mit einem Gerät?"},
}
def st(d, style):
    q, ctx = d["q"], d.get("ctx", [])
    if style == 0: return A.state(q, ctx)
    s = ""
    if ctx: s += "Vorher im Gespräch:\n" + "\n".join(f"Nutzer: {u}\nAssistent: {x[:120]}" for u, x in ctx) + "\n"
    return s + f"Gehörter Satz: {q}"
res = {}
for style in (0, 1):
    R = [a.m.system_one(st(d, style), QS)["answers"] for d in data]
    for k in QS: res[(style, k)] = np.array([r[k]["noul"] for r in R])
def auc(p):
    pos, neg = p[y == 1], p[y == 0]; return float(np.mean([(x > z) + .5 * (x == z) for x in pos for z in neg]))
for (style, k), p in res.items(): print(style, k, "AUC", round(auc(p), 3) if k in "AB" else round(auc(1 - p), 3))
for style in (0, 1):
    for nm, p in [("A", res[(style, "A")]), ("B", res[(style, "B")]), ("B*(1-tv)", res[(style, "B")] * (1 - res[(style, "tv")])), ("B*(1-tv)*(1-human)", res[(style, "B")] * (1 - res[(style, "tv")]) * (1 - res[(style, "human")])),
                  ("A*(1-tv)*(1-human)", res[(style, "A")] * (1 - res[(style, "tv")]) * (1 - res[(style, "human")])), ("mean(A,B)", (res[(style, "A")] + res[(style, "B")]) / 2)]:
        print(style, nm, "AUC", round(auc(p), 3), " ".join(f"{t}:{(p[y==1]>=t).mean():.2f}/{(p[y==0]>=t).mean():.2f}" for t in (.3, .5, .7, .85)))
