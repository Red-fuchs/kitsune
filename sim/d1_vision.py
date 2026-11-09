#!/usr/bin/env python
"""d1-2B vs d1-omni-2B on pictures: classify, yes/no, count, boxes, read text.
gpu-run -n d1-vision --ram 10 ~/.venvs/mlxvlm/bin/python -I sim/d1_vision.py"""
import sys, json, pathlib, time, gc, traceback
HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(__import__('os').environ.get('KITSUNE_MLXVLM_SRC') or str(pathlib.Path.home() / 'Claude-Workspace/lfm3/mlx-vlm')))
from mlx_vlm import load

M = pathlib.Path.home() / "AI-Model/models/LiquidAI"
EX = pathlib.Path.home() / "Claude-Workspace/lfm3/mlx-vlm/examples/images"
D1T = pathlib.Path.home() / "Claude-Workspace/d1-test/images"
ANIMAL = {"type": "choice", "instructions": "Welches Tier ist zu sehen?", "criteria": {"hund": "Hund", "katze": "Katze", "fisch": "Fisch", "keins": "kein Tier"}}
KIND = {"type": "choice", "instructions": "Was für ein Bild ist das?", "criteria": {"foto": "Foto einer Szene", "diagramm": "Diagramm oder Grafik", "screenshot": "Bildschirmfoto einer Webseite/App", "beleg": "Kassenzettel oder Beleg", "dokument": "Textdokument/Papier"}}
# (image, questions, gold) — gold per key: value for choice, bool for noul, int for boxes count, substring for text
CASES = [
    (D1T / "dog.png", {"a": ANIMAL}, {"a": "hund"}),
    (D1T / "cat.png", {"a": ANIMAL}, {"a": "katze"}),
    (D1T / "fish.png", {"a": ANIMAL}, {"a": "fisch"}),
    (EX / "cats.jpg", {"a": ANIMAL, "n": {"type": "boxes", "instructions": "cat", "max": 4},
                       "c": {"type": "choice", "instructions": "Wie viele Katzen?", "criteria": {"eine": "eine Katze", "zwei": "zwei Katzen", "mehr": "drei oder mehr Katzen"}}},
     {"a": "katze", "n": 2, "c": "zwei"}),
    (EX / "desktop_setup.png", {"k": KIND, "plant": {"type": "noul", "instructions": "Ist eine Pflanze zu sehen?"},
                                "cat": {"type": "noul", "instructions": "Ist eine Katze zu sehen?"},
                                "spk": {"type": "boxes", "instructions": "loudspeaker", "max": 4},
                                "drum": {"type": "noul", "instructions": "Ist ein Schlagzeug/eine Trommel zu sehen?"}},
     {"k": "foto", "plant": True, "cat": False, "spk": 2, "drum": True}),
    (EX / "password.jpg", {"k": KIND, "pw": {"type": "text", "instructions": "Wie lautet das WLAN-Passwort?"},
                           "user": {"type": "text", "instructions": "Wie lautet der Username?"},
                           "wifi": {"type": "noul", "instructions": "Geht es um WLAN-Zugangsdaten?"}},
     {"k": "beleg", "pw": "aaeu", "user": "fqpp", "wifi": True}),
    (EX / "graph.png", {"k": KIND}, {"k": "diagramm"}),
    (EX / "renewables_california.png", {"k": KIND}, {"k": "diagramm"}),
    (EX / "webpage_1.png", {"k": KIND}, {"k": "screenshot"}),
    (EX / "paper.png", {"k": KIND}, {"k": "dokument"}),
]


def val(a):
    t = a["type"]
    if t == "noul": return a["noul"] >= 0.5, round(a["noul"], 2)
    if t == "choice": return a["choice"], round(a.get("confidence", 0), 2)
    if t == "boxes": return a["count"], a.get("presence")
    return a.get(t), a.get("found")


res = {}
for name, path in (("d1-2B", "d1-2B-MLX-8bit"), ("d1-omni", "d1-Omni-2B-MLX-8bit")):
    m, _ = load(str(M / path)); hit = tot = 0; rows = []
    for img, qs, gold in CASES:
        t = time.perf_counter()
        try:
            out = m.system_one(None, qs, media=[{"type": "image", "path": str(img)}])["answers"]
        except Exception as e:
            rows.append(f"{img.name}: ERROR {e!r}"); traceback.print_exc(); continue
        ms = round((time.perf_counter() - t) * 1000)
        for k, g in gold.items():
            v, extra = val(out[k]); good = (g.lower() in str(v).lower()) if isinstance(g, str) and out[k]["type"] == "text" else v == g
            hit += good; tot += 1
            rows.append(f"{'✓' if good else '✗'} {img.name:28} {k:5} → {v} ({extra}) soll {g}  [{ms} ms]")
    res[name] = {"score": f"{hit}/{tot}", "rows": rows}
    print(name, res[name]["score"], *rows, sep="\n", flush=True)
    del m; gc.collect()
(HERE / "data/d1_vision.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
