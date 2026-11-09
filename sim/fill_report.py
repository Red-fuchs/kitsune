"""Summarise sim reports of the fill scenarios: python3 sim/fill_report.py <tagprefix>... (e.g. _fill_c0_x0 _fill_c2_x800). Per config over all rooms: wiki/netz turns, filler used, underruns, first answer audio."""
import sys, json, glob, statistics as st, collections
for tag in sys.argv[1:]:
    rows = collections.defaultdict(list)
    for f in sorted(glob.glob(f"{__file__.rsplit('/', 1)[0]}/reports/*{tag}_*.json")):
        d = json.load(open(f))
        for s in d["scenarios"]:
            for t in s["turns"]:
                tool = (t.get("tool") or "").split(":")[0]
                if tool in ("wiki_suche", "netz_suche"): rows[tool].append((d["room"], t))
    for tool, l in rows.items():
        u = [t.get("under") or 0 for _, t in l]; e = [t["ms"].get("e2e_first_audio_ms") for _, t in l if t["ms"].get("e2e_first_audio_ms")]
        print(f"{tag} {tool}: n={len(l)} turns with underruns {sum(1 for x in u if x)} ({sum(u)} callbacks = {sum(u) * 0.043:.1f} s) | first audible med {int(st.median(e)) if e else '-'} ms | rooms {sorted(set(r for r, _ in l))}")
