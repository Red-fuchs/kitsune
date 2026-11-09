"""e2e (end of user speech -> first audible Kitsune sample, measured on the virtual speaker) per tool path from sim json reports. python3 sim/latency.py sim/reports/<ts>_*.json"""
import sys, json, statistics as st, collections
by = collections.defaultdict(list)
for f in sys.argv[1:]:
    for s in json.load(open(f))["scenarios"]:
        for d in s["says"]:
            if "e2e_ms" in d and d.get("tool"): by[d["tool"].split(":")[0]].append(d["e2e_ms"])
print("| path | n | median ms | p90 ms |\n|---|---|---|---|")
allv = [v for l in by.values() for v in l]
for k, l in sorted(by.items(), key=lambda kv: st.median(kv[1])):
    print(f"| {k} | {len(l)} | {int(st.median(l))} | {int(sorted(l)[int(0.9 * (len(l) - 1))])} |")
print(f"| all | {len(allv)} | {int(st.median(allv))} | {int(sorted(allv)[int(0.9 * (len(allv) - 1))])} |")
