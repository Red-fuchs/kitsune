#!/usr/bin/env python
"""Merge per-room json files of one or more runs into one markdown: report.py <ts> [<ts> ...] -> reports/merged.md"""
import sys, json, glob, pathlib, datetime
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import run
rep = pathlib.Path(__file__).parent / "reports"; allres = {}
for ts in sys.argv[1:]:
    for f in sorted(glob.glob(str(rep / f"{ts}_*.json"))):
        d = json.load(open(f)); allres[d["room"]] = d["scenarios"]
md = run.md_report(allres, dict(ts="merged " + " ".join(sys.argv[1:]), voice="thorsten", wall_min="?")) + run.bias_md(allres)
(rep / "merged.md").write_text(md); print(md)
