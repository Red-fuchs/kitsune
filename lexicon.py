"""ASR text fixes before routing: (1) spelled letters 'K I M I C' -> 'KIMIC'; (2) user-editable lexicon.tsv (variant<TAB>canonical, '#' comments), reloaded on mtime change."""
import re, pathlib

F = pathlib.Path(__file__).parent / "lexicon.tsv"
_SPELL = re.compile(r"(?<![\wÄÖÜäöü])(?:[A-Za-zÄÖÜ]\.?[ \t]+){2,}[A-Za-zÄÖÜ]\.?(?![\wÄÖÜäöü])")
_cache = {"mt": None, "rules": []}


def spell(q):
    """>=3 single letters in a row -> one upper-case word ('K I M I C' -> 'KIMIC', 'K. I. M. I. C.' -> 'KIMIC')."""
    return _SPELL.sub(lambda m: re.sub(r"[^A-Za-zÄÖÜ]", "", m.group()).upper(), q)


def rules():
    try: mt = F.stat().st_mtime
    except OSError: return []
    if mt != _cache["mt"]:
        out = []
        for ln in F.read_text().splitlines():
            if not ln.strip() or ln.lstrip().startswith("#") or "\t" not in ln: continue
            v, c = (x.strip() for x in ln.split("\t", 1))
            if v and c: out.append((re.compile(r"(?<![\wäöüß])" + r"[\s\-]*".join(map(re.escape, v.split())) + r"(?![\wäöüß])", re.I), c, len(v)))
        _cache.update(mt=mt, rules=[(p, c) for p, c, _ in sorted(out, key=lambda x: -x[2])])
    return _cache["rules"]


def apply(q):
    q = spell(q)
    for p, c in rules(): q = p.sub(c, q)
    return q
