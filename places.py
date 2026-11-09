"""ASR place-name correction: phonetic fuzzy match over fakt.db places. API: cands(s)->[(label,score)], decide(c)."""
import sqlite3, json, os, re, sys, math, hashlib, unicodedata
from collections import Counter, defaultdict

DB = os.path.expanduser("~/AI-Model/fakt/fakt.db")
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
MINSL, RESORT_SL = 40, 15
_P = [("tsch", "s"), ("sch", "s"), ("ch", "x"), ("ck", "k"), ("ph", "f"), ("tz", "s"), ("ts", "s"), ("z", "s"), ("c", "k"), ("qu", "kf"), ("q", "k"),
      ("v", "f"), ("w", "f"), ("au", "o"), ("eu", "oi"), ("äu", "oi"), ("ie", "i"), ("ee", "i"), ("ea", "i"), ("ey", "ei"), ("y", "i"), ("ä", "e"), ("ö", "o"), ("ü", "u"), ("é", "e"), ("ß", "s"),
      ("d", "t"), ("b", "p"), ("g", "k"), ("h", "")]
_ADM = {'Oblast', 'Landkreis', 'Provinz', 'Kreis', 'Bezirk', 'Region', 'Departement', 'Distrikt', 'Gemeinde', 'Verwaltungsgemeinschaft', 'Präfektur', 'Gespanschaft', 'Gouvernement', 'Okrug', 'Rajon', 'Kanton', 'Verwaltungsbezirk', 'Stadtbezirk', 'Arrondissement', 'Woiwodschaft', 'Powiat'}
_PRE = re.compile(r"^(herr|frau|in|nach|von)\s+")

def norm(s):
    s = unicodedata.normalize("NFKD", s.lower()); s = "".join(c for c in s if not unicodedata.combining(c) or c in "̈")  # keep umlaut marks for folding
    s = unicodedata.normalize("NFC", s)
    return re.sub(r"[^a-zäöüß ]", " ", s.replace("-", " "))

def key(s):
    s = _PRE.sub("", norm(s).strip()).replace(" ", "")
    for a, b in _P: s = s.replace(a, b)
    s = re.sub(r"(.)\1+", r"\1", s)
    return re.sub(r"[^a-z]", "", s)

def skel(k):  # consonant skeleton; vowels are what ASR garbles most
    r = re.sub(r"[aeiou]+", "", k[1:]); return k[:1] + r

def tris(k): k = f"^{k}$"; return {k[i:i + 3] for i in range(len(k) - 2)}

_G = ["lnmr", "sxf", "tkp", "aeiouäöü"]
_GC = {(a, b): .5 for g in _G for a in g for b in g if a != b}
_GC.update({(a, b): .35 for a in "lnm" for b in "lnm" if a != b})
_GC.update({(a, b): .3 for a in "sx" for b in "sx" if a != b})
_W = {"f": .6, "x": .6, "s": .7, "h": .5}
_V = set("aeiou")

def lev(a, b):  # weighted: confusable phonemes (l/n/m, s/f/x, t/k/p, vowels) are cheap substitutions
    d = [j * .8 for j in range(len(b) + 1)]
    for i in range(1, len(a) + 1):
        p, d[0] = d[0], i * .8; x = a[i - 1]
        for j in range(1, len(b) + 1):
            y = b[j - 1]
            p, d[j] = d[j], min(d[j] + (.6 if x in _V else _W.get(x, 1)), d[j - 1] + (.6 if y in _V else _W.get(y, 1)), p + (0 if x == y else _GC.get((x, y), 1)))
    return d[-1]

def lr(a, b): return 1 - lev(a, b) / max(len(a), len(b), 1)

def _build(path):
    c = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    c.execute("create temp table pl as select q from facts where p=1082 union select q from facts where p=31 and (val like '%Seebad%' or val like '%Kurort%' or val like '%Ferienort%' or val like '%Badeort%' or val like '%Wintersportort%' or val like '%Touristenort%' or val like '%Urlaubsort%')")
    res = c.execute("select q from facts where p=31 and (val like '%Seebad%' or val like '%Kurort%' or val like '%Ferienort%' or val like '%Badeort%')").fetchall()
    rs = {r[0] for r in res}
    its = [(q, l, sl) for q, l, sl in c.execute("select i.q,i.label,i.sl from items i join pl using(q)") if l and (sl >= MINSL or (q in rs and sl >= RESORT_SL))]
    ok = {q: (l, sl) for q, l, sl in its}
    seen, out = set(), []
    for q, n in c.execute("select q,norm from nm where q in (select q from pl)"):
        if q not in ok or len(n) < 3: continue
        l, sl = ok[q]; k = key(n)
        if l.split()[0] in _ADM: continue
        if len(k) < 3 or (q, k) in seen: continue
        seen.add((q, k)); out.append([l, k, sl, norm(n).strip()])
    out.sort(key=lambda x: -x[2])
    json.dump(out, open(path, "w"), ensure_ascii=False, separators=(",", ":"))
    return out

class Idx:
    def __init__(s):
        st = os.stat(DB); h = hashlib.md5(f"{st.st_size}{st.st_mtime_ns}{MINSL}{RESORT_SL}v8{[p for p in _P]}".encode()).hexdigest()[:10]
        os.makedirs(CACHE, exist_ok=True); path = os.path.join(CACHE, f"places-{h}.json")
        s.d = json.load(open(path)) if os.path.exists(path) else _build(path)
        s.sk = [skel(e[1]) for e in s.d]
        s.inv = defaultdict(list)
        for i, e in enumerate(s.d):
            for t in tris(e[1]): s.inv[t].append(i)
        s.sinv = defaultdict(list)
        for i, k in enumerate(s.sk):
            for t in tris(k): s.sinv[t].append(i)

    def cands(s, q, n=8):
        k = key(q); nq = norm(q).strip()
        if len(k) < 2: return []
        qs, qt, qst = skel(k), tris(k), tris(skel(k))
        cnt = Counter()
        for t in qt:
            p = s.inv.get(t, ())
            if len(p) < 15000: cnt.update(p)
        for t in qst:
            p = s.sinv.get(t, ())
            if len(p) < 15000: cnt.update(p)
        best = {}
        for i, _ in cnt.most_common(600):
            l, kk, sl, nn = s.d[i]
            if abs(len(kk) - len(k)) > max(4, len(k) // 2): continue
            tk = tris(kk); tri = 2 * len(qt & tk) / (len(qt) + len(tk))
            sc = .6 * lr(k, kk) + .1 * tri + .3 * lr(qs, s.sk[i])
            if kk == k: sc = 1.0 if nn == nq else (1.0 if key(l) == kk else .88)
            sc = sc * sc * (.90 + .10 * min(1, math.log(sl + 1) / math.log(250)))  # square: fuzzy scores are generous, spread them
            if sc > best.get(l, 0): best[l] = sc
        return sorted(((l, round(min(v, 1.0), 3)) for l, v in best.items()), key=lambda x: -x[1])[:n]

_I = None
def cands(q, n=8):
    global _I
    _I = _I or Idx(); return _I.cands(q, n)

def decide(c, hi=.8, lo=.6):
    if not c: return ("none", None)
    (l, s) = c[0]; r = [x for x in c[1:] if l not in x[0]]
    if s >= hi and not (r and r[0][1] >= s - .1): return ("use", l)
    if lo <= s: return ("ask", l)
    return ("none", None)

if __name__ == "__main__":
    for a in sys.argv[1:]:
        c = cands(a)
        for l, s in c[:5]: print(f"{s:.3f}  {l}")
        print(a, "->", decide(c))
