"""Tool schemas (router grammar 'e': call-only), grammar generator, allowlisted executors."""
import atexit, json, os, re, select, shutil, subprocess, pathlib, threading
from datetime import datetime

HOME = pathlib.Path.home()
DRY = lambda: os.environ.get("KITSUNE_DRYRUN", "0") == "1"


def tool(name, desc, props=None, req=None):
    props = props or {}
    return {"type": "function", "function": {"name": name, "description": desc,
            "parameters": {"type": "object", "properties": props, "required": list(props) if req is None else req}}}


S = lambda d: {"type": "string", "description": d}
TOOLS = [
    tool("kal_heute", "Zeigt die Termine von heute."),
    tool("kal_woche", "Zeigt die Termine der nächsten Tage dieser Woche."),
    tool("kal_eintragen", "Trägt einen neuen Termin in den Kalender ein.",
         {"titel": S("Titel des Termins"), "wann": S("Datum und Uhrzeit genau so, wie der Nutzer es gesagt hat, z.B. 'morgen um halb drei'")}),
    tool("netz_news", "Aktuelle Nachrichten (Tagesschau, DLF, heise, Spiegel), optional zu einem Thema.", {"thema": S("Thema (optional)")}, []),
    tool("wetter", "Wettervorhersage für einen Ort.", {"ort": S("Ort, z.B. Basel"), "wann": S("heute, morgen, übermorgen oder Wochentag (optional)")}, ["ort"]),
    tool("netz_suche", "Sucht aktuelle Infos im Internet: Preise, Sportergebnisse, Fahrpläne, Öffnungszeiten.", {"frage": S("Suchanfrage")}),
    tool("netz_lies", "Liest eine Webseite oder einen Treffer aus der letzten Liste.", {"url": S("URL oder Nummer des Treffers")}),
    tool("wiki_suche", "Schlägt Allgemeinwissen offline in der Wikipedia nach: Personen, Orte, Begriffe, Geschichte.", {"frage": S("Vollständige, eigenständige Frage")}),
    tool("bibliothek", "Durchsucht die libertäre Bibliothek (Rothbard, Hoppe, Mises, Spooner, Kinsella u. a.): Was sagt Autor X zu Thema Y.", {"frage": S("Vollständige, eigenständige Frage")}),
    tool("fakt", "Exakte Zahlenfakten: Einwohnerzahl, Höhe, Fläche, Geburtsdatum, Hauptstadt.", {"frage": S("Vollständige, eigenständige Frage")}),
    tool("dict", "Übersetzt ein Wort oder eine kurze Wendung zwischen Deutsch, Bulgarisch und Englisch (Wörterbuch) und sagt es vor.", {"wort": S("Das Wort, in der Ausgangssprache"), "nach": S("Zielsprache: bg, de oder en")}),
    tool("gpu_status", "Zeigt Auslastung und laufende Jobs der GPU."),
    tool("wissen", "Beantwortet eine Frage aus eigenem Wissen ohne Nachschlagen: Erklärungen, Rat, Rechnen, Folgefragen zum Gespräch.", {"frage": S("Vollständige, eigenständige Frage")}),
    tool("plaudern", "Smalltalk und Fragen an den Assistenten selbst: Wie geht es dir, wer bist du, Gefühle, Witze, Lob, Beleidigungen."),
    tool("antwort", "Nur für Hallo, Danke, Tschüss, Okay.", {"text": S("Höchstens drei Wörter")}),
]
TOOL_IDS = ["<tool_call>", "</tool_call>", "<function=", "</function>", "<parameter=", "</parameter>"]
WD = ["Montag", "Dienstag", "Mittwoch", "Donnerstag", "Freitag", "Samstag", "Sonntag"]


CHAT_RULE = ("" if os.environ.get("KITSUNE_ROUTER_CHAT", "0") != "1" else  # opt-in: the prompt rule alone cost a tool regression (wake_direkt) and gained less than rules.chat_demote
            
             "Ohne klare Werkzeug-Absicht wählst du plaudern: Aussagen, Erzählungen, Gefühle, Lob und Beleidigungen des Nutzers ohne Frage oder Auftrag ('Ich habe heute ... gebacken', 'Mein Bruder hat Geburtstag', 'Du bist ein dummer Assistent', 'Ich bin müde') "
             "und Fragen an dich selbst ('Hast du eine Lieblingsfarbe?', 'Bist du müde?') -> plaudern, auch wenn heute, morgen, Orte oder Namen vorkommen. Kalender nur bei Termin, Kalender oder 'Was steht an'. ")


def sys_router(now=None):
    """static: the tool-prompt prefix must be byte-identical every turn (KV prefix cache). Date, place and pending state go into the user message (ctx.state_line)."""
    return ("Du bist der Router eines deutschen Sprachassistenten. "
            "Antworte IMMER mit genau einem Werkzeug-Aufruf. Kalender -> kal_*; Nachrichten -> netz_news; Wetter, Temperatur, Regen, Wind -> wetter (nur dann); Preise, Sport, aktuelle Infos -> netz_suche; "
            "Personen, Orte, Begriffe -> wiki_suche; Libertäre Autoren/Theorie, 'was sagt Hoppe/Rothbard/Mises …' -> bibliothek; Zahlen wie Einwohner oder Höhe -> fakt; Übersetzung eines Worts ('Was heißt X auf Bulgarisch', 'Wie sagt man X auf Englisch') -> dict; Erklärungen, Rat, Rechnen, Folgefragen -> wissen; "
            "Smalltalk, Fragen an dich selbst, Gefühle, Witze -> plaudern; nur Hallo, Danke, Tschüss -> antwort. Fragen immer vollständig und eigenständig formulieren. "
            + CHAT_RULE +
            "Jede neue vollständige Frage wählt ihr Werkzeug neu, unabhängig vom vorigen Werkzeug. "
            "Das Transkript stammt aus Spracherkennung und kann Fehler haben; interpretiere Namen und Begriffe phonetisch. "
            "Beispiele: 'Was läuft auf der Gebühr' = GPU -> gpu_status; 'Wetter in Blau tief' = Plowdiw -> wetter; 'Herr Kensoli-Bach' = Sunny Beach; 'Zunge' = Kitsune; 'heute Morgen' = heute früh, 'morgen' = der nächste Tag.")


def user_prefix(now: datetime): return f"[Heute ist {WD[now.weekday()]}, {now:%Y-%m-%d}, {now:%H:%M} Uhr] "


def lark(ids, tools=TOOLS):
    """call-only grammar; tool tags are special tokens, referenced as <[id]>."""
    TC, TCE, FN, FNE, PA, PAE = (f"<[{ids[k]}]>" for k in TOOL_IDS)
    rules, alts = [], []
    for t in tools:
        f = t["function"]; n = f["name"]; props = f["parameters"]["properties"]; req = set(f["parameters"]["required"])
        body = ""
        for k in props:
            p = f'{PA} "{k}>\\n" VAL "\\n" {PAE} "\\n"'
            body += (f"({p})? " if k not in req else p + " ")
        rules.append(f'call_{n}: {TC} "\\n" {FN} "{n}>\\n" {body}{FNE} "\\n" {TCE}')
        alts.append(f"call_{n}")
    return "start: call\n\ncall: " + " | ".join(alts) + "\n" + "\n".join(rules) + "\nVAL: /[^<\\n]+/\n"


def parse_call(txt):
    m = re.search(r"<tool_call>(.*?)</tool_call>", txt, re.S)
    if not m: return None, {}
    f = re.search(r"<function=([^>\n]+)>(.*?)</function>", m.group(1), re.S)
    if not f: return None, {}
    args = {k.strip(): v.strip() for k, v in re.findall(r"<parameter=([^>\n]+)>\n?(.*?)\n?</parameter>", f.group(2), re.S)}
    return f.group(1).strip(), args


def run(cmd, timeout=25, env=None):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env={**os.environ, **(env or {})})
        return p.returncode, (p.stdout or "") + ("" if p.returncode == 0 else p.stderr)
    except subprocess.TimeoutExpired:
        return 124, "Zeitüberschreitung"
    except FileNotFoundError:
        return 127, "nicht verfügbar"


def _find(name, extra=()):
    return shutil.which(name) or next((str(p) for p in map(pathlib.Path, extra) if p.exists()), None)


def _j(s, default=None):
    try: return json.loads(s)
    except Exception: return default


def _hm(iso): return iso[11:16]


def fmt_events(js):
    seen, out = set(), []
    for e in js:
        k = (e.get("title"), e.get("start"))
        if k in seen: continue
        seen.add(k)
        d = datetime.fromisoformat(e["start"][:19])
        when = "ganztägig" if e.get("allDay") else f"{_hm(e['start'])} bis {_hm(e['end'])} Uhr"
        out.append(f"{WD[d.weekday()]} {d:%d.%m.}, {when}: {e['title']}" + (f" ({e['location']})" if e.get("location") else ""))
    return "\n".join(out) or "Keine Termine."


_WS = {"p": None}
_WS_LOCK = threading.Lock()
_KP = __import__("concurrent.futures").futures.ThreadPoolExecutor(2)
_KC = {}  # which -> (monotonic, result): read-only calendar cache (60 s), warmed in a thread as soon as a turn starts -> kal_heute answers without the 0.8 s `kal` subprocess
_KF = {}  # which -> in-flight Future (a running prefetch is awaited, never duplicated)


def _kal_fetch(which):
    import time
    rc, o = run(["kal", which]); js = _j(o)
    res = js if isinstance(js, list) else f"Kalender nicht lesbar: {o[:200]}"
    if isinstance(res, list): _KC[which] = (time.monotonic(), res)
    return res


def t_kal(which, ttl=60):
    import time, concurrent.futures as cf
    c = _KC.get(which)
    if c and time.monotonic() - c[0] < ttl: return c[1]
    with _WS_LOCK:
        f = _KF.get(which)
        if f is None or f.done(): f = _KF[which] = _KP.submit(_kal_fetch, which)
    return f.result()


def prefetch_kal(ttl=45):
    import time
    for w in ("today", "week"):
        c = _KC.get(w)
        if not (c and time.monotonic() - c[0] < ttl) and not (_KF.get(w) and not _KF[w].done()): t_kal_bg(w)


def t_kal_bg(w):
    with _WS_LOCK:
        f = _KF.get(w)
        if f is None or f.done(): _KF[w] = _KP.submit(_kal_fetch, w)


def t_news(thema=None):
    rc, o = run(["netz", "rss", "--json", "-n", "40" if thema else "6", "--since", "24h"]); js = _j(o)
    if not isinstance(js, list): return f"Nachrichten nicht abrufbar: {o[:200]}", []
    if thema:
        ws = [w for w in re.findall(r"\w{3,}", thema.lower())]
        js = [i for i in js if any(w in (i["title"] + " " + i.get("summary", "")).lower() for w in ws)][:6]
        if not js: return f"Keine aktuellen Meldungen zum Thema {thema} gefunden.", []
    items = [(i["link"], i["title"]) for i in js]
    return "\n".join(f"{n}. [{i['source']}] {i['title']} - {i.get('summary', '')}" for n, i in enumerate(js, 1)), items


def t_suche(q):
    if fx := os.environ.get("KITSUNE_SUCHE_FIXTURE"):  # sim only (DuckDuckGo blocks bots): {"delay": s, "out": text, "items": [[url, title]]}
        import time; js = json.loads(pathlib.Path(fx).read_text()); time.sleep(js.get("delay", 0)); return js["out"], [tuple(x) for x in js.get("items", [])]
    rc, o = run(["netz", "suche", "--json", "-n", "4", q]); js = _j(o)
    if not isinstance(js, list) or not js: return f"Keine Treffer ({o[:100]})", []
    return "\n".join(f"{n}. {i['title']}: {i.get('snippet', '')}" for n, i in enumerate(js, 1)), [(i["url"], i["title"]) for i in js]


def _cut(t, n):
    """t cut to <= n chars at a sentence end (else word end)."""
    if len(t) <= n: return t
    h = t[:n]; i = max(h.rfind(". "), h.rfind("! "), h.rfind("? "))
    return h[:i + 1] if i >= n * 0.5 else h[:h.rfind(" ")].rstrip(",;:") + "…" if " " in h else h


def trim_ctx(name, out, n=None):
    """tool output -> what Gemma needs: title + lead of the top passage first, then (shorter) further hits until ~n chars. Fewer prompt tokens = earlier first token.
    n: KITSUNE_CTX_CHARS (default 800; 0 = old behaviour, 2500 hard cap). netz_lies (reading a page) keeps a longer lead."""
    n = int(os.environ.get("KITSUNE_CTX_CHARS", "800")) if n is None else n
    if n <= 0: return out[:2500]
    if name == "netz_lies": n = max(n, 1500)
    if len(out) <= n: return out
    left, res = n, []
    for i, l in enumerate(x for x in out.split("\n") if x.strip()):
        if left < 90: break
        l = _cut(l.strip(), left if i == 0 else min(left, 260)); res.append(l); left -= len(l) + 1
    return "\n".join(res)


def spoken_url(u):
    u = u.strip().lower().replace(" punkt ", ".").replace(" ", "")
    return u if u.startswith("http") else "https://" + u


def t_lies(url, last):
    n = {"erste": 1, "erster": 1, "erstes": 1, "zweite": 2, "dritte": 3, "vierte": 4, "fünfte": 5, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5}.get(url.strip().lower().rstrip("."))
    if n and n <= len(last): url = last[n - 1][0]
    elif n or "." not in url: return "Welchen Artikel meinst du? Ich habe gerade keine Liste."
    else: url = spoken_url(url)
    rc, o = run(["netz", "lies", "--json", "--max", "2500", url], timeout=30); js = _j(o)
    return (js or {}).get("text") or f"Seite nicht lesbar: {o[:150]}"




def _wiki_serve(w, env, q, k=3, timeout=15):
    """Query a lazily started persistent `wiki serve`; raises on any failure (caller falls back to CLI)."""
    with _WS_LOCK:
        for attempt in (0, 1):
            p = _WS["p"]
            if p is None or p.poll() is not None:
                p = _WS["p"] = subprocess.Popen([w, "serve"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, env={**os.environ, **env})
            try:
                p.stdin.write(json.dumps({"q": q, "k": k}) + "\n"); p.stdin.flush()
                r = select.select([p.stdout], [], [], timeout if attempt == 0 else 60)[0]
                if not r: raise TimeoutError("wiki serve")
                line = p.stdout.readline()
                if not line: raise BrokenPipeError("wiki serve died")
                return json.loads(line).get("hits") or []
            except Exception:
                try: p.kill()
                except Exception: pass
                _WS["p"] = None
                if attempt: raise


def wiki_stop():
    p = _WS["p"]; _WS["p"] = None
    if p and p.poll() is None:
        try: p.stdin.close(); p.wait(2)
        except Exception: p.kill()


atexit.register(wiki_stop)


def t_wiki(q):
    w = _find("wiki", [HOME / ".local/bin/wiki", HOME / "Projects/wiki/target/release/wiki"])
    if not w: return "Wikipedia nicht verfügbar."
    env = {} if (HOME / "AI-Model/wiki").exists() else {"WIKI_DIR": str(HOME / "Projects/wiki/idx")}
    js = None
    if not os.environ.get("KITSUNE_WIKI_CLI"):
        try: js = _wiki_serve(w, env, q)
        except Exception: js = None
    if js is None:
        rc, o = run([w, "ask", "--json", "-k", "3", q], env=env); js = _j(o)
    if not isinstance(js, list) or not js: return "Kein Wikipedia-Treffer (Index evtl. nur Beispieldaten)."
    return "\n".join(f"{i['title']}{' / ' + i['section'] if i.get('section') else ''}: {i['text'][:500]}" for i in js)


_BS = {"p": None}
_BS_LOCK = threading.Lock()


def _buch_serve(w, q, k=3, timeout=20):
    """Query a lazily started persistent `buch serve` (own process); raises on any failure."""
    with _BS_LOCK:
        for attempt in (0, 1):
            p = _BS["p"]
            if p is None or p.poll() is not None:
                p = _BS["p"] = subprocess.Popen([w, "serve"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
            try:
                p.stdin.write(json.dumps({"q": q, "k": k}) + "\n"); p.stdin.flush()
                r = select.select([p.stdout], [], [], timeout if attempt == 0 else 60)[0]
                if not r: raise TimeoutError("buch serve")
                line = p.stdout.readline()
                if not line: raise BrokenPipeError("buch serve died")
                return json.loads(line).get("hits") or []
            except Exception:
                try: p.kill()
                except Exception: pass
                _BS["p"] = None
                if attempt: raise


def buch_stop():
    p = _BS["p"]; _BS["p"] = None
    if p and p.poll() is None:
        try: p.stdin.close(); p.wait(2)
        except Exception: p.kill()


atexit.register(buch_stop)


def fmt_buch(hits):
    """hits -> one spoken-ready line each: 'Laut Hoppe, Die Privatisierung der Verteidigung: text' (no page numbers, no URLs)."""
    out = []
    for h in hits:
        au = (h.get("author") or h.get("title", "").split(":")[0]).split()[-1:] or [""]
        out.append(f"Laut {au[0]}, {h.get('work') or h.get('title', '')}: " + " ".join(h.get("text", "").split())[:500])
    return "\n".join(out)


def t_buch(q):
    w = os.environ.get("KITSUNE_BUCH") or _find("buch", [HOME / ".local/bin/buch", HOME / "Projects/wiki/target/release/buch"])
    if not w: return "Die Bibliothek ist nicht verfügbar."
    try: js = _buch_serve(w, q)
    except Exception: return "Die Bibliothek ist nicht verfügbar."
    return fmt_buch(js) if js else "Dazu steht nichts in der Bibliothek."


def t_fakt(q):
    w = _find("fakt", [HOME / ".local/bin/fakt", HOME / "Projects/fakt/target/release/fakt"])
    if not w: return "nicht verfügbar"
    rc, o = run([w, "--json", q]); return o[:1500] if rc == 0 else "nicht verfügbar"


def t_wetter(ort, tage=3):
    """-> parsed dict from `netz wetter --json`, or error string. KITSUNE_WX_FIXTURE=<json file> mocks it."""
    if fx := os.environ.get("KITSUNE_WX_FIXTURE"): return _j(pathlib.Path(fx).read_text())
    rc, o = run(["netz", "wetter", ort, "--tage", str(tage), "--json"], timeout=15); js = _j(o)
    if rc == 0 and isinstance(js, dict) and js.get("tage"): return js
    return "Den Ort kenne ich nicht." if rc == 2 else "Die Wetterdaten sind gerade nicht erreichbar."


def t_gpu():
    rc, o = run(["gpu-run", "--status"]); return o.strip()[:800]


def kal_add_cmd(titel, at):
    return ["kal", "add", titel, "--at", at]


def t_kal_add(titel, at):
    cmd = kal_add_cmd(titel, at)
    if DRY(): return "[dry-run] " + " ".join(cmd), True
    rc, o = run(cmd)
    if rc == 0: _KC.clear()  # new entry -> cached day/week lists are stale
    return (o.strip()[:200] or "eingetragen"), rc == 0


# ---------- speakable templates (no LLM) ----------
_U = ("null ein zwei drei vier fünf sechs sieben acht neun zehn elf zwölf dreizehn vierzehn fünfzehn sechzehn siebzehn achtzehn neunzehn").split()
_T = {2: "zwanzig", 3: "dreißig", 4: "vierzig", 5: "fünfzig", 6: "sechzig", 7: "siebzig", 8: "achtzig", 9: "neunzig"}


def nw(n, end=False):
    """int -> German words (|n| < 1000); end=True -> 'eins' instead of 'ein'."""
    n = int(n)
    if n < 0: return "minus " + nw(-n, end)
    if n >= 1000: return str(n)
    if n >= 100: return ("ein" if n // 100 == 1 else _U[n // 100]) + "hundert" + (nw(n % 100, end) if n % 100 else "")
    if n < 20: return "eins" if (n == 1 and end) else _U[n]
    return _T[n // 10] if n % 10 == 0 else f"{_U[n % 10]}und{_T[n // 10]}"


def _deg(a, b): return f"{nw(a, True)} bis {nw(b, True)} Grad" if a != b else f"{nw(a, True)} Grad"


def say_wetter(js, wann=""):
    o, d, w = js["ort"], js["tage"], (wann or "").lower()
    def day(i, lbl, wind=True):
        x = d[i]; s = f"{lbl} {_deg(x['min'], x['max'])}, {x['wetter']}"
        if x.get("regen_prob", 0) >= 40 and x["wetter"] not in ("Nieselregen", "Regen", "leichter Regen", "starker Regen"): s += f", Regenwahrscheinlichkeit {nw(x['regen_prob'])} Prozent"
        if wind and x.get("wind_max", 0) >= 5: s += f", Wind bis {nw(x['wind_max'])} Kilometer pro Stunde"
        return s
    pick = []
    for key, i in (("übermorgen", 2), ("uebermorgen", 2), ("morgen", 1), ("heute", 0)):
        if key in w: pick = [(i, key)]; break
    else:
        for i, x in enumerate(d):
            if x["wochentag"].lower() in w: pick = [(i, x["wochentag"])]; break
        else:
            if "wochenende" in w: pick = [(i, x["wochentag"]) for i, x in enumerate(d) if x["wochentag"] in ("Samstag", "Sonntag")]
    if w and not pick and not any(k in w for k in ("jetzt", "gerade", "aktuell")):
        if len(d) < 7 and re.search(r"montag|dienstag|mittwoch|donnerstag|freitag|samstag|sonntag|wochenende", w): return None  # need more days
    if pick:
        return f"In {o} " + ". ".join(day(i, lbl if lbl in ("heute", "morgen", "übermorgen", "uebermorgen") else "am " + lbl) for i, lbl in pick) + "."
    n = js["jetzt"]
    s = f"In {o} sind es gerade {nw(n['temp'], True)} Grad, {n['wetter']}. " + day(0, "Heute") + "."
    if len(d) > 1: s += " " + day(1, "Morgen", False) + "."
    return s


def _ev_title(t): return re.sub(r"\s+", " ", re.sub(r"[^\w\s.,:;!?'/&+\-–äöüÄÖÜß]", "", re.sub(r"\s*\([^)]*\)", "", t))).strip()


def _tm(iso):
    h, m = int(iso[11:13]), int(iso[14:16])
    return f"{nw(h, True)} Uhr" + (f" {nw(m)}" if m else "")


def say_kal(js, which="today", now=None):
    now = now or datetime.now()
    if not isinstance(js, list): return js
    seen, ev = set(), []
    for e in sorted(js, key=lambda e: e["start"]):
        k = (_ev_title(e["title"]).lower(), e["start"][:10])
        if k in seen: continue
        seen.add(k); ev.append(e)
    if not ev: return "Heute steht nichts im Kalender." if which == "today" else "In den nächsten Tagen steht nichts im Kalender."
    def item(e, day=False):
        t = _ev_title(e["title"])
        return t if e.get("allDay") else f"um {_tm(e['start'])} {t}"
    if which == "today":
        s = f"Heute hast du {nw(len(ev), True) if len(ev) > 1 else 'einen'} Termin" + ("e" if len(ev) > 1 else "") + ". "
        return s + ". ".join(item(e) for e in ev[:6]) + ("." if len(ev) <= 6 else f". Dazu noch {nw(len(ev) - 6)} weitere.")
    days = {}
    for e in ev: days.setdefault(e["start"][:10], []).append(e)
    out = []
    for dk, es in list(days.items())[:6]:
        d = datetime.fromisoformat(dk); diff = (d.date() - now.date()).days
        lbl = {0: "Heute", 1: "Morgen"}.get(diff, WD[d.weekday()])
        out.append(f"{lbl}: " + ", ".join(item(e) for e in es[:4]) + (f" und {nw(len(es) - 4)} weitere" if len(es) > 4 else ""))
    return ". ".join(out) + "."


LANG = {"bg": "Bulgarisch", "de": "Deutsch", "en": "Englisch", "fr": "Französisch", "es": "Spanisch", "it": "Italienisch"}
LANGTTS = {"en": "English", "fr": "French", "es": "Spanish", "it": "Italian", "de": "German"}  # LFM3-Audio locales; Bulgarian is NOT supported (garbage) -> transliteration only
_CY = dict(zip("абвгдежзийклмнопрстуфхцчшщъьюя", ["a", "b", "w", "g", "d", "e", "sch", "s", "i", "j", "k", "l", "m", "n", "o", "p", "r", "s", "t", "u", "f", "ch", "z", "tsch", "sch", "schtsch", "a", "", "ju", "ja"]))
translit = lambda w: "".join(_CY.get(c, c) for c in w.lower())


def say_dict(wort, nach="bg"):
    """-> (spoken German sentence, (foreign word, TTS language) | None, IPA | None). Bulgarian: no TTS, German-readable transliteration only."""
    nach = nach.strip().lower()[:2]; nach = nach if nach in LANG else "bg"
    if not wort.strip(): return "Welches Wort soll ich übersetzen?", None, None
    rc, o = run(["dict", wort.strip(), "--to", nach, "--json"]); js = _j(o)
    if not isinstance(js, dict) or not js.get("results"): return f"Für {wort.strip()} habe ich keine Übersetzung gefunden.", None, None
    r = js["results"][0]; w = r["word"]
    if nach in LANGTTS and nach != "de": return f"{wort.strip()} heißt auf {LANG[nach]}:", (w, LANGTTS[nach]), r.get("ipa")
    return f"{wort.strip()} heißt auf {LANG[nach]}: {translit(w) if re.search('[а-я]', w.lower()) else w}.", None, r.get("ipa")


def say_gpu(txt):
    jobs = list(dict.fromkeys(x.strip() for x in re.findall(r"pid \d+ · ([^·]+?) ·", txt)))[:3]
    s = "Die GPU ist frei." if "GPU frei" in txt else "Die GPU ist belegt" + (f", es läuft: {', '.join(jobs)}." if jobs else ".")
    m = re.search(r"RAM frei ~?([\d.,]+) GB", txt); sw = re.search(r"Swap ([\d.,]+) GB", txt)
    if m: s += f" Es sind etwa {nw(round(float(m.group(1).replace(',', '.'))))} Gigabyte Arbeitsspeicher frei"
    if sw: s += f", der Swap liegt bei {nw(round(float(sw.group(1).replace(',', '.'))))} Gigabyte"
    return s.rstrip(".") + "."


def _bignum(m):
    t = m.group(0); v = t.replace(".", "")
    if not v.isdigit(): return t
    n = int(v)
    if n >= 1_000_000: x = n / 1e6; return (f"{x:.1f}".replace(".", ",").rstrip("0").rstrip(",") if x < 100 else str(round(x))) + (" Million" if round(x, 1) == 1.0 else " Millionen")
    if n >= 10000: return f"etwa {round(n / 1000)} tausend"
    return str(n)


def say_fakt(o):
    js = _j(o)
    if isinstance(js, dict) and js.get("text"): txt = js["text"]
    else: return "Dazu habe ich keinen Eintrag."
    txt = re.sub(r"\d{1,3}(?:\.\d{3})+", _bignum, txt)
    txt = re.sub(r"\s*\((?:Stand|Wikidata)[^)]*\)", "", txt)
    return txt.replace(": ", " – ", 1) if txt.count(":") > 1 else txt
