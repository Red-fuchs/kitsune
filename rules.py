"""Pure, model-free dialogue rules (unit-tested in test_rules.py)."""
import re, difflib

W = lambda u: re.findall(r"[a-zäöüß]+", u.lower())
YES = {"ja", "jo", "jawohl", "klar", "gerne", "genau", "okay", "ok", "bitte", "passt", "richtig", "stimmt", "mach", "machen", "sicher", "natürlich", "eintragen",
       "bestätige", "bestätigt", "bestätigen", "trag", "tu", "los", "mhm", "yes"}
NO = {"nein", "nee", "nö", "nicht", "abbrechen", "stopp", "stop", "lass", "lassen", "vergiss", "doch", "stornieren", "verwerfen", "abbruch", "no"}
QW = {"wer", "wie", "was", "wo", "wann", "warum", "wieso", "weshalb", "welche", "welcher", "welches", "wohin", "woher", "wieviel", "wieviele", "wem", "wen", "wessen", "womit"}
CMD = {"trag", "lies", "zeig", "zeige", "such", "suche", "sag", "sage", "erzähl", "erzähle", "erkläre", "mach", "öffne", "nenn", "gib", "hilf", "start", "starte"}
STOP = {"stopp", "stop", "halt", "ruhe", "still", "pst", "genug", "danke"}
WX = re.compile(r"wetter|temperatur|\bregen|regnet|schnee|\bsonn(?!tag|abend)|\bwind(?!ow)|\bwarm|\bkalt|\bgrad\b|schauer|gewitter|bewölk|nebel|frost|vorhersage", re.I)  # word starts: 'Geschwindigkeit' is no wind, 'Sonntag' no sun
IGNORE = re.compile(r"nicht an dich|nicht an die gericht|nicht dich gemeint|nicht mit dir|nicht zu dir|nicht gemeint|ignorier|hör (nicht|gar nicht|mal nicht) (zu|hin)|hör weg|red nicht mit dir|rede nicht mit dir|"
                    r"nicht für dich|nicht du gemeint|das gilt nicht dir|vergiss (das|es) (gleich|wieder)|schalt dich weg|nicht angesprochen", re.I)
STATUS = re.compile(r"\b(schon|bereits|eingetragen|gemacht|erledigt|fertig|passiert|geklappt|gespeichert)\b", re.I)


def yes_no(u):
    w = W(u)
    if not w or len(w) > 6: return None
    if STATUS.search(u) and re.search(r"\bhast\b|\bhab\b|\bist\b|\bhaben\b", u.lower()): return None
    if NO & set(w): return False if "doch" not in w or len(NO & set(w)) > 1 else True
    return True if YES & set(w) else None


def is_status_q(u):
    """'hast du's schon gemacht?' / 'hast du den Termin eingetragen?' -> re-ask, don't start a new entry."""
    l = u.lower()
    return bool(STATUS.search(l) and re.search(r"\b(hast|hab|habt|ist|wurde|hat|schon|bereits)\b", l)) and len(W(l)) <= 9


def pending_decide(stage, q):
    """-> 'yes' | 'no' | 'status' | None (=> new request / try to parse as answer)."""
    if is_status_q(q): return "status"
    yn = yes_no(q)
    if stage == "confirm": return None if yn is None else ("yes" if yn else "no")
    return "no" if (yn is False and len(W(q)) <= 4 and not re.search(r"\d|uhr", q.lower())) else None


def false_start(q):
    """barge-in capture that is just noise/fragment ('oder', 'Sport.', 'ja')."""
    w = W(q)
    if not w: return True
    if len(w) > 2 or ("?" in q and len(w) > 1): return False
    if NAME.fullmatch(q): return True
    return not (set(w) & (QW | CMD) or NAME.search(q))


def stop_word(q): return bool(set(W(q)) & STOP) and len(W(q)) <= 2


HARDSTOP = {"stopp", "stop", "halt", "ruhe", "genug", "pst"}
_BSTOP = re.compile(r"\b(hör auf|aufhören|sei still|sei ruhig|warte mal|warte|moment mal|lass das|nicht weiter|das reicht|leise)\b", re.I)


_CONTQ = re.compile(r"^(und|aber|also|okay|ok|ja)[,\s]+(warum|wieso|weshalb|wie|was|wer|wo|wann|welche\w*)\b", re.I)


def barge_contq(q):
    """'Und warum ist der?' while Kitsune explains: a follow-up question on its topic (TV/third parties rarely open like this); 'Schatz'-style vocatives excluded."""
    return bool(_CONTQ.match(q.strip())) and not re.search(r"\b(schatz|mama|papa|liebling)\b", q, re.I)


def barge_no(q):
    """short capture while Kitsune talks that starts a 'no/stop' ('Okay, nein.', 'Nee, warte'): the verify window often holds only the first 0.6-1 s of 'Okay, nein, nein, stopp'."""
    w = W(q)
    return 0 < len(w) <= 4 and bool(set(w) & ({"nein", "nee", "nö", "warte", "moment"} | HARDSTOP))


def barge_stop(q):
    """interruption meant as 'stop talking' even when longer than a bare stop word: 'Okay, nein, nein, nein, stopp.', 'Warte mal', 'hör auf'."""
    w = W(q)
    if not w or len(w) > 8: return False
    return bool(set(w) & HARDSTOP) or bool(_BSTOP.search(q)) or w.count("nein") >= 2


def self_echo(q, said):
    """barge-in text that is (a fragment of) what Kitsune itself just said -> speaker echo, never user speech. <=2 words: all of them occur in the answer; longer: >=75 % of the words
    sit in in-order runs of >=2 words from the answer. A real stop word that the answer does not contain always passes."""
    qw, aw = W(q), W(said)
    if not qw or not aw: return False
    if (set(qw) & HARDSTOP) - set(aw): return False
    if len(qw) <= 2: return all(w in set(aw) for w in qw)
    blocks = difflib.SequenceMatcher(None, qw, aw, autojunk=False).get_matching_blocks()
    return sum(b.size for b in blocks if b.size >= 2) / len(qw) >= 0.75


def not_addressed(q): return bool(IGNORE.search(q))


def wakes(q):
    """while ignoring: only the name or a clear question wakes Kitsune."""
    return bool(NAME.search(q)) or ("?" in q and bool(W(q)) and W(q)[0] in QW)


def echo(ans, q):
    a, b = W(ans), W(q)
    if len(a) < 3 or not b: return False
    ov = len(set(a) & set(b)) / len(set(a))
    return ov >= 0.6 or difflib.SequenceMatcher(None, " ".join(a), " ".join(b)).ratio() >= 0.6


# ---- fuzzy wake name (phonetic key + edit distance; ASR variants: Kizune, Kizude, Kitsume, Kissune, Isune, Zune ...) ----
_PH = [("sch", "s"), ("tsch", "s"), ("ck", "k"), ("ph", "f"), ("tz", "s"), ("ts", "s"), ("z", "s"), ("c", "k"), ("q", "k"), ("v", "f"), ("w", "f"),
       ("ie", "i"), ("y", "i"), ("ä", "e"), ("ö", "o"), ("ü", "u"), ("é", "e"), ("ß", "s"), ("d", "t"), ("b", "p"), ("g", "k"), ("h", "")]


def phon(w):
    w = w.lower()
    for a, b in _PH: w = w.replace(a, b)
    return re.sub(r"(.)\1+", r"\1", w)


def _lev(a, b):
    d = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        p, d[0] = d[0], i
        for j in range(1, len(b) + 1): p, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, p + (a[i - 1] != b[j - 1]))
    return d[-1]


NAME_PH = phon("kitsune")  # kisune
GREET = {"hey", "hallo", "hi", "ok", "okay", "ey", "he", "na", "also", "und", "so", "nochmal", "mal", "gut", "danke", "ja", "nein", "alles", "klar"}


_SHAPE = re.compile(r"k?[ie]?su[nmtl][eiau]?")
ALIAS = {"zunge", "kitzune"}


def name_at(q, first=6):
    """-> (index into W(q), distance ratio, n words) of the best name-like word, or None. Name-shaped words ('kisune', 'sune', 'kisute') count within the first `first` words,
    else only one phonetic typo (ratio <=.17). Also tries 2-word splits ('kit sune', 'ki zune')."""
    ws = W(q); best = None
    cands = [(i, ws[i], 1) for i in range(len(ws))] + [(i, ws[i] + ws[i + 1], 2) for i in range(len(ws) - 1) if ws[i][:1] == "k" and len(ws[i]) <= 3]
    for i, w, n in cands:
        p = phon(w)
        if w in ALIAS and i < 2: best = (i, 0.2, 1); continue
        if len(p) < 4: continue
        r = _lev(p, NAME_PH) / max(len(p), len(NAME_PH))
        if _SHAPE.fullmatch(p) and abs(len(p) - len(NAME_PH)) <= 2 and i < first: r = min(r, 0.2)
        if r <= 0.17 or (r <= 0.34 and r == min(r, 0.2) and i < first and _SHAPE.fullmatch(p)):
            if best is None or r < best[1]: best = (i, r, n)
    return best


class _Name:
    def search(self, q): return name_at(q) is not None
    def fullmatch(self, q): return only_name(q)
NAME = _Name()


def strip_name(q):
    """q without the name token (and a greeting before it), original casing/punctuation kept; '' if only the name/greeting remains."""
    b = name_at(q)
    if not b: return q.strip()
    toks = list(re.finditer(r"[A-Za-zÄÖÜäöüßéÉ]+", q)); i, n = b[0], b[2]
    drop = {j for j in range(i, i + n)} | {j for j in range(i) if toks[j].group().lower() in GREET}
    keep = [t for j, t in enumerate(toks) if j not in drop]
    if not keep: return ""
    out = q[keep[0].start():].strip(" ,.;:!-")
    # remove the dropped middle part (name between kept tokens)
    if i < len(toks) and keep[0].start() < toks[i].start():
        out = (q[keep[0].start():toks[i].start()] + q[toks[i + n - 1].end():]).strip(" ,.;:-")
    out = re.sub(r"\s+([?.!,])", r"\1", out); out = re.sub(r",([?.!])", r"\1", out)
    return out[:1].upper() + out[1:]


END_RE = re.compile(r"\b(pause|schluss|ende|feierabend|still|leg dich schlafen|schlaf)\b|das war'?s|das wäre alles|das war alles|mehr brauch(e)? ich nicht|tschüss|auf wiedersehen", re.I)


def ends_conversation(q):
    """'Kitsune, Pause' / 'Schluss' / 'danke, das war's' -> back to standby."""
    l = q.lower()
    if len(W(l)) > 8: return False
    return bool(END_RE.search(l)) and (name_at(q) is not None or bool(re.search(r"das war'?s|das wäre alles|das war alles|mehr brauch|schluss|tschüss|auf wiedersehen", l)))


def only_name(q):
    """wake with just the name ('Kitsune?', 'Hey Kitsune')."""
    return name_at(q) is not None and not strip_name(q)


ACKS = {"ja", "nein", "okay", "ok", "genau", "bitte", "hallo", "tschüss", "gerne", "klar", "richtig", "stimmt"}


def gate_hard(q, *, in_window, pending, ignoring):
    """-> 'pass' | 'drop' | None (ask the addressee model). Pure."""
    ws = W(q)
    if NAME.search(q) or pending or stop_word(q) or not_addressed(q): return "pass"
    if ignoring: return "pass"  # the engine's ignoring branch stays silent unless name/clear question
    if len(ws) < 3 and not (set(ws) & (QW | CMD | ACKS)) and "?" not in q: return "drop"  # 'Sport.', 'oder' ...
    if in_window and (len(ws) <= 5 or is_question_or_cmd(q)): return "pass"  # short follow-ups and questions/commands; long declaratives go to the addressee model
    return None


QCMD = QW | CMD | {"kannst", "könntest", "würdest", "magst", "sag", "zeig", "ist", "hast", "gibt", "kann", "darf", "soll"}


def is_question_or_cmd(q): return "?" in q or bool(set(W(q)) & QCMD)


# ---- endpointing: does the transcript end mid-sentence? (closed German word lists) ----
_PREP = set("in an auf mit von zu bei nach für über unter vor zwischen gegen durch ohne um bis seit aus am im zum zur vom beim ins ans".split())
_ART = set("der die das den dem des eine einen einem einer eines mein meine meinen dein deine".split())
_CONJ = set("und oder aber dass weil wenn ob als denn sondern also dann äh ähm ähm äähm".split())
_NUM = set("null eins zwei drei vier fünf sechs sieben acht neun zehn elf zwölf dreizehn vierzehn fünfzehn zwanzig dreißig vierzig fünfzig halb viertel dreiviertel".split())
_FRAG = re.compile(r"(\bum|\bam|\bgegen|\bbis|\bvon|\bab)\s+(halb\s+)?(\d{1,2}([:.]\d\d)?|\w+)$")


_SEP = {"an", "auf", "aus", "ab"}


def incomplete(q):
    """transcript likely cut mid-sentence (ends on preposition/article/conjunction/number/'halb'/time fragment) -> endpointing waits longer (1.2 s)."""
    ws = W(q)
    if not ws: return False
    last = ws[-1]
    if last in _SEP and len(ws) >= 3 and re.search(r"[.?!]\s*$", q): return False  # 'Was steht heute an?' 'Mach das Licht an.': separable verb particle, not a cut-off preposition
    if last in _PREP or last in _ART or last in _CONJ or last in _NUM or last.isdigit(): return True
    if re.search(r"\d\s*$", q.strip(" .,!?")) and "uhr" not in last: return True
    return False


_TITLE_NOISE = set("trag trage mach mache setz setze schreib schreibe lege leg erstelle füge ein hinzu termin kalender in den dem einen einem mir mich bitte am um uhr halb viertel dreiviertel morgen heute übermorgen vormittag nachmittag abend früh morgens abends nachmittags "
                   "montag dienstag mittwoch donnerstag freitag samstag sonntag wochenende nächste nächsten kommenden kommende diese diesen woche und oder auch noch mal einmal für ganzen ganze tag den der die das".split()) | _NUM


def title_missing(titel, utter=""):
    """True if no real title is left once command verb / date / time words are removed ('Trag morgen um halb drei')."""
    ws = [w for w in W(titel or "") if w not in _TITLE_NOISE and not w.isdigit()]
    return not ws


def starts_directed(q):
    """question/command shape at the START of the utterance (or '?'): the only thing that earns an 'Hm?' for a middling addressee score."""
    ws = W(q)
    return bool(ws) and ("?" in q and len(ws) <= 12 or ws[0] in QCMD or (len(ws) > 1 and ws[1] in QCMD and ws[0] in ("und", "aber", "also", "sag", "mal", "hey")))


_FR_KAL = re.compile(r"^(und )?(was steht heute (noch )?(an|im kalender|auf dem plan)|welche termine (habe|hab) ich heute|was habe ich heute (für termine|vor|an termine)|habe ich heute (noch )?termine|was ist heute (los|geplant))\W*$", re.I)
_FR_GPU = re.compile(r"\bgpu\b", re.I)
_FR_GPU2 = re.compile(r"läuft|auslastung|ausgelastet|frei|belegt|status|beschäftigt", re.I)
_FR_FAKT = re.compile(r"^(und )?(wie viele einwohner (hat|hatte|leben in)|was ist die hauptstadt von|wie hoch ist der|wie hoch ist die|wie groß ist die fläche von) \S.*\W*$", re.I)


_WEB = re.compile(r"\bim (web|internet|netz|www)\b|\bonline\b|\bgoogl?e\w*|\bgoogel\w*", re.I)
_WEB_FILL = set("ich du ist bist kannst könntest kann würdest bitte mal für mich mir im web internet netz www online google googel googeln googlen suche suchen such nach nachschauen nachsehen schau schaue schauen "
                "laut nachschlagen recherchiere recherchieren recherchier finde finden doch eben einmal über und dass ob zwar nämlich".split())
CONT_RE = re.compile(r"^(und zwar|zwar|nämlich|ich meine|ich mein|genauer|genauer gesagt|und noch|außerdem)\b", re.I)  # explicit continuation openers only (TV chatter starts with 'also'/'dann')


def is_continuation(q): return bool(CONT_RE.match(q.strip()))


_DAY = r"heute|morgen|übermorgen|montag|dienstag|mittwoch|donnerstag|freitag|samstag|sonntag|wochenende"
_ELL = re.compile(rf"^(?:(?:und|aber|bitte|dann|okay|ok|ja|nein|eigentlich|lieber|doch)[\s,]+)*(?P<m>was ist mit\s+|wie ist es\s+|wie sieht'?s\s+|wie sieht es\s+|wie wird es\s+|wie wird's\s+)?(?:(?:am|zum|für)\s+(?=(?:{_DAY})\b))?(?P<w>{_DAY})?[\s,]*(?P<p>(?:in|für|bei|nach|den|dem|im)\s+)?(?P<o>(?!(?:{_DAY}|aus|bitte)\b)[a-zäöü][\wäöüß-]+(?:\s+(?!(?:{_DAY}|aus|bitte)\b)[a-zäöü][\wäöüß-]+)?)?[\s,]*(?:(?:am|zum|für)\s+)?(?P<w2>{_DAY})?(?:\s+(?:bitte|aus))?[\s.?!]*$", re.I)


def ellipsis(q):
    """'und in Basel?', 'Bitte in Berlin.', 'und morgen?', 'und übermorgen in Sofia?' -> {'ort': .., 'wann': ..} (only slots said); None otherwise. Caller restricts it to right after a weather turn."""
    if len(W(q)) > 6 or not (mt := _ELL.match(q.strip())): return None
    o, w = mt.group("o"), mt.group("w") or mt.group("w2")
    if o and not (o[0].isupper() and (mt.group("p") or (mt.group("m") or "").lower().startswith("was ist mit"))): return None  # a place needs a capital + preposition ('und dann?' is no place)
    if o and o.split()[0].lower() in QW | {"der", "die", "das", "dem", "den", "ein", "einem", "einer", "mir", "dir", "uns", "meinem", "meiner"}: return None
    return {k: v for k, v in (("ort", o), ("wann", w and w.lower())) if v} or None


_ORT = re.compile(r"\b(?:in|für|bei|nach)\s+([A-ZÄÖÜ][\wäöüß-]+(?:\s+[A-ZÄÖÜ][\wäöüß-]+)?)")


def grounded(arg, q):
    """router argument actually said? every 3+ letter word of arg starts (first 4 letters) somewhere in the utterance; tolerant of ASR/inflection ('Plowdiw' vs 'Plowdiws')."""
    qa = re.findall(r"[a-zäöüß]+", q.lower()); w = [x for x in re.findall(r"[a-zäöüß]+", arg.lower()) if len(x) >= 3]
    return bool(w) and all(any(y.startswith(x[:4]) for y in qa) for x in w)


def wx_slots(q):
    """full weather question -> {'ort': .., 'wann': ..} by rule ('' ort = default place / ask); None when the sentence names no weather."""
    if not WX.search(q): return None
    w = re.search(rf"\b({_DAY})\b", q, re.I); o = None
    for mt in _ORT.finditer(q):
        c = mt.group(1).split()
        while c and (c[-1].lower() in _DAY.split("|") or c[-1].lower() in {"aus", "bitte"}): c.pop()
        if c and c[0].lower() not in set(_DAY.split("|")) | {"der", "die", "das", "dem", "den"}: o = " ".join(c)
    return {"ort": o or "", "wann": w.group(1).lower() if w else "heute"}


_TEMPO = re.compile(r"\b(sprich|sprechen|rede|reden|red)\b.*\b(schneller|langsamer|normal|zügiger|flotter|ruhiger)\b|\b(schneller|langsamer|zügiger|flotter)\s+(sprechen|reden|bitte)\b|\bnicht so (schnell|langsam)\b", re.I)


def tempo_cmd(q):
    """'Sprich bitte schneller.' / 'Kannst du langsamer reden?' / 'nicht so schnell' -> faster|slower|normal; None otherwise (the user means the assistant's speech rate)."""
    if len(W(q)) > 10 or not (mt := _TEMPO.search(q)): return None
    t = q.lower()
    if "nicht so schnell" in t: return "slower"
    if "nicht so langsam" in t: return "faster"
    if re.search(r"normal", t): return "normal"
    return "faster" if re.search(r"schneller|zügiger|flotter", t) else "slower"


def strip_cont(q): return re.sub(r"^(und zwar|zwar|nämlich|ich meine|ich mein|genauer gesagt|genauer)\b[\s,:]*", "", q.strip(), flags=re.I).strip(" .,!?")


def web_query(q):
    """'... im Web für mich nachschauen, ist Kim' -> 'Kim'; None if no web phrase; '' if the phrase has no topic."""
    if not _WEB.search(q) or WX.search(q) or re.search(r"termin|kalender", q, re.I): return None
    if name_at(q): q = strip_name(q) or q   # 'Kitsune, such im Internet nach X': the assistant's name is not part of the topic
    t = re.findall(r"[\wäöüÄÖÜß]+", _WEB.sub(" ", q))
    while t and t[0].lower() in _WEB_FILL: t.pop(0)
    while t and t[-1].lower() in _WEB_FILL: t.pop()
    return " ".join(t)


def fast_route(q):
    """deterministic route for a few unambiguous, high-frequency phrasings (skips the 0.3-0.6 s router); None -> use the router."""
    q = q.strip()
    if name_at(q) and (q2 := strip_name(q)) and len(W(q2)) >= 3: q = q2  # 'Kitsune, was steht heute an?' in an open conversation: the name is not part of the request
    if len(W(q)) <= 16 and (wq := web_query(q)) is not None: return ("netz_suche", {"frage": wq}) if wq else ("netz_ask", {})
    if len(W(q)) > 9: return None
    if _FR_KAL.match(q): return "kal_heute", {}
    if _FR_GPU.search(q) and _FR_GPU2.search(q) and not WX.search(q) and not re.search(r"termin|kalender", q, re.I): return "gpu_status", {}
    if _FR_FAKT.match(q): return "fakt", {"frage": q.rstrip(" .?!")}
    return None


# ---- chat default: statements / self-directed questions must not reach a tool ----
_CHATTY = {"ich", "mein", "meine", "meinen", "meinem", "meiner", "du", "wir", "mir", "mich", "es", "man", "ach", "oh", "naja", "tja", "hm", "hmm", "uff", "boah", "leider", "eigentlich", "gestern", "heute", "vorhin", "jetzt"}
_IMP = re.compile(r"^(und |dann |okay |ok |also |bitte |kitsune )*(erklär|erzähl|sag|zeig|nenn|gib|such|lies|schlag|schau|check|find|berechne|rechne|übersetz|beschreib|vergleich|fass|trag|mach|setz|leg|öffne|start|hilf)", re.I)
_CAL = re.compile(r"termin|kalender|steht|\bplan\b|\bvor\b|vorhaben|woche|anstehen|ansteht|erinner|geplant|ist los", re.I)
_SELFQ = re.compile(r"^(und |aber |sag |hey |okay )*(hast|bist|magst|fühlst|träumst|schläfst|liebst|hasst|isst|trinkst|findest|langweilst|vermisst|freust|hörst|siehst|lebst|brauchst) du\b", re.I)
_ENTRY = re.compile(r"termin|eintrag|trag|uhr|halb|viertel|\d{1,2}[:.]\d\d|erinner|block|vormerk|notier", re.I)
_DEMOTABLE = {"kal_heute", "kal_woche", "kal_eintragen", "wiki_suche", "wissen", "wetter"}
_PRON = {"ich", "mir", "mich", "mein", "meine", "meinen", "meinem", "du", "wir", "es", "man", "ach", "oh", "naja", "tja", "hm", "uff", "boah"}


def chat_demote(name, q, pending=False):
    """router tool choice -> 'plaudern' | 'wissen' | None (keep). Statements (chatty start, no question/command shape), self-directed questions ('Hast du ...?')
    and calendar picks without any calendar cue are not tool requests. Never touches question-shaped fact queries (they stay non-chat: calendar picks without cue go to Gemma 'wissen')."""
    if name not in _DEMOTABLE or pending: return None
    ws = W(q)
    if not ws: return None
    if name == "wetter":  # 'Ich glaube, es regnet bald, mir tut der Kopf weh.' (statement, no weather word) is small talk; real weather requests keep the tool
        return "plaudern" if ws[0] in _PRON and "?" not in q and not re.search(r"wetter|temperatur|vorhersage|grad\b", q, re.I) and not (set(ws) & (QW | CMD)) else None
    if WX.search(q): return None
    qshape = "?" in q or ws[0] in QCMD or (len(ws) > 1 and ws[0] in ("und", "aber", "also", "dann", "okay", "ok", "hey") and ws[1] in QCMD) or bool(_IMP.match(q.strip()))
    if _SELFQ.match(q.strip()) and not re.search(r"termin|kalender|wetter|erklär|weißt|wissen|nachschau|nachsehen|suche|such", q, re.I): return "plaudern"
    entry = bool(_ENTRY.search(q)) or (ws[0] not in _CHATTY and bool(re.search(r"\bum (halb |viertel |dreiviertel )?(\d+|" + "|".join(_NUM) + r")\b", q.lower())))
    if name.startswith("kal_") and not _CAL.search(q) and not (name == "kal_eintragen" and entry):
        if re.search(r"datum|spät|uhrzeit|wievielte|welcher tag|wochentag", q, re.I): return "wissen"  # date/time questions are not calendar reads (Gemma has the date in its state line)
        if re.search(r"\b(du|dir|dich|dein\w*)\b", q, re.I) or not qshape: return "plaudern"
        return None  # ASR-garbled calendar questions ('Was geht heute an?') keep the router's pick
    if name == "kal_eintragen" and not entry and not qshape: return "plaudern"
    if not qshape and ws[0] in _CHATTY: return "plaudern"
    return None
