"""Context builder shared by router and brain: state line, running summary, last tool-result handle, profile, self description. Pure functions (unit-tested)."""
import os, re
from datetime import datetime
import tools as T

WD = T.WD
PROFILE = __import__("os").environ.get("KITSUNE_PROFILE", "Nutzer: bevorzugt kurze deutsche Antworten.")
SELF = ("Du bist Kitsune, ein lokaler Sprachassistent auf dem Mac des Nutzers. LFM3-Audio hört und spricht, Gemma denkt; alles läuft offline, nur die Netz-Werkzeuge (Nachrichten, Websuche, Wetter) gehen ins Internet. "
        "Werkzeuge: Kalender, Wetter, Nachrichten, Websuche, Wikipedia, exakte Fakten, Wörterbuch, GPU-Status.")
ASR = ("Das Transkript stammt aus Spracherkennung und kann Fehler enthalten (Namen, Orte, Fachbegriffe). Interpretiere sie phonetisch: "
       "'Gebühr' = GPU, 'Blau tief' = Plowdiw, 'Herr Kensoli-Bach' = Sunny Beach, 'Zunge' = Kitsune, 'heute Morgen' = heute früh, 'morgen' = der nächste Tag.")
STYLE = ("Antworte in höchstens zwei kurzen Sätzen, gesprochene Sprache, keine Liste, kein Markdown, keine Emojis, keine Klammern. "
         "Schreibe Zahlen, Jahre und Daten in Ziffern (z. B. 12. Mai 2024), sie werden automatisch vorlesbar gemacht. Bist du unsicher, sag es kurz und ehrlich. Du duzt den Nutzer.")


def ort(last_place=None): return os.environ.get("KITSUNE_ORT") or last_place or ""


def state_line(now: datetime, place="", mode="active", pend=None):
    s = f"Heute ist {WD[now.weekday()]}, {now:%Y-%m-%d}, {now:%H:%M} Uhr"
    if place: s += f"; Ort des Nutzers: {place}"
    if pend: s += f"; Rückfrage offen ({pend.get('stage')}): {pend.get('ask', '')}"
    return f"[{s}] "


def handle(items):
    """last tool-result list as numbered short titles ('die erste Meldung')."""
    return "Letzte Treffer: " + "; ".join(f"{i}. {t[:70]}" for i, (_, t) in enumerate(items[:5], 1)) if items else ""


def summary(topics, keep=4):
    """1-line running summary of everything older than the last `keep` turns. topics: list of 'tool(arg)' strings, oldest first."""
    old = topics[:-keep] if len(topics) > keep else []
    return "Zuvor besprochen: " + ", ".join(old[-6:]) if old else ""


def topic(tool, args, q):
    a = next((v for v in (args or {}).values() if isinstance(v, str) and v), "")
    return f"{tool}({a[:30]})" if a else tool


def brain_context(now, place, hist, topics, items, n_hist=4):
    """system-side context for the brain: self, profile, state, summary, handle."""
    parts = [SELF, PROFILE, state_line(now, place).strip("[] "), summary(topics, n_hist), handle(items)]
    return "\n".join(p for p in parts if p)


# ---- speakable numbers for TTS (Gemma may still emit digits) ----
def nw_big(n):
    if n < 1000: return T.nw(n)
    if n < 1_000_000:
        h, r = divmod(n, 1000)
        return ("ein" if h == 1 else nw_big(h)) + "tausend" + (T.nw(r) if r else "")
    for div, one, many in ((10**9, "eine Milliarde", "Milliarden"), (10**6, "eine Million", "Millionen")):
        if n >= div:
            v = round(n / div, 1); w = int(v)
            if v == 1: return ("rund " if n != div else "") + one
            return ("rund " if n != w * div else "") + nw_big(w) + (f" Komma {T.nw(round(v * 10) % 10)}" if v != w else "") + " " + many
    return str(n)


def ordw(n, pre=""):
    """ordinal with the ending the preceding word asks for: 'am' -> zweiten, 'der' -> zweite, none -> zweiter."""
    b = {1: "ers", 3: "drit", 7: "sieb", 8: "ach"}.get(n) or (T.nw(n) + ("s" if n >= 20 else ""))
    p = pre.strip().split()[-1].lower() if pre.strip() else ""
    return b + ("ten" if p in _DATW else "te" if p in ("der", "die", "das") else "ter")


MON = "Januar Februar März April Mai Juni Juli August September Oktober November Dezember".split()
_DATW = {"am", "vom", "zum", "im", "beim", "dem", "den", "ab", "bis"}
_DAT = r"(?i:\b(?:am|vom|zum|im|beim|bis zum|seit dem|ab dem|ab|bis|der|die|das|dem|den)\s+)"
_ABBR = [(r"\bz\.\s?B\.", "zum Beispiel"), (r"\bd\.\s?h\.", "das heißt"), (r"\bu\.\s?a\.", "unter anderem"), (r"\bbzw\.", "beziehungsweise"), (r"\busw\.", "und so weiter"),
         (r"\bca\.", "circa"), (r"\bNr\.", "Nummer"), (r"\bArt\.", "Artikel"), (r"\bAbs\.", "Absatz"), (r"\bMio\.", "Millionen"), (r"\bMrd\.", "Milliarden"),
         (r"\bkm/h\b", "Kilometer pro Stunde"), (r"(?<=\d)\s?km\b", " Kilometer"), (r"(?<=\d)\s?m\b(?!/)", " Meter"), (r"(?<=\d)\s?kg\b", " Kilo"), (r"(?<=\d)\s?€", " Euro"), (r"€\s?(\d[\d.,]*)", r"\1 Euro"),
         (r"\bSt\.\s(?=[A-ZÄÖÜ])", "Sankt "), (r"\bDr\.", "Doktor"), (r"\bProf\.", "Professor")]
_UNIT = r"\s*(?:Euro|Dollar|Lewa|Leva|Meter|Kilometer|Kilo|Gramm|Tonnen|Liter|Einwohner|Menschen|Personen|Stück|Grad|Prozent|Mal|Millionen|Milliarden)\b"


def _pre(p): return (p[:-len(p.lstrip())] + "bis zum ") if p.strip().lower() == "bis" else p  # 'bis 31.12.' -> 'bis zum einunddreißigsten Dezember'


def year(n): return (T.nw(n // 100) + "hundert" + (T.nw(n % 100) if n % 100 else "")) if 1100 <= n < 2000 else nw_big(n)


def speakable(s):
    for a, b in _ABBR: s = re.sub(a, b, s)
    s = re.sub(r"(\d+)\s*°\s*C?", lambda m: f"{m.group(1)} Grad", s)
    s = re.sub(r"\b(\d{1,2})[:.](\d{2})\s*Uhr", lambda m: f"{T.nw(int(m.group(1)), True)} Uhr" + (f" {T.nw(int(m.group(2)))}" if int(m.group(2)) else ""), s)
    s = re.sub(rf"({_DAT})?\b([0-3]?\d)\.([01]?\d)\.(\d{{4}}|\d{{2}})?(?!\d)", lambda m: f"{_pre(m.group(1) or '')}{ordw(int(m.group(2)), m.group(1) or '')} {MON[int(m.group(3)) - 1]}" + (f" {year(int(m.group(4)) if len(m.group(4)) == 4 else 2000 + int(m.group(4)))}" if m.group(4) else "")
              if 1 <= int(m.group(2)) <= 31 and 1 <= int(m.group(3)) <= 12 else m.group(), s)  # 2.8.2026 -> zweiter August zweitausendsechsundzwanzig
    s = re.sub(rf"({_DAT})?\b([0-3]?\d)\.\s+(?=(?:{'|'.join(MON)})\b)", lambda m: f"{_pre(m.group(1) or '')}{ordw(int(m.group(2)), m.group(1) or '')} " if 1 <= int(m.group(2)) <= 31 else m.group(), s)  # 'am 2. August'
    s = re.sub(rf"({_DAT})([1-9]\d?)\.\s+(?=[A-ZÄÖÜ][a-zäöüß])", lambda m: f"{m.group(1)}{ordw(int(m.group(2)), m.group(1))} ", s)  # 'im 3. Stock', 'der 2. Platz'
    s = re.sub(r"(\d+(?:,\d+)?)\s*%", r"\1 Prozent", s)
    s = re.sub(r"\b(\d{1,3}(?:\.\d{3})+)\b", lambda m: m.group(1).replace(".", ""), s)
    s = re.sub(r"\b(\d+),(\d+)\b", lambda m: f"{nw_big(int(m.group(1)))} Komma {' '.join(T.nw(int(c)) for c in m.group(2))}", s)
    s = re.sub(rf"\b(1[1-9]\d\d)\b(?!{_UNIT})", lambda m: year(int(m.group())), s)  # 1989 -> neunzehnhundertneunundachtzig unless a quantity
    s = re.sub(r"\b1\b(?=\s*(?:[.!?,;]|$))", "eins", s)
    return re.sub(r"\b\d+\b", lambda m: nw_big(int(m.group())), s)
