"""German relative date/time resolver for ASR fragments (stdlib only)."""
import calendar
import datetime as dt
import re
from dataclasses import dataclass


@dataclass
class Result:
    date: dt.date | None = None
    time: dt.time | None = None
    allday: bool = False
    ok: bool = False
    note: str = ""


# ---- vocab (umlauts are normalised to ae/oe/ue/ss before lookup) ----
UNITS = ("null ein zwei drei vier fuenf sechs sieben acht neun zehn elf zwoelf dreizehn vierzehn "
         "fuenfzehn sechzehn siebzehn achtzehn neunzehn").split()
TENS = {2: "zwanzig", 3: "dreissig", 4: "vierzig", 5: "fuenfzig", 6: "sechzig"}
CARD = {w: i for i, w in enumerate(UNITS)}
CARD.update(eins=1, eine=1, einen=1, einem=1, einer=1, zwo=2)
for _t, _w in TENS.items():
    CARD[_w] = _t * 10
    for _u in range(1, 10):
        if _t < 6:
            CARD[f"{UNITS[_u]}und{_w}"] = _t * 10 + _u


def _cardname(n):
    return UNITS[n] if n < 20 else TENS[n // 10] if n % 10 == 0 else f"{UNITS[n % 10]}und{TENS[n // 10]}"


ORD = {}
for _n in range(1, 32):
    ORD[{1: "erst", 3: "dritt", 7: "siebt", 8: "acht"}.get(_n) or _cardname(_n) + ("t" if _n < 20 else "st")] = _n
ORD["siebent"] = 7


def ordinal(w):
    for suf in ("en", "er", "es", "em", "e", ""):
        if w.endswith(suf) and (st := w[:len(w) - len(suf)]) in ORD:
            if st == "acht" and not suf:
                continue  # plain "acht" is the cardinal
            return ORD[st]


MONTH = {m: i + 1 for i, m in enumerate(
    "januar februar maerz april mai juni juli august september oktober november dezember".split())}
MONTH.update(jaenner=1, marz=3, sept=9)
WD = dict(montag=0, dienstag=1, mittwoch=2, donnerstag=3, freitag=4, samstag=5, sonnabend=5, sonntag=6)
DPW = {"morgens": "frueh", "frueh": "frueh", "vormittag": "vormittag", "vormittags": "vormittag",
       "mittag": "mittag", "mittags": "mittag", "nachmittag": "nachmittag", "nachmittags": "nachmittag",
       "abend": "abend", "abends": "abend", "nacht": "nacht", "nachts": "nacht"}
DEFT = dict(frueh=(8, 0), vormittag=(10, 0), mittag=(12, 0), nachmittag=(15, 0), abend=(19, 0), nacht=(22, 0))
UNITP = {"stunde": "h", "minute": "min", "tag": "d", "woche": "w", "monat": "mo"}
DAYREF = {"heute": 0, "uebermorgen": 2, "vorgestern": -2, "gestern": -1}

WDN = "Montag Dienstag Mittwoch Donnerstag Freitag Samstag Sonntag".split()
MON = ("Januar Februar März April Mai Juni Juli August September Oktober November Dezember").split()
HW = "zwölf eins zwei drei vier fünf sechs sieben acht neun zehn elf".split()


def norm(s):
    s = s.lower()
    for a, b in (("ä", "ae"), ("ö", "oe"), ("ü", "ue"), ("ß", "ss")):
        s = s.replace(a, b)
    return s


def tokens(s):
    """-> list of ('w',word) ('n',int,ex24) ('o',int) ('d',d,m,y) ('t',h,m,ex24)"""
    s = norm(s)
    s = re.sub(r"(\d{4})-(\d\d)-(\d\d)", lambda m: f" @d{int(m[3])}_{int(m[2])}_{m[1]} ", s)
    s = re.sub(r"(?<!\d)(\d{1,2})\.(\d{1,2})\.(\d{4}|\d\d(?!\d))?",
               lambda m: f" @d{int(m[1])}_{int(m[2])}_{m[3] or 0} ", s)
    tm = lambda m: f" @t{int(m[1])}_{int(m[2])}_{int(len(m[1]) == 2 and m[1][0] == '0')} "
    s = re.sub(r"(?<!\d)(\d{1,2}):(\d\d)(?!\d)", tm, s)
    s = re.sub(r"(?<!\d)(\d{1,2})\.(\d\d)(?=\s*uhr)", tm, s)
    s = re.sub(r"\bin (?:einer|ner) halben stunde\b", "in 30 minuten", s)
    s = re.sub(r"\banderthalb stunden?\b", "90 minuten", s)
    s = re.sub(r"\b(" + "|".join(WD) + r")(vormittag|nachmittag|mittag|morgen|abend|nacht)s?\b", r"\1 \2", s)
    out = []
    for t in re.findall(r"@[dt][\d_]+|\d+\.?|[a-z]+", s):
        pw = out[-1][1] if out and out[-1][0] == "w" else None
        if t[0] == "@":
            v = [int(x) for x in t[2:].split("_")]
            out.append((t[1], *v))
        elif t[0].isdigit():
            ex = int(len(t.rstrip(".")) == 2 and t[0] == "0")
            out.append(("o", int(t[:-1])) if t.endswith(".") and pw != "um" else ("n", int(t.rstrip(".")), ex))
        elif t in CARD:
            out.append(("n", CARD[t], 0))
        elif (o := ordinal(t)):
            out.append(("o", o))
        else:
            out.append(("w", t))
    return out


def disamb(h, dp, ex):
    """12h -> 24h guess."""
    if ex or h >= 13 or h == 0:
        return h
    if dp in ("frueh", "vormittag"):
        return h
    if dp == "nacht":
        return h + 12 if h >= 6 and h != 12 else 0 if h == 12 else h
    if dp in ("abend", "nachmittag"):
        return h + 12 if h < 12 else h
    if dp == "mittag":
        return h + 12 if h <= 5 else h
    return h + 12 if h <= 6 else h


def add_months(d, n):
    m = d.month - 1 + n
    y, m = d.year + m // 12, m % 12 + 1
    return dt.date(y, m, min(d.day, calendar.monthrange(y, m)[1]))


def next_dom(dd, today):
    y, m = today.year, today.month
    for _ in range(14):
        if dd <= calendar.monthrange(y, m)[1] and dt.date(y, m, dd) >= today:
            return dt.date(y, m, dd)
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)


def next_md(dd, mm, today):
    for y in range(today.year, today.year + 9):
        try:
            if (d := dt.date(y, mm, dd)) >= today:
                return d
        except ValueError:
            pass


def resolve(phrase: str, now: dt.datetime) -> Result:
    T = tokens(phrase)
    W = lambda j: T[j][1] if 0 <= j < len(T) and T[j][0] == "w" else None
    N = lambda j: T[j] if 0 <= j < len(T) and T[j][0] == "n" else None
    today = now.date()
    day = wd = cal = dom = rel = hm = dp = None
    wmode, ex, allday, mid, nextweek = "", False, False, False, False

    def fin(t, H, e, s):
        h, m = divmod(t, 60)
        if h == 0 and 1 <= H <= 12:
            h = 12
        return h % 24, m, e, s

    def clock(i):
        k = T[i]
        if k[0] == "n":
            nx = W(i + 1)
            if nx == "uhr":
                n2 = N(i + 2)
                if n2 and n2[1] < 60 and not (W(i + 3) or "").startswith(tuple(UNITP)):
                    return k[1], n2[1], k[2], i + 3
                return k[1], 0, k[2], i + 2
            if nx in ("nach", "vor"):
                j = i + 2
                half = W(j) == "halb"
                j += half
                h = N(j)
                if h and h[1] <= 24 and k[1] < 60:
                    base = h[1] * 60 - 30 * half
                    return fin(base + (k[1] if nx == "nach" else -k[1]), h[1], h[2], j + 1)
            return None
        w = W(i)
        if w == "halb" and (h := N(i + 1)):
            return fin(h[1] * 60 - 30, h[1], h[2], i + 2)
        if w == "dreiviertel" and (h := N(i + 1)):
            return fin(h[1] * 60 - 15, h[1], h[2], i + 2)
        if w == "viertel":
            if W(i + 1) in ("nach", "vor") and (h := N(i + 2)):
                return fin(h[1] * 60 + (15 if W(i + 1) == "nach" else -15), h[1], h[2], i + 3)
            if h := N(i + 1):
                return fin(h[1] * 60 - 45, h[1], h[2], i + 2)

    def month_date(dd, j):
        y = 0
        if (yy := N(j + 1)) and yy[1] >= 1000:
            y = yy[1]
        return (dd, MONTH[W(j)], y), 2 + bool(y)

    i = 0
    while i < len(T):
        k, w, step = T[i], W(i), 1
        if k[0] == "d":
            cal = k[1:]
        elif k[0] == "t":
            hm, ex = (k[1], k[2]), bool(k[3])
        elif k[0] in "on" and W(i + 1) in MONTH:
            cal, step = month_date(k[1], i + 1)
        elif k[0] == "o":
            if W(i - 1) in ("am", "den", "zum", "dem", "vom", "bis"):
                dom = k[1]
        elif k[0] == "n":
            if c := clock(i):
                hm, ex, step = (c[0], c[1]), bool(c[2]), c[3] - i
            elif W(i - 1) == "um" and not (W(i + 1) or "").startswith(tuple(UNITP)):
                hm, ex = (k[1], 0), bool(k[2])
        elif w in ("halb", "viertel", "dreiviertel"):
            if c := clock(i):
                hm, ex, step = (c[0], c[1]), bool(c[2]), c[3] - i
        elif w == "in" and N(i + 1) and (u := next((v for p, v in UNITP.items() if (W(i + 2) or "").startswith(p)), None)):
            rel, step = (N(i + 1)[1], u), 3
        elif w in DAYREF:
            day = DAYREF[w]
        elif w == "morgen":
            if day is not None or wd is not None or cal or dom:
                dp = "frueh"
            else:
                day = 1
        elif w in DPW:
            dp = DPW[w]
        elif w == "mitternacht":
            mid = True
        elif w in ("ganztaegig", "ganztags") or (w == "ganzen" and W(i + 1) == "tag"):
            allday = True
        elif w in WD:
            wd = WD[w]
            pw = W(i - 1) or ""
            wmode = ("over" if pw.startswith("uebernaechst") else
                     "next" if pw.startswith(("naechst", "kommend")) else "bare")
        elif w == "woche" and (W(i - 1) or "").startswith("naechst"):
            nextweek = True
        i += step

    # ---- time ----
    t, carry = None, 0
    if allday:
        hm = None
    elif hm:
        h, m = hm
        if h == 24 and m == 0:
            h, carry = 0, 1
        elif h < 24 and m < 60:
            t = dt.time(disamb(h, dp, ex), m)
    elif mid:
        t, carry = dt.time(0, 0), 1
    elif dp:
        t = dt.time(*DEFT[dp])
    if mid and t is None:
        t, carry = dt.time(0, 0), 1
    if carry and t is None:
        t = dt.time(0, 0)

    # ---- relative hours / minutes ----
    if rel and rel[1] in ("h", "min") and not allday:
        t0 = (now + dt.timedelta(**{"hours" if rel[1] == "h" else "minutes": rel[0]})).replace(second=0, microsecond=0)
        return _done(Result(t0.date(), t0.time(), False, True), now, "")

    note, d, explicit = "", None, True
    if not (cal or dom or day is not None or wd is not None or rel or t or allday or nextweek or mid):
        return Result(note="nichts erkannt")
    try:
        if cal:
            dd, mm, yy = cal
            yy = yy + 2000 if 0 < yy < 100 else yy
            d = dt.date(yy, mm, dd) if yy else next_md(dd, mm, today)
            if d is None:
                raise ValueError
        elif dom:
            d = next_dom(dom, today)
            if d is None:
                raise ValueError
        elif day is not None:
            d = today + dt.timedelta(day)
        elif wd is not None:
            if nextweek:
                d = today - dt.timedelta(today.weekday()) + dt.timedelta(7 + wd)
            else:
                shift = 7 * rel[0] if rel and rel[1] == "w" else 0
                base = today + dt.timedelta(shift)
                delta = (wd - base.weekday()) % 7
                if wmode == "bare" and not shift:
                    tt = dt.time(23, 59) if carry else t
                    if delta == 0 and not (tt and tt > now.time()):
                        delta = 7
                elif wmode in ("next", "over") and delta == 0:
                    delta = 7
                d = base + dt.timedelta(delta + (7 if wmode == "over" else 0))
        elif rel:
            n, u = rel
            d = today + dt.timedelta(n * (7 if u == "w" else 1)) if u != "mo" else add_months(today, n)
        elif nextweek:
            d = today + dt.timedelta(7)
        else:
            explicit = False
            d = today
            if not carry and dt.datetime.combine(d, t) <= now:
                d += dt.timedelta(1)
                note = "Uhrzeit schon vorbei, nächster Tag angenommen"
    except ValueError:
        return Result(note="ungültiges Datum")
    d += dt.timedelta(carry)
    return _done(Result(d, t, allday, True), now, note)


def _done(r, now, note):
    notes = [note] if note else []
    if r.time is None and not r.allday:
        notes.append("keine Uhrzeit")
    if (dt.datetime.combine(r.date, r.time) < now) if r.time else (r.date < now.date()):
        notes.append("liegt in der Vergangenheit")
    r.note = "; ".join(notes)
    return r


def speak(res: Result, now: dt.datetime) -> str:
    if not res.ok or res.date is None:
        return "Das habe ich nicht verstanden"
    d, diff = res.date, (res.date - now.date()).days
    dstr = f"{WDN[d.weekday()]} den {d.day}. {MON[d.month - 1]}" + (f" {d.year}" if d.year != now.year else "")
    lead = {0: "heute", 1: "morgen, ", 2: "übermorgen, ", -1: "gestern, "}.get(diff, "") + ("" if diff == 0 else dstr)
    tp = ""
    if res.allday:
        tp = "ganztägig"
    elif (t := res.time):
        h, m = t.hour, t.minute
        if h == 0 and m == 0:
            tp = "um Mitternacht"
        elif diff == 0 and m == 30:
            tp = f"um halb {HW[(h + 1) % 12]}"
        elif diff == 0 and m == 15:
            tp = f"um viertel nach {HW[h % 12]}"
        elif diff == 0 and m == 45:
            tp = f"um viertel vor {HW[(h + 1) % 12]}"
        else:
            tp = f"um {h} Uhr" + (f" {m}" if m else "")
    if diff == 0:
        return f"{lead} {tp}" if tp and not res.allday else f"{lead}, {tp}" if tp else lead
    return f"{lead}, {tp}" if tp else lead


def kal_at(res: Result) -> str:
    if not res.ok or res.date is None:
        return ""
    if res.allday or res.time is None:
        return res.date.isoformat()
    return f"{res.date.isoformat()} {res.time:%H:%M}"
