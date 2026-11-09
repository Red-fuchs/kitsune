"""Opt-in LFM3-Audio talker for small talk (KITSUNE_TALKER=lfm; default unset = Gemma persona as before). Layers (KITSUNE_TALK_LAYERS, default '1234'):
1 prompt + few-shots, 2 factual questions never reach the talker (rules), 3 lark grammar without digits, 4 pre-speech check: unknown name/number -> hand off to Gemma."""
import os, re, time
import llguidance as llg
from mlx_vlm import stream_generate
from mlx_vlm.prompt_utils import apply_chat_template
import mlx_vlm.structured as st
import ctx as C

LAYERS = os.environ.get("KITSUNE_TALK_LAYERS", "1234")
RULES = (" Du bist nur zum Plaudern da und kennst keine Fakten. Nenne niemals Namen, Zahlen, Jahre, Orte oder Daten, die der Nutzer nicht selbst gesagt hat. "
         "Bei Wissensfragen sag nur: Das schlage ich lieber nach. Sprich den Nutzer immer mit du an, nie mit Sie.")
FEW = [("Wie hoch ist die Zugspitze?", "Das schlage ich lieber nach, ich will dir nichts Falsches sagen."),
       ("Ich bin ziemlich kaputt heute.", "Oh, das klingt anstrengend. Gönn dir nachher in Ruhe eine Pause."),
       ("Wie geht's dir?", "Mir geht es gut, danke. Und dir?"),
       ("Hast du ein Lieblingstier?", "Ich mag Füchse, schon wegen meines Namens. Und du?"),
       ("Sag mal was Lustiges.", "Ich bin ein Fuchs im Computer. Da ist es überraschend warm."),
       ("Wer hat Faust geschrieben?", "Das schlage ich lieber kurz nach."),
       ("Mir ist langweilig.", "Dann plaudern wir ein bisschen. Was beschäftigt dich gerade?")]
PERSONA = (C.SELF + " " + C.PROFILE + " " + C.STYLE + " Wiederhole niemals den Satz des Nutzers. Behaupte keine Fähigkeiten, die du nicht hast. "
           "Bei Beleidigungen oder anzüglichen Bemerkungen: kurz, ruhig und freundlich abwinken, keine Moralpredigt, nicht einsteigen.")

# L2: a factual question word + something that is not about the assistant
FACT = re.compile(r"\b(wer (ist|war|hat|waren|erfand)|wann|in welchem jahr|wie (viele|viel|hoch|alt|weit|lang|groß|teuer|spät|heißt|heißen|macht man)|was (kostet|ist|sind|bedeutet|heißt)|welche[rsmn]?|wo (liegt|ist|befindet)|wie heißt|welches datum)\b", re.I)
SELFQ = re.compile(r"\b(du|dir|dich|dein\w*|kitsune)\b", re.I)
def fact_q(q): return bool(FACT.search(q)) and not SELFQ.search(q)

GRAM = llg.grammar_from("lark", "start: TXT\nTXT: /[^0-9<>*#]+/\n")  # L3: no digits, no markup, free text; EOS allowed whenever TXT has >= 1 char

# L4 lexicon: words that may be capitalised in a harmless small-talk sentence (function words + frequent nouns); everything else capitalised is a possible invented name
FUNC = set("""ich du er sie es wir ihr ja nein na oh ach hm also aber und oder doch dann wenn weil dass das der die den dem des ein eine einen einem einer eines mein meine meinen dein deine deinen dir dich dein mir mich uns euch
 gerne gern leider vielleicht manchmal immer nie nur auch schon noch mal kein keine keinen hier dort danke bitte hallo hey tschüss gut gute guten schön schöne klar sicher natürlich genau wow super okay ok eigentlich ehrlich ehrlicherweise
 wie was wer wo wann warum wieso weshalb wozu wohin woher welche welcher welches man etwas nichts alles viel viele wenig jeder jede jedes sehr ganz bisschen heute morgen gestern jetzt gleich bald später früh""".split())
COMMON = set("""tag tage tages morgen abend nacht nachmittag vormittag woche wochen wochenende monat jahr zeit moment augenblick minute stunde pause ruhe schlaf traum träume träumen
 kaffee tee wasser essen trinken hunger durst lust laune stimmung gefühl gefühle freude spaß glück sorge sorgen stress arbeit job feierabend urlaub reise ferien wetter sonne regen wind schnee himmel musik lied song film buch bücher witz witze spiel
 freund freunde freundin freundlichkeit mensch menschen leute kollege nutzer assistent assistentin sprachassistent mac rechner computer gerät handy telefon ding sache sachen frage fragen antwort antworten idee ideen vorschlag tipp rat hilfe
 wort worte satz sätze sprache stimme ohr ohren kopf herz bauch auge augen hand hände tisch zimmer haus wohnung stadt land welt leben alltag plan pläne ziel tür fenster bett sofa licht farbe farben rot blau grün gelb schwarz weiß fuchs
 kater katze hund tier tiere vogel baum blume garten park platz weg straße bus bahn zug auto fahrrad kuchen brot obst gemüse suppe pizza schokolade eis kekse anfang ende mitte teil grund fall art weise mühe lob kompliment danke nachricht nachrichten
 mach machen neues neue ort orte snack tools tool entschuldigung problem probleme lösung chance fehler erfolg spaß runde gespräch unterhaltung quatsch unsinn blödsinn kleinigkeit besserung erholung energie kraft motivation""".split())
SUFFIX = re.compile(r"(ung|heit|keit|schaft|chen|lein|nis|tät|ion|ismus|ling|sein|zeit|werk)(en|s|e)?$", re.I)
NUMW = re.compile(r"\b(null|zwei|drei|vier|fünf|sechs|sieben|acht|neun|zehn|elf|zwölf|dreizehn|dreißig|\w+zehn|\w+zig|\w*hundert|\w*tausend|million\w*|milliard\w*|jahrhundert\w*)\b", re.I)
W = lambda s: re.findall(r"[\wäöüÄÖÜß'-]+", s)


def check(text, user="", notes="", hist=()):
    """-> list of reasons the sentence must not be spoken (empty = ok). Words from user text, notes, history and the persona are allowed."""
    ok = {w.lower() for w in W(user + " " + notes + " " + " ".join(u + " " + a for u, a in hist) + " " + PERSONA)}
    bad = []
    if re.search(r"\d", text): bad.append("digit")
    for m in NUMW.finditer(text):
        w = m.group().lower()
        if w not in ok and not any(w in o for o in ok if len(o) > 5): bad.append("num:" + w)
    for m in re.finditer(r"\b[A-ZÄÖÜ][\wäöüß-]+", text):
        w = m.group(); lw = w.lower()
        if lw in ok or lw in FUNC or lw in COMMON or SUFFIX.search(lw) or any(lw.startswith(c) and len(lw) - len(c) <= 3 for c in COMMON if len(c) > 3): continue
        bad.append("name:" + w)
    return bad


def nstatic(layers): return 1 + (2 * len(FEW) if "1" in layers else 0)


def msgs(k, q, layers=None, notes=None):
    layers = LAYERS if layers is None else layers
    sysm = PERSONA + (RULES if "1" in layers else "")
    if notes: sysm += "\nNotizen (nur nutzen, wenn sie zur Frage passen; nie vorlesen): " + notes
    ms = [{"role": "system", "content": sysm}]
    if "1" in layers: ms += [m for u, a in FEW for m in ({"role": "user", "content": u}, {"role": "assistant", "content": a})]
    for u, a in k.hist[-k.n_hist:]: ms += [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
    return ms + [{"role": "user", "content": q}]


_PC = {}  # static prefix (system + few-shots) -> KV state; own dict so the router's prefix cache (k.rc) is not evicted
CACHE = os.environ.get("KITSUNE_TALK_CACHE", "1") != "0"


def _prefix(k, pref):
    import copy
    from mlx_vlm.generate import PromptCacheState
    import mlx.core as mx
    if pref not in _PC:
        tk = getattr(k.Ap, "tokenizer", k.Ap); ims = tk.convert_tokens_to_ids("<|im_start|>")
        force = lambda tokens, logits: mx.where(mx.arange(logits.shape[-1]) == ims, logits, -1e9)
        s0 = PromptCacheState()
        for _ in stream_generate(k.A, k.Ap, pref, max_tokens=1, temperature=0.0, prompt_cache_state=s0, logits_processors=[force]): pass
        _PC[pref] = s0
    s0 = _PC[pref]; s = PromptCacheState(); s.cache = copy.deepcopy(s0.cache); s.token_ids = list(s0.token_ids); return s


def gen(k, ms, layers=None, prefill="", max_tokens=60, temp=0.3, static=0, cache=None):
    """text stream from LFM3-Audio (no audio). prefill: raw text appended to the assistant turn start (e.g. '<think>notes</think>\\n'). static: number of leading messages that never change (KV prefix cache)."""
    layers = LAYERS if layers is None else layers
    p = apply_chat_template(k.Ap, k.A.config, ms) + prefill
    kw = {"logits_processors": [st.LLGuidanceLogitsProcessor(GRAM, k.ltok)]} if "3" in layers else {}
    if (CACHE if cache is None else cache) and static:
        try:
            pref = apply_chat_template(k.Ap, k.A.config, ms[:static]); pref = pref[:pref.rindex("<|im_start|>assistant")]  # drop the generation prompt
            if p.startswith(pref): kw["prompt_cache_state"] = _prefix(k, pref)
        except Exception as e: print("talk cache off:", e)
    for r in stream_generate(k.A, k.Ap, p, max_tokens=max_tokens, temperature=temp, top_p=0.95, **kw): yield r.text


def talk(k, q, layers=None, on_block=None):
    """engine hook: sentence stream for small talk. L4: each sentence is checked before it is yielded; a blocked sentence hands the whole answer to Gemma (nothing spoken yet) or ends it."""
    layers = LAYERS if layers is None else layers
    sent = re.compile(r"(.+?[.!?…])(\s+|$)", re.S); buf, spoken = "", 0
    hist = k.hist[-k.n_hist:]
    for ch in gen(k, msgs(k, q, layers), layers, static=nstatic(layers)):
        buf += ch
        while (m := sent.match(buf)):
            s = m.group(1)
            if "4" in layers and (bad := check(s, q, "", hist)):
                if on_block: on_block(s, bad)
                if not spoken:
                    import engine as E
                    yield from k.brain_stream(k.brain_msgs(q, None, "plaudern"), 40)
                return
            spoken += 1; yield s + " "; buf = buf[m.end():]
    if buf.strip():
        if "4" in layers and (bad := check(buf, q, "", hist)):
            if on_block: on_block(buf, bad)
            if not spoken: yield from k.brain_stream(k.brain_msgs(q, None, "plaudern"), 40)
            return
        yield buf
