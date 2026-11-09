"""Addressee model: d1-2B 'noul' question 'is this utterance directed at a voice assistant?' (one forward pass, calibrated p)."""
import sys, time, pathlib
HOME = pathlib.Path.home()
sys.path.insert(0, str(__import__('os').environ.get('KITSUNE_MLXVLM_SRC') or HOME / 'Claude-Workspace/lfm3/mlx-vlm'))
D1 = HOME / "AI-Model/models/LiquidAI/d1-2B-MLX-8bit"
Q = {"addr": {"type": "noul", "instructions": "Ist diese Äußerung an einen Sprachassistenten gerichtet (nicht Fernsehen/Radio oder Gespräch mit Dritten)?"}}


def state(q, ctx=()):
    s = "Du bist der Sprachassistent Kitsune im Wohnzimmer. Das Transkript kommt aus automatischer Spracherkennung und kann Fehler enthalten.\n"
    if ctx: s += "Bisheriges Gespräch:\n" + "\n".join(f"Nutzer: {u}\nKitsune: {a[:120]}" for u, a in list(ctx)[-2:]) + "\n"
    return s + f"Neue Äußerung aus dem Raum: „{q}“"


def tool_q(tools):
    """d1 choice question over Kitsune's tools (name -> description); netz_lies is a deterministic follow-up, never picked from scratch."""
    return {"type": "choice", "instructions": "Welches Werkzeug braucht der Sprachassistent für diese Äußerung?", "criteria": {n: d for n, d in tools if n != "netz_lies"}}


QB = {"barge": {"type": "noul", "instructions": "Will der Nutzer Kitsune damit unterbrechen, stoppen oder korrigieren (nicht Fernsehen/Radio, nicht Gespräch mit Dritten, nicht Kitsunes eigene Stimme als Echo, nicht nur zustimmendes Mhm)?"}}


def state_barge(q, speaking, ctx=()):
    s = "Du bist der Sprachassistent Kitsune. Das Transkript kommt aus automatischer Spracherkennung, ist oft nur der Anfang eines Satzes und kann Fehler enthalten.\n"
    if ctx: s += "Bisheriges Gespräch:\n" + "\n".join(f"Nutzer: {u}\nKitsune: {a[:120]}" for u, a in list(ctx)[-1:]) + "\n"
    return s + f"Kitsune spricht gerade: „{speaking[:200]}“\nWährenddessen im Raum gehört: „{q}“"


class Addressee:
    def __init__(self, path=D1):
        from mlx_vlm import load
        self.m, self.p = load(str(path)); self.ms = None

    def p_addr(self, q, ctx=()):
        t = time.perf_counter()
        out = self.m.system_one(state(q, ctx), Q)
        self.ms = round((time.perf_counter() - t) * 1000)
        return float(out["answers"]["addr"]["noul"])

    def ask(self, q, ctx=(), tools=None, addr=True):
        """one forward pass: P(addressed) and/or tool choice -> {'p': .., 'tool': .., 'conf': .., 'ms': ..}."""
        t = time.perf_counter(); qs = {**(Q if addr else {}), **({"tool": tool_q(tools)} if tools else {})}
        a = self.m.system_one(state(q, ctx), qs)["answers"]; self.ms = round((time.perf_counter() - t) * 1000)
        r = {"ms": self.ms}
        if addr: r["p"] = float(a["addr"]["noul"])
        if tools: r["tool"], r["conf"] = a["tool"]["choice"], round(float(a["tool"]["confidence"]), 3)
        return r

    def p_barge(self, q, speaking, ctx=()):
        """P(user wants to interrupt/correct) for a capture during playback; one forward pass, also returns P(addressed) on the plain state for comparison."""
        t = time.perf_counter(); a = self.m.system_one(state_barge(q, speaking, ctx), QB)["answers"]; self.ms = round((time.perf_counter() - t) * 1000)
        return float(a["barge"]["noul"])
