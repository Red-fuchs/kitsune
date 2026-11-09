import sys, pathlib; HERE = pathlib.Path(__file__).parent; sys.path[:0] = [str(HERE.parent)]
import os; os.environ["KITSUNE_GATE"] = "0"
from datetime import datetime
import engine as E
k = E.Kitsune(dry=True)
H = [("Was steht heute an?", "Heute hast du einen Termin. Allerheiligen.")]
Q = ["Was steht heute an?", "was steht heute an", "Kitsune, was steht heute an?", "Was geht heute an?", "Und wie viele Einwohner hat Hamburg?", "Wie viele Einwohner hat Hamburg?", "Was läuft gerade auf der Gebühr?", "Wie ist das Wetter in Blau tief?", "Was heißt Danke auf Bulgarisch?", "Wie sagt man Hallo auf Englisch?", "Wer war Goethe?", "Erkläre mir kurz, wie ein Regenbogen entsteht.", "Trag morgen um halb drei ein.", "Und morgen?"]
for hist in ([],):
    k.hist = hist
    for q in Q:
        n, a = k.route(q, datetime.now()); print(bool(hist), q, "->", n, a)
