import time, unittest
import places as P

TOP = [("Blau tief", "Plowdiw"), ("Plaudiv", "Plowdiw"), ("Plowdiw", "Plowdiw"), ("Plovdiv", "Plowdiw"), ("Hamburg", "Hamburg"), ("Sunny Beach", "Sonnenstrand"),
       ("Sofia", "Sofia"), ("Basen", "Basel"), ("Münschen", "München"), ("Frankfurt am Main", "Frankfurt am Main"), ("Köln", "Köln"), ("Nürnberk", "Nürnberg"),
       ("Stuttgart", "Stuttgart"), ("Düsseldorff", "Düsseldorf"), ("Warna", "Warna"), ("Burgas", "Burgas"), ("Salzburk", "Salzburg"), ("Wien", "Wien")]

class T(unittest.TestCase):
    def test_top(self):
        bad = [(q, w, P.cands(q)[:2]) for q, w in TOP if not (P.cands(q) and P.cands(q)[0][0] == w)]
        print(f"precision@1 {len(TOP) - len(bad)}/{len(TOP)}", bad)
        self.assertLessEqual(len(bad), 2)
        for q, w in [("Blau tief", "Plowdiw"), ("Plaudiv", "Plowdiw"), ("Basen", "Basel")]: self.assertEqual(P.cands(q)[0][0], w)

    def test_decide(self):
        self.assertEqual(P.decide(P.cands("Hamburg")), ("use", "Hamburg"))
        self.assertEqual(P.decide(P.cands("Plowdiw")), ("use", "Plowdiw"))
        self.assertEqual(P.decide(P.cands("Plovdiv")), ("use", "Plowdiw"))
        self.assertEqual(P.decide(P.cands("Sunny Beach")), ("use", "Sonnenstrand"))
        self.assertEqual(P.decide(P.cands("Blau tief")), ("ask", "Plowdiw"))
        for q in ("Plaudiv", "Basen"): self.assertEqual(P.decide(P.cands(q))[0], "ask")
        self.assertEqual(P.decide([("A", .9), ("B", .85)]), ("ask", "A"))
        self.assertEqual(P.decide([("A", .9), ("A Kreis", .88)]), ("use", "A"))
        self.assertEqual(P.decide([("A", .5)]), ("none", None))
        self.assertEqual(P.decide([]), ("none", None))
        self.assertEqual(P.decide(P.cands("guten Morgen"))[0], "none")

    def test_sonnenstrand_besteffort(self):
        r = [l for l, _ in P.cands("Tensulibach", 3)]
        print("Tensulibach top3:", r)  # known weak: ASR string shares ~no phonemes with Sonnenstrand/Sunny Beach

    def test_speed(self):
        P._I = None; t = time.time(); P.cands("Hamburg"); st = time.time() - t
        t = time.time(); [P.cands(q) for q, _ in TOP]; qt = (time.time() - t) / len(TOP)
        print(f"startup {st:.2f}s, query {qt * 1000:.0f}ms")
        self.assertLess(st, 1.5); self.assertLess(qt, .15)

if __name__ == "__main__": unittest.main()
