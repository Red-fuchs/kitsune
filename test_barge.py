"""Barge calibration prior (no audio): python -I -m unittest discover -s . -p 'test_barge.py'"""
import unittest, tempfile, pathlib
import numpy as np
import barge as B


class T(unittest.TestCase):
    def test_prior_roundtrip_and_floor(self):
        old = B.G_FILE; B.G_FILE = pathlib.Path(tempfile.mkdtemp()) / "g.json"
        try:
            B.Room.gmin = B.G_MIN_SPK
            B.Room.save_prior("mic|speakers|40", np.full(len(B.BANDS), 0.12)); B.Room.save_prior("mic|speakers|80", np.full(len(B.BANDS), 0.01))
            self.assertAlmostEqual(float(B.Room.load_prior("mic|speakers|40").mean()), 0.12)
            self.assertAlmostEqual(float(B.Room.load_prior("mic|speakers|80").mean()), B.G_MIN_SPK)  # a stored too-low G is lifted to the speaker floor
            self.assertIsNone(B.Room.load_prior("other|key"))
        finally: B.G_FILE = old; B.Room.gmin = B.G_MIN


if __name__ == "__main__": unittest.main()
