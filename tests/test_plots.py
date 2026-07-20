"""Trace-plot scaling -- the display-free parts (no Tk window is opened).

The y-scaling is the only place a plot can lie: a joint that barely moves must
not be drawn as if it swung wildly, and a joint that never moves must not divide
by a zero range.
"""

import unittest

import numpy as np

from gui.plots import TracePlotWindow

_range = TracePlotWindow._range


class TestTraceRange(unittest.TestCase):
    def test_spans_both_curves(self):
        """The action can leave the state's range -- it leads it -- so the band
        must cover both or the command would be clipped out of the picture."""
        state = np.array([0.0, 1.0, 2.0])
        action = np.array([-1.0, 1.0, 3.0])
        lo, hi = _range(state, action)
        self.assertLess(lo, -1.0)
        self.assertGreater(hi, 3.0)

    def test_pads_so_curves_do_not_touch_the_frame(self):
        lo, hi = _range(np.array([0.0, 10.0]), None)
        self.assertLess(lo, 0.0)
        self.assertGreater(hi, 10.0)

    def test_flat_signal_gets_a_band(self):
        """A joint that never moves would otherwise scale by zero."""
        lo, hi = _range(np.full(5, 0.7), None)
        self.assertGreater(hi - lo, 0.5)
        self.assertLess(lo, 0.7)
        self.assertGreater(hi, 0.7)

    def test_handles_a_missing_action(self):
        lo, hi = _range(np.array([-2.0, 5.0]), None)
        self.assertLess(lo, -2.0)
        self.assertGreater(hi, 5.0)


if __name__ == "__main__":
    unittest.main()
