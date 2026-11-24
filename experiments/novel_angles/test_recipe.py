import unittest

import mne
import numpy as np

from recipe import compare


class RecipeTest(unittest.TestCase):
    def test_compares_waveforms_only_at_shared_event_starts(self):
        def epochs(values, event_starts):
            info = mne.create_info(["Cz"], 250, "eeg")
            events = np.column_stack((event_starts, np.zeros(3, int), np.ones(3, int)))
            return mne.EpochsArray(np.array(values)[:, None, :], info, events, verbose="ERROR")

        reference = epochs([[0, 1], [1, 2], [2, 3]], [0, 10, 20])
        actual = epochs([[1, 2], [2, 3], [4, 5]], [10, 20, 30])
        result = compare(actual, reference)
        self.assertEqual(result["common_epochs"], 2)
        self.assertEqual(result["event_jaccard"], .5)
        self.assertFalse(result["same_events"])
        self.assertAlmostEqual(result["waveform_correlation"], 1)
        self.assertEqual(result["rmse"], 0)
