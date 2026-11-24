import unittest
from unittest.mock import patch

import numpy as np
from sklearn.svm import SVC

from split_critique import ONE_CLASS, compare, genuine_cutoff, group_range, ledger


class SplitCritiqueTest(unittest.TestCase):
    def test_genuine_cutoff_uses_low_score_tail(self):
        scores = np.arange(20)[::-1]
        cutoff = genuine_cutoff(scores)
        self.assertEqual(cutoff, 2)
        self.assertEqual(np.sum(scores < cutoff), 2)

    def test_group_range_uses_within_claimant_differences(self):
        summary = group_range({0: [0, .2], 1: [.4, .5]})
        self.assertEqual(summary["claimants_with_zero_and_at_least_15pct"], 1)
        self.assertAlmostEqual(summary["median_within_claimant_span"], .15)
        self.assertAlmostEqual(summary["max_within_claimant_span"], .2)

    def test_pair_coverage_and_cutoffs_are_unchanged_by_late_probes(self):
        rng = np.random.default_rng(42)
        groups = np.arange(10).reshape(5, 2)
        train = rng.normal(size=(10, 8, 3))
        probes = rng.normal(size=(10, 2, 8, 3))
        ns = {"get_model_pipeline": lambda *_: SVC()}
        with patch("split_critique.ledger", wraps=ledger) as baseline:
            result = compare(train, probes, groups, ns)
        changed = probes.copy()
        changed[:, 1] += 99
        with patch("split_critique.ledger", wraps=ledger) as intervention:
            compare(train, changed, groups, ns)
        cutoffs = lambda mocked: [call.args[3] for call in mocked.call_args_list]
        np.testing.assert_array_equal(cutoffs(baseline), cutoffs(intervention))
        self.assertEqual(result["identity_pairs"], {"rotation": 80, "crossfit": 90})
        self.assertEqual(set(result["results"]["all_other_genuine_cal"]), set(ONE_CLASS))
        self.assertEqual(result["supervised_membership"]["Logistic"]["seen_donors"]["attempts"], 1280)
        self.assertEqual(result["supervised_membership"]["Logistic"]["unseen_test"]["attempts"], 640)


if __name__ == "__main__":
    unittest.main()
