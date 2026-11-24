import unittest

import numpy as np

from round4 import variant_rows
from round6 import (adjustments, audited_rotations, audited_rows, calibration_ranks, cp_upper,
                    far_power, fused_calibration_score, logistic_fit_data, meta_analysis, minimum_n,
                    standard_error, standardized)


class Round6Test(unittest.TestCase):
    def test_multiplicity_adjustments_restore_input_order(self):
        holm, bh = adjustments([.03, .002, .01, .09])
        np.testing.assert_allclose(holm, [.06, .008, .03, .09])
        np.testing.assert_allclose(bh, [.04, .008, .02, .09])

    def test_cp_best_case_and_minimum_with_errors(self):
        self.assertEqual(minimum_n(0), 299)
        for errors in (0, 1, 20):
            n = minimum_n(errors)
            self.assertLessEqual(cp_upper(errors, n), .01)
            self.assertGreater(cp_upper(errors, n - 1), .01)

    def test_distinct_identity_and_attempt_events(self):
        # Two accepts on one of two impostor identities.
        ledger = np.array([[.9, 0, 0, 1, .8], [.85, 0, 0, 1, .8], [.4, 0, 0, 2, .8],
                           [.95, 1, 0, 0, .8]])
        draws = np.array([[1, 1, 1], [1, 2, 0], [1, 0, 2]])
        power = far_power(ledger, draws)
        self.assertEqual((power["attempts"], power["accepted"]), (3, 2))
        self.assertEqual((power["impostor_identities"], power["identities_ever_accepted"]), (2, 1))
        self.assertAlmostEqual(power["clustered_upper"], .95)

    def test_role_audit_rejects_duplicate(self):
        groups = np.arange(10).reshape(5, 2).tolist()
        self.assertEqual(len(audited_rotations(groups, 10)), 20)
        groups[-1][0] = 1
        with self.assertRaises(ValueError):
            audited_rotations(groups, 10)

    def test_fit_slices_ignore_probe_visits_and_nonfit_identities(self):
        rng = np.random.default_rng(177)
        split = tuple(np.array(a) for a in ([0, 1], [2, 3], [4], [5]))
        x = rng.normal(size=(6, 3, 2, 6, 3))
        changed = x.copy()
        changed[:, 1:] += 100
        changed[4:] -= 787
        np.testing.assert_array_equal(logistic_fit_data(x, split, 8), logistic_fit_data(changed, split, 8))
        before = standardized(x, split[1], 1, 6)
        after = standardized(changed, split[1], 1, 6)
        np.testing.assert_array_equal(before[:4, :1], after[:4, :1])

    def test_only_calibration_impostors_set_cutoff_and_fusion_scale(self):
        split = tuple(np.array(a) for a in ([0], [1], [2], [3]))
        x = np.linspace(0, 1, 24).reshape(3, 2, 4, 1)
        before = audited_rows(x, split)
        changed = x.copy()
        changed[1:] += 100
        after = variant_rows(changed, split, "attempt")
        np.testing.assert_array_equal(before["late"][:, 4], after["late"][:, 4])
        np.testing.assert_array_equal(calibration_ranks(x, 1)[0], calibration_ranks(changed, 1)[0])
        self.assertEqual(calibration_ranks(x, 1)[0, 0, -1, 0], 4)

    def test_fusion_percentiles_use_inclusive_ties_and_claimant_specific_ranks(self):
        scores = np.array([[[[1, 10], [2, 10], [2, 20], [4, 30]],
                            [[0, 9], [2, 10], [3, 15], [5, 31]]]])
        result = calibration_ranks(scores, 1)
        np.testing.assert_array_equal(result[0, 1], [[0, 0], [3, 2], [3, 2], [4, 4]])

    def test_fused_scores_keep_equal_rank_sums_tied(self):
        first, second = np.indices((13, 13)).reshape(2, -1)
        def scores(ranks):
            result = np.zeros((16, 2, 12, 1))
            result[0, 0, :, 0] = np.arange(1, 13)
            result[1:, 1, :, 0] = np.pad(ranks, (0, 11)).reshape(15, 12)
            return result

        fused = fused_calibration_score(scores(first), scores(second), 1)
        late = fused[1:, 1].ravel()[:first.size]
        np.testing.assert_array_equal(late, (first + second) / 24)
        self.assertEqual(len(np.unique(late)), 25)

    def test_meta_analysis_consistent_effects(self):
        effect = dict(point=-.05, low=-.07, high=-.03)
        self.assertGreater(standard_error(effect), 0.)
        pooled = meta_analysis([effect] * 4)
        self.assertAlmostEqual(pooled["point"], -.05)
        self.assertEqual(pooled["i2"], 0.)
        self.assertLess(pooled["high"], 0.)


if __name__ == "__main__":
    unittest.main()
