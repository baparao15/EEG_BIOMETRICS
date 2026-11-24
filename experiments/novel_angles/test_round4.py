import unittest

import numpy as np

from round4 import gates, repeatability_weights, variant_rows, variant_scores


class Round4Test(unittest.TestCase):
    def test_repeatability_favors_stable_between_person_features(self):
        enrollment = np.array([[[[0., 0.], [0., 1.]]] * 2,
                               [[[1., 0.], [1., 1.]]] * 2])
        np.testing.assert_allclose(repeatability_weights(enrollment), [1., .25])

    def test_variant_uses_separate_state_templates_and_enrollment_reactivity(self):
        x = np.zeros((4, 3, 2, 2, 2))
        x[:, :, 0, :, 0] = 1
        x[:, :, 1, :, 1] = 1
        x[3, -1] = x[3, -1, ::-1]
        split = tuple(np.array([i]) for i in range(4))
        alpha = np.zeros(x.shape[:-1])
        alpha[3, -1, 0] = 1
        out = variant_scores(x, alpha, split, windows=2, enrollment_windows=4)
        np.testing.assert_allclose(out["identity-maximum"], 1)
        np.testing.assert_allclose(out["worst-state"][:, 0], 1)
        np.testing.assert_allclose(out["worst-state"][-1, -1], 0)
        np.testing.assert_allclose(out["reactivity"][-1, -1], .75)

    def test_identity_maximum_threshold_excludes_held_out_attempts(self):
        split = tuple(np.array(ids) for ids in ([0], [1], [2, 3, 4, 5, 6], [7]))
        scores = np.linspace(0., .9, 7 * 2 * 24).reshape(7, 2, 24, 1)
        out = variant_rows(scores, split, "identity-maximum")
        cutoff = np.nextafter(scores[:5, 0].max(), np.inf)
        for ledger in out.values():
            np.testing.assert_array_equal(ledger[:, 4], cutoff)
            self.assertEqual(set(ledger[:, 3]), {0, 7})
        pooled = variant_rows(scores, split, "worst-state")
        self.assertLess(pooled["late"][0, 4], cutoff)

    def test_zero_observed_far_is_not_a_replication_win(self):
        improved = dict.fromkeys(("eer", "frr", "far"), {"high": -.001})
        self.assertFalse(gates(improved, improved, {"point": 0., "high": 0.})["replication_candidate"])
        self.assertTrue(gates(improved, improved, {"point": .002, "high": .008})["replication_candidate"])


if __name__ == "__main__":
    unittest.main()
