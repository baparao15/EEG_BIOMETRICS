import unittest

import numpy as np

from benchmark import FAR, ledger, metric, probe_features, representations, roles, tangent, threshold


class BenchmarkTest(unittest.TestCase):
    def test_each_subject_rotates_through_disjoint_roles(self):
        splits = list(roles(29))
        for split in splits:
            np.testing.assert_array_equal(np.sort(np.concatenate(split)), np.arange(29))
        for role in zip(*splits):
            np.testing.assert_array_equal(np.sort(np.concatenate(role)), np.arange(29))

    def test_threshold_is_conservative_with_boundary_ties(self):
        scores = np.r_[np.arange(197), 197, 197, 197].astype(np.float32)
        self.assertLessEqual(np.mean(scores.astype(float) >= threshold(scores)), FAR)
        self.assertEqual(np.sum(np.arange(200) >= threshold(np.arange(200))), 2)

    def test_ledger_only_admits_unseen_impostors_and_matching_targets(self):
        scores = np.array([[[.8, .95]], [[.99, .7]], [[.1, .6]]])
        rows = ledger(scores, np.array([0, 1, 2]), np.array([0, 1]), .5)
        self.assertEqual(len(rows), 4)
        np.testing.assert_allclose(metric(rows, np.ones(3)), [0, .5, 0])

    def test_pair_and_single_modes_use_two_epochs_per_attempt(self):
        x = np.zeros((2, 4, 2, 24, 1))
        x[:, :, 0, :, 0] = np.arange(24) * 2
        x[:, :, 1, :, 0] = np.arange(24) * 2 - 5
        np.testing.assert_array_equal(probe_features(x, "CE")[0, 0, :, 0], np.arange(12) * 4 + 1)
        np.testing.assert_array_equal(probe_features(x, "pair-delta"), np.full((2, 4, 24, 1), 5))
        np.testing.assert_array_equal(probe_features(x, "ce-repeat-delta"), np.full((2, 4, 12, 1), -2))

    def test_paired_mean_and_pooled_single_state_use_identical_windows(self):
        windows = 1 << np.arange(48, dtype=np.int64)
        x = windows.reshape(1, 1, 2, 24, 1)
        paired = probe_features(x, "pair-mean")
        single = np.concatenate([probe_features(x, state) for state in ("CE", "OE")], axis=2)
        self.assertEqual(paired.shape, single.shape)
        self.assertEqual(int(paired.sum() * 2), int(windows.sum()))
        self.assertEqual(int(single.sum() * 2), int(windows.sum()))

    def test_whitening_maps_reference_to_zero(self):
        reference = np.array([[2, .2], [.2, 1]])
        np.testing.assert_allclose(tangent(reference, reference), np.zeros(3), atol=1e-12)

    def test_late_alignment_only_uses_donors_from_the_probe_visit(self):
        cov = np.ones((4, 4, 2, 24, 1, 1))
        cov[1, 3] = 4
        cov[2:, 3] = 2
        arrays = {"cov": cov, "late_visits": np.array(["V3", "V4", "V3", "V4"])}
        aligned = representations(arrays, ["Cz"], np.array([0, 1]), "cov", "all", "visit-condition")
        np.testing.assert_allclose(aligned[2:, 3, 0, 0, 0], np.log([2, .5]))
        unsupported = representations(arrays, ["Cz"], np.array([0]), "cov", "all", "visit-condition")
        self.assertTrue(np.isnan(unsupported[3, 3]).all())
        self.assertTrue(np.isfinite(unsupported[3, :3]).all())


if __name__ == "__main__":
    unittest.main()
