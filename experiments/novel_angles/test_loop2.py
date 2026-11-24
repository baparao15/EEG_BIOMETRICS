import unittest

import numpy as np

from benchmark import metric
from loop2 import align, boot_metrics, mixed_probes, rotations, score_modes, training_samples


class Loop2Test(unittest.TestCase):
    def test_mixed_design_uses_identical_windows_and_enrollment_budget(self):
        x = (1 << np.arange(48, dtype=np.int64)).reshape(1, 1, 2, 24, 1)
        probes = mixed_probes(x)
        single = np.concatenate([probes["ce"], probes["oe"]], axis=-2)
        self.assertEqual(probes["pair"].shape, single.shape)
        self.assertEqual(probes["pair"].sum(), single.sum())
        self.assertEqual(probes["pair"].sum() * 2, x.sum())
        for n in (1, 2):
            x = np.ones((2, n, 2, 32, 1))
            self.assertEqual(training_samples(x, n, "mixed").shape, (2, 32, 1))
            self.assertEqual(training_samples(x, n, "CE").shape, (2, 32, 1))

    def test_roles_and_frozen_calibration_use_disjoint_people(self):
        split = next(rotations([(0, 1), (2,), (3,), (4,), (5,)]))
        claim, train, cal, test = split
        probe = np.zeros((6, 3, 4, 1))
        probe[cal, -2] = .5
        probe[cal, -1] = 1.
        probe[test, -2:] = .6
        calls = []

        def scorer(train_data, score_inputs, targets, refs):
            calls.append(refs)
            return np.repeat(score_inputs, len(targets), axis=-1)

        modes = {"a": probe, "b": probe}
        fit = score_modes({"shared": probe}, modes, split, {"a": "shared", "b": "shared"}, scorer,
                          {"pooled": ("a", "b")})
        self.assertEqual(len(calls), 1)
        np.testing.assert_array_equal(calls[0], train)
        self.assertEqual(len(fit["late", "pooled"]), 32)
        self.assertEqual(set(fit["late", "pooled"][:, 3]), set((*claim, *test)))
        self.assertEqual(metric(fit["late", "pooled"], np.ones(6, int))[1], 1.)
        np.testing.assert_array_equal(fit["late", "pooled"][:, 4], fit["cal-unseen", "pooled"][:, 4])

    def test_weighted_bootstrap_matches_pilot_metric_with_ties(self):
        rows = np.array([[.5, 1, 0, 0, .55], [.6, 1, 1, 1, .55], [.5, 0, 0, 1, .55],
                         [.6, 0, 1, 0, .55], [.2, 0, 1, 2, .55], [.9, 0, 0, 2, .55]])
        weights = np.array([[1, 1, 1], [1, 2, 0], [1, 0, 2], [3, 0, 0]])
        np.testing.assert_allclose(boot_metrics(rows, weights), [metric(rows, w) for w in weights])

    def test_same_visit_alignment_does_not_pool_nominal_late_visits(self):
        cov = np.ones((4, 3, 2, 24, 1, 1))
        cov[1, -1] = 4
        cov[2:, -1] = 2
        visits = np.array([("V0", "V2", v) for v in ("V3", "V4", "V3", "V4")])
        values = align(cov, np.array([0, 1]), visits, 1, "visit-condition")
        np.testing.assert_allclose(values[2:, -1, 0, 0, 0], np.log([2, .5]))
