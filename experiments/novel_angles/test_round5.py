from collections import defaultdict
import unittest

import numpy as np
from lightgbm import LGBMClassifier
import torch
from torch import nn

from round5 import append_rotation, average_probes, logs_of_covariance, spatial_regions
from round5_models import RAW_MODELS, SIAMESE, StableBinary, contrastive, encoder, estimator


class Round5Test(unittest.TestCase):
    def test_lightgbm_small_fits_can_split_without_changing_full_budget(self):
        def kit_pipeline(name, n_features):
            return LGBMClassifier(n_estimators=3, verbose=-1)

        ns = {"get_model_pipeline": kit_pipeline}
        for n, leaf in ((64, 20), (32, 10), (16, 5), (8, 2)):
            with self.subTest(n=n):
                x = np.arange(n).reshape(-1, 1)
                y = (x[:, 0] >= n // 2).astype(int)
                model = estimator("LightGBM", 1, 42, ns, n).fit(x, y)
                self.assertEqual(model.min_child_samples, leaf)
                self.assertGreater(np.unique(model.predict_proba(x)[:, 1]).size, 1)

    def test_attempts_use_same_windows_and_frozen_impostor_cutoff(self):
        x = np.arange(4 * 3 * 2 * 4).reshape(4, 3, 2, 4, 1)
        subjects = np.array([2, 0, 3])
        attempts = average_probes(x, subjects, 4)
        np.testing.assert_allclose(attempts[..., :4, :].sum(axis=-2), attempts[..., 4:, :].sum(axis=-2))
        split = tuple(np.array([i]) for i in range(4))
        ledgers = defaultdict(list)
        append_rotation(ledgers, {"example": attempts}, split, 4)
        for arm, slice_ in (("single", slice(0, 4)), ("pair", slice(4, 8))):
            cut = np.nextafter(attempts[0, 0, slice_, 0].max(), np.inf)
            for visit in ("cal-unseen", "late"):
                ledger, = ledgers["example", arm, visit]
                self.assertEqual(set(ledger[:, 3]), {0, 3})
                np.testing.assert_array_equal(ledger[:, 4], cut)

    def test_log_covariance_and_regional_columns(self):
        a = logs_of_covariance(np.array([[[1., 0.], [0., 1.]]]))
        np.testing.assert_allclose(a, [[-np.log(2), 0., -np.log(2)]])
        idx = spatial_regions(["O1", "FP1", "C3", "P3"])
        self.assertEqual([block[0] for block in idx], [20, 40, 60, 0])

    def test_encoders_and_contrastive_gradient(self):
        torch.manual_seed(42)
        for model in (*RAW_MODELS, SIAMESE):
            dims = (4,) if model == SIAMESE else (4, 250)
            z = encoder(model, dims)(torch.rand(4, *dims))
            self.assertEqual(z.shape, (4, 32))
            loss = contrastive(z, torch.tensor([0, 0, 1, 1]))
            self.assertTrue(torch.isfinite(loss))
            loss.backward()

    def test_stable_binary_rejects_nonfinite_and_clips_large_inputs(self):
        def model(n):
            return nn.Sequential(nn.Linear(n, 1), nn.Flatten(0))
        x = np.array([[1e20], [-1e20], [1.], [-1.]])
        estimator = StableBinary(model, 42).fit(x, [1, 0, 1, 0])
        self.assertTrue(np.isfinite(estimator.predict_proba(x)).all())
        with self.assertRaises(FloatingPointError):
            estimator.predict_proba(np.array([[np.nan]]))


if __name__ == "__main__":
    unittest.main()
