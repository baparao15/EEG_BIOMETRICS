from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import mne
import numpy as np

from loop3_dortmund import eligibility, inventory, stage_channels


class DortmundTest(TestCase):
    def test_staging_reorders_electrodes_without_swapping_signal_identity(self):
        with TemporaryDirectory() as directory:
            source = Path(directory) / "source_epo.fif"
            target = Path(directory) / "staged_epo.fif"
            data = np.array([[[2., 3.], [4., 5.]]])
            epochs = mne.EpochsArray(data, mne.create_info(["Fp2", "Fp1"], 250, "eeg"),
                                     verbose="ERROR")
            epochs.save(source, fmt="double", verbose="ERROR")
            self.assertTrue(stage_channels(source, target, ("FP1", "FP2")))
            staged = mne.read_epochs(target, preload=True, verbose="ERROR")
            self.assertEqual(staged.ch_names, ["Fp1", "Fp2"])
            np.testing.assert_array_equal(staged.get_data(), data[:, ::-1])
            self.assertEqual(mne.read_epochs(source, preload=False, verbose="ERROR").ch_names, ["Fp2", "Fp1"])
            with self.assertRaisesRegex(ValueError, "electrode set"):
                stage_channels(source, target, ("FP1", "C3"))

    def test_inventory_uses_followup_pre_and_same_day_post_calibration(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)

            def add(person, day, block, state):
                folder = root / person / f"ses-{day}" / "eeg"
                folder.mkdir(parents=True, exist_ok=True)
                path = folder / f"{person}_ses-{day}_task-Eyes{state}_acq-{block}_eeg.edf"
                path.symlink_to("annex.edf")
                return path

            expected = {}
            for day, block, visit in ((1, "pre", "V0"), (1, "post", "V2"), (2, "pre", "V3")):
                for state, label in (("Closed", "CE"), ("Open", "OE")):
                    expected[visit, label] = add("sub-a", day, block, state)
                    if (day, block, state) != (2, "pre", "Open"):
                        add("sub-b", day, block, state)
            add("sub-b", 2, "post", "Open")
            self.assertEqual(inventory(root), {"sub-a": expected})

    def test_eligibility_requires_every_block_and_does_not_mask_unexpected_errors(self):
        population = {"sub-a": {("V0", "CE"): None, ("V2", "OE"): None}}
        counts = {"sub-a_V0_CE": 32, "sub-a_V2_OE": 24}
        self.assertEqual(eligibility(population, counts, []), ["sub-a"])
        self.assertEqual(eligibility(population, {**counts, "sub-a_V0_CE": 31}, []), [])
        counts.pop("sub-a_V0_CE")
        with self.assertRaisesRegex(ValueError, "Incomplete preparation"):
            eligibility(population, counts, [])
        rejection = {"recording": "sub-a_V0_CE",
                     "error": "RuntimeError('One PCA component captures most of the explained variance')"}
        self.assertEqual(eligibility(population, counts, [rejection]), [])
        with self.assertRaisesRegex(ValueError, "Unexpected preprocessing failures"):
            eligibility(population, counts, [{**rejection, "error": "ValueError('Malformed output')"}])
