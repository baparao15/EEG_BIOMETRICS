from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch
import zlib

import numpy as np

from loop2 import mixed_probes, training_samples
from loop3_wbcic import ENROLLMENT_WINDOWS, PROBE_WINDOWS, UnusableSession, event_blocks, rest_prefix, timestamp


def fake_member(data):
    compressed = zlib.compress(data, wbits=-15)

    @contextmanager
    def stream(_):
        yield iter([compressed])

    return SimpleNamespace(file_size=len(data), CRC=zlib.crc32(data)), stream


def signal_header():
    header = bytearray(b" " * 512)
    header[:8] = b"\xffBIOSEMI"
    header[168:184] = b"20.03.1913.53.10"
    header[184:192] = b"512     "
    header[236:244] = b"3       "
    header[244:252] = b"1       "
    header[252:256] = b"1   "
    header[352:360] = b"?V      "
    header[360:376] = b"-375000 375000  "
    header[472:480] = b"2       "
    return bytes(header)


class WBCICTest(TestCase):
    def test_short_budget_keeps_both_arms_on_same_windows(self):
        x = (1 << np.arange(16, dtype=np.int64)).reshape(1, 1, 2, 8, 1)
        probes = mixed_probes(x, PROBE_WINDOWS)
        single = np.concatenate([probes["ce"], probes["oe"]], axis=-2)
        self.assertEqual(single.shape, (1, 1, 6, 1))
        self.assertEqual(probes["pair"].sum(), single.sum())
        self.assertEqual(single.sum() * 2, x[..., :6, :].sum())
        mixed = training_samples(x, 1, "mixed", ENROLLMENT_WINDOWS)
        np.testing.assert_array_equal(mixed.ravel(), x[..., :4, :].ravel())
        np.testing.assert_array_equal(training_samples(x, 1, "CE", ENROLLMENT_WINDOWS).ravel(),
                                      x[..., 0, :, :].ravel())

    def test_event_blocks_require_two_unambiguous_sixty_second_segments(self):
        header = bytearray(b" " * 768)
        header[168:184] = b"20.03.1913.53.10"
        header[184:192] = b"768     "
        events = b"+15.25\x150\x147\x14\0+77\x150\x148\x14\0+137\x141\x14\0"
        member, stream = fake_member(bytes(header) + events)
        with patch("loop3_wbcic.member_stream", stream):
            self.assertEqual(event_blocks(member), (timestamp(header), {"OE": 15.25, "CE": 77.0}))
        member, stream = fake_member(bytes(header) + events.replace(b"+137", b"+136"))
        with patch("loop3_wbcic.member_stream", stream), self.assertRaisesRegex(UnusableSession, "60-second"):
            event_blocks(member)

    def test_signal_prefix_preserves_samples_and_rewrites_record_count(self):
        header = signal_header()
        samples = bytes(range(18))
        member, stream = fake_member(header + samples)
        with TemporaryDirectory() as directory, patch("loop3_wbcic.member_stream", stream):
            output = Path(directory) / "rest.bdf"
            rest_prefix(member, output, 1.5, timestamp(header))
            cropped = output.read_bytes()
            self.assertEqual(int(cropped[236:244]), 2)
            self.assertEqual(cropped[352:360], b"uV      ")
            self.assertEqual(cropped[512:], samples[:12])

    def test_signal_prefix_rejects_checksum_mismatch_inside_or_after_prefix(self):
        header = signal_header()
        samples = bytes(range(18))
        original = header + samples
        member, _ = fake_member(original)
        for offset, replacement in ((183, b"1"), (len(header), b"\xff"), (len(original) - 1, b"\xff")):
            changed = original[:offset] + replacement + original[offset + 1:]
            _, stream = fake_member(changed)
            with self.subTest(offset=offset), TemporaryDirectory() as directory:
                output = Path(directory) / "rest.bdf"
                with patch("loop3_wbcic.member_stream", stream):
                    with self.assertRaisesRegex(ValueError, "signal member checksum"):
                        rest_prefix(member, output, 1.5, timestamp(header))
