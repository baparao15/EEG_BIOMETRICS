"""Prepare and score the prespecified WBCIC-SHU three-day short-duration arm."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from datetime import datetime
from functools import partial
from io import SEEK_CUR, SEEK_END, SEEK_SET
from itertools import chain
import json
import math
from pathlib import Path
import re
import struct
from tempfile import TemporaryDirectory
from urllib.request import Request, urlopen
from zipfile import ZIP_DEFLATED, ZipFile
import zlib

import mne
import numpy as np

from loop2 import check_anchor, kit_module, ledgers, load_features, summarize
from loop3_dortmund import epoch_file
from recipe import NOTEBOOK_SHA, execute, file_sha


URL = "https://ndownloader.figshare.com/files/51001884"
ARCHIVE_BYTES = 65630556299
BLOCK_SECONDS = 60
PROBE_WINDOWS = 6
ENROLLMENT_WINDOWS = 8
VISITS = ("V0", "V2", "V3")
EVENT_RECORD = re.compile(rb"(\+\d+(?:\.\d+)?)(?:\x15[^\x00\x14]+)?\x14([0-9]+)\x14")


class UnusableSession(ValueError):
    pass


@contextmanager
def open_range(start, end):
    request = Request(f"{URL}?wb_range={start}-{end}", headers={"Range": f"bytes={start}-{end}"})
    with urlopen(request, timeout=90) as response:
        if (response.status != 206
                or response.headers.get("Content-Range") != f"bytes {start}-{end}/{ARCHIVE_BYTES}"):
            raise ValueError("Unexpected Figshare archive range")
        yield response


class ArchiveIndex:
    def __init__(self):
        self.position = 0

    def tell(self):
        return self.position

    def seek(self, offset, whence=SEEK_SET):
        origin = {SEEK_SET: 0, SEEK_CUR: self.position, SEEK_END: ARCHIVE_BYTES}[whence]
        self.position = origin + offset
        return self.position

    def read(self, count=-1):
        end = ARCHIVE_BYTES if count == -1 else min(ARCHIVE_BYTES, self.position + count)
        if end <= self.position:
            return b""
        with open_range(self.position, end - 1) as response:
            data = response.read()
        if len(data) != end - self.position:
            raise ValueError("Truncated Figshare ZIP index")
        self.position = end
        return data


def archive_inventory():
    files = defaultdict(dict)
    with ZipFile(ArchiveIndex()) as archive:
        for member in archive.infolist():
            match = re.fullmatch(r".+/2C dataset/(sub-\d+)/(ses-\d+)/eeg/(data|evt)\.bdf", member.filename)
            if match:
                subject, visit, kind = match.groups()
                if (visit, kind) in files[subject]:
                    raise ValueError("Duplicate BDF archive member")
                files[subject][visit, kind] = member
    expected = {(f"ses-0{day}", kind) for day in (1, 2, 3) for kind in ("data", "evt")}
    if len(files) != 51 or any(members.keys() != expected for members in files.values()):
        raise ValueError("Unexpected two-class WBCIC archive inventory")
    return files


@contextmanager
def member_stream(member):
    if member.compress_type != ZIP_DEFLATED:
        raise ValueError("Unsupported WBCIC compression")
    start = member.header_offset
    end = min(ARCHIVE_BYTES - 1, start + 30 + 131070 + member.compress_size - 1)
    with open_range(start, end) as stream:
        header = stream.read(30)
        if header[:4] != b"PK\x03\x04":
            raise ValueError("Invalid WBCIC member header")
        name_len, extra_len = struct.unpack_from("<HH", header, 26)
        stream.read(name_len + extra_len)
        remaining = member.compress_size

        def chunks():
            nonlocal remaining
            while remaining:
                chunk = stream.read(min(1 << 17, remaining))
                if not chunk:
                    raise ValueError("Truncated WBCIC member")
                remaining -= len(chunk)
                yield chunk

        yield chunks()


def timestamp(header):
    try:
        return datetime.strptime(header[168:184].decode("ascii"), "%d.%m.%y%H.%M.%S")
    except (ValueError, UnicodeError) as error:
        raise UnusableSession("invalid recording timestamp") from error


def event_blocks(member):
    with member_stream(member) as chunks:
        data = zlib.decompress(b"".join(chunks), -15)
    if len(data) != member.file_size or zlib.crc32(data) != member.CRC:
        raise ValueError("Invalid WBCIC event member checksum")
    start = int(data[184:192])
    events = [(float(t), int(code)) for t, code in EVENT_RECORD.findall(data[start:])]
    markers = {code: [t for t, value in events if value == code] for code in (7, 8)}
    motor = min((t for t, code in events if code in (1, 2)), default=-1)
    if any(len(values) != 1 for values in markers.values()):
        raise UnusableSession("missing or duplicate eye-state marker")
    oe, ce = markers[7][0], markers[8][0]
    if not (0 <= oe and oe + BLOCK_SECONDS <= ce and ce + BLOCK_SECONDS <= motor):
        raise UnusableSession("incomplete 60-second rest support before next marker")
    return timestamp(data), {"CE": ce, "OE": oe}


def rest_prefix(member, output, last_second, expected_stamp):
    decompressor = zlib.decompressobj(-15)
    with member_stream(member) as chunks:
        first = next(chunks)
        header = decompressor.decompress(first, 256)
        header_size = int(header[184:192])
        header += decompressor.decompress(decompressor.unconsumed_tail, header_size - 256)
        if len(header) != header_size or not header.startswith(b"\xffBIOSEMI"):
            raise ValueError("Invalid WBCIC data header")
        channels = int(header[252:256])
        bytes_per_record = 3 * sum(int(header[i:i + 8])
                                   for i in range(256 + 216 * channels, 256 + 224 * channels, 8))
        record_duration = float(header[244:252])
        requested = math.ceil(last_second / record_duration)
        available = int(header[236:244])
        if header_size + available * bytes_per_record != member.file_size:
            raise ValueError("WBCIC signal inventory conflicts with BDF header")
        records = min(requested, available)
        unit_start = 256 + 96 * channels
        unit_end = unit_start + 8 * channels
        units = bytearray(header[unit_start:unit_end])
        for offset in range(0, len(units), 8):
            low = unit_end + offset
            high = low + len(units)
            if (units[offset:offset + 8].strip() == b"?V"
                    and int(header[low:low + 8]) == -375000
                    and int(header[high:high + 8]) == 375000):
                units[offset:offset + 8] = b"uV      "
        with output.open("wb") as file:
            file.write(header[:236] + f"{records:<8}".encode()
                       + header[244:unit_start] + units + header[unit_end:])
            remaining = records * bytes_per_record
            size = len(header)
            crc = zlib.crc32(header)
            limit = 1 << 20
            for compressed in chain((decompressor.unconsumed_tail,), chunks):
                while True:
                    block = decompressor.decompress(compressed, limit)
                    size += len(block)
                    if size > member.file_size or decompressor.unused_data:
                        raise ValueError("Invalid WBCIC signal member size or stream")
                    crc = zlib.crc32(block, crc)
                    if remaining:
                        keep = min(remaining, len(block))
                        file.write(block[:keep])
                        remaining -= keep
                    compressed = decompressor.unconsumed_tail
                    if not compressed and len(block) < limit:
                        break
            if not decompressor.eof or size != member.file_size or remaining:
                raise ValueError("Invalid WBCIC signal member size or stream")
            if crc != member.CRC:
                raise ValueError("Invalid WBCIC signal member checksum")
        if timestamp(header) != expected_stamp:
            raise UnusableSession("signal and event timestamps differ")
        if requested > available:
            raise UnusableSession("recording ends before rest block")


def setup_bdf(path, onset):
    raw = mne.io.read_raw_bdf(path, exclude=["ECG", "HEOR", "HEOL", "VEOU", "VEOL"], verbose="ERROR")
    if len(raw.ch_names) != 59:
        raise UnusableSession("unexpected scalp channel count")
    raw.set_montage(mne.channels.make_standard_montage("standard_1005"))
    sfreq = raw.info["sfreq"]
    start = round(onset * sfreq)
    last = start + round(BLOCK_SECONDS * sfreq) - 1
    return raw.crop(start / sfreq, last / sfreq).load_data()


def configure_bdf(ns, onset):
    ns["step1_channel_setup"] = partial(setup_bdf, onset=onset)


def prepare_participant(job):
    subject, members, kit, output = job
    sessions = []
    exclusions = Counter()
    for visit in ("ses-01", "ses-02", "ses-03"):
        try:
            stamp, blocks = event_blocks(members[visit, "evt"])
            sessions.append((stamp, blocks, members[visit, "data"]))
        except UnusableSession as error:
            exclusions[str(error)] += 1
    if exclusions:
        return None, exclusions, 0
    if len({stamp.date() for stamp, _, _ in sessions}) != 3:
        return None, Counter({"fewer than three distinct dates": 1}), 0

    generated = 0
    for visit, (stamp, blocks, member) in zip(VISITS, sorted(sessions), strict=True):
        with TemporaryDirectory(dir=output) as directory:
            path = Path(directory) / "rest.bdf"
            if not all(epoch_file(output, subject, visit, state).is_file() for state in blocks):
                try:
                    rest_prefix(member, path, max(blocks.values()) + BLOCK_SECONDS, stamp)
                except UnusableSession as error:
                    exclusions[str(error)] += 1
                    continue
            for state, onset in blocks.items():
                key = subject, visit, state
                result = execute((kit, output, None, key, path, 50.0),
                                 partial(configure_bdf, onset=onset))
                if "error" in result:
                    error = result["error"]
                    if error.startswith("UnusableSession("):
                        category = "unexpected scalp channel count"
                    elif "One PCA component captures most" in error:
                        category = "single-component ICA rejection"
                    else:
                        category = f"recipe {error.split('(', 1)[0]}"
                    exclusions[category] += 1
                else:
                    generated += 1
                    minimum = ENROLLMENT_WINDOWS if visit == "V0" else PROBE_WINDOWS
                    if result["epochs"] < minimum:
                        exclusions[f"too few retained epochs ({'enrollment' if visit == 'V0' else 'probe'})"] += 1
    return (subject if not exclusions else None), exclusions, generated


def prepare(args):
    if file_sha(args.kit / "eeg_preprocessing_pipeline.ipynb") != NOTEBOOK_SHA:
        raise ValueError("Unexpected preprocessing notebook")
    (args.fifs / "logs").mkdir(parents=True, exist_ok=True)
    inventory = archive_inventory()
    eligible, counts, subjects_by_reason, total = [], Counter(), Counter(), 0
    jobs = [(person, members, args.kit, args.fifs) for person, members in sorted(inventory.items())]
    with ProcessPoolExecutor(args.workers) as pool:
        for i, (person, exclusions, generated) in enumerate(pool.map(prepare_participant, jobs), 1):
            counts.update(exclusions)
            subjects_by_reason.update(exclusions.keys())
            total += generated
            if person:
                eligible.append(person)
            print(f"Prepared {i}/{len(jobs)} participants ({len(eligible)} eligible)", flush=True)
    summary = dict(inventory_subjects=len(inventory), prepared_recordings=total,
                   eligible_subjects=len(eligible), exclusion_events=dict(counts),
                   subjects_by_exclusion_reason=dict(subjects_by_reason))
    print(json.dumps(summary), flush=True)
    (args.fifs / "prepared.json").write_text(json.dumps({**summary, "eligible": eligible}) + "\n")


def score(args):
    anchor = check_anchor(args.anchor)
    summary = json.loads((args.fifs / "prepared.json").read_text())
    eligible = summary.pop("eligible")
    if len(set(eligible)) != summary["eligible_subjects"] or len(eligible) < 5:
        raise ValueError("Invalid eligible cohort")
    inputs = args.fifs / "scoring-inputs"
    inputs.mkdir(exist_ok=True)
    if any(path.name not in eligible for path in inputs.iterdir()):
        raise ValueError("Stale WBCIC scoring inputs")
    for person in eligible:
        target = inputs / person
        target.mkdir(exist_ok=True)
        for visit in VISITS:
            for state in ("CE", "OE"):
                path = epoch_file(args.fifs, person, visit, state)
                link = target / path.name
                if not link.exists():
                    link.symlink_to(path.resolve())

    kit = kit_module(args.kit)
    notebook = Path(__file__).resolve().parents[2] / "07_complete_results_all_models_all_sessions.ipynb"
    people, channels, visits, arrays, min_epochs = load_features(
        inputs, "wbcic-shu", args.kit, notebook, args.workers, expected_people=len(eligible),
        windows=PROBE_WINDOWS, enrollment_windows=ENROLLMENT_WINDOWS,
    )
    groups = [tuple(sorted(g)) for g in np.array_split(np.random.default_rng(42).permutation(len(people)), 5)]
    data = ledgers(arrays, channels, visits, groups, kit.notebook_env(notebook),
                   windows=PROBE_WINDOWS, enrollment_windows=ENROLLMENT_WINDOWS)
    result = summarize(data, len(people), 500, np.repeat("fixed-oe-first", len(people)))
    result.update(summary, dataset="WBCIC-SHU Figshare v5 2C", protocol="modified-duration three-day",
                  subjects=len(people), shortest_recording=min_epochs, draws=500,
                  probe_windows_per_state=PROBE_WINDOWS, enrollment_windows=ENROLLMENT_WINDOWS,
                  anchor_frr_penalties_pp=anchor)
    temp = args.output.with_suffix(".partial")
    temp.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temp.replace(args.output)
    for scorer in ("cosine", "svm"):
        vals = result["contrasts"][f"{scorer}/pair-minus-single"]["eer"]
        print(f"{scorer}: EER difference {100 * vals['point']:.2f} pp "
              f"[{100 * vals['low']:.2f}, {100 * vals['high']:.2f}]", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "score"))
    parser.add_argument("fifs", type=Path)
    parser.add_argument("kit", type=Path)
    parser.add_argument("--anchor", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    args.fifs /= "crc-checked"
    if args.stage == "prepare":
        prepare(args)
    elif args.anchor is None or args.output is None:
        parser.error("score requires --anchor and --output")
    else:
        score(args)


if __name__ == "__main__":
    main()
