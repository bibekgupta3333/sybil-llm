"""Raw VeReMi-Extension trace files -> per-receiver message streams with ground-truth labels.

Implements tracker 4.5 (reproduce the VeReMi-Extension paper's plausibility-check baseline, Table III). Reads
`data/VeReMi-Dataset/<Family>_<group>/<run>/traceJSON-<vehicle>-<pseudo>-A<code>-<start>-<sim>.json` (read only,
RULE 2). Each trace file is one receiving vehicle: type 2 records are its own GPS fixes, type 3 records are the BSMs
it received. A received BSM is labelled misbehaving when its sending vehicle (`sender`) is an attacker in that run,
i.e. its own trace file carries a non-zero attack code (the ground-truth file `traceGroundTruthJSON-<sim>.json` has
no attacker field; its `sender` ids are the trace-file vehicle ids). Every received BSM at every receiver counts.
"""

from __future__ import annotations

import dataclasses
import json
import re
from collections.abc import Iterator
from pathlib import Path

import numpy as np

_TRACE_RE = re.compile(r"^traceJSON-(\d+)-(\d+)-A(\d+)-(\d+)-(\d+)\.json$")
_GROUND_TRUTH_RE = re.compile(r"^traceGroundTruthJSON-(\d+)\.json$")
UNKNOWN_LABEL = -1


class TraceFormatError(ValueError):
    """A trace file name or record does not match the VeReMi-Extension format."""


@dataclasses.dataclass(frozen=True)
class TraceFileName:
    """Fields of a `traceJSON-*.json` file name.

    Attributes:
        vehicle: SUMO vehicle id of the receiving vehicle (= `sender` of its own broadcasts).
        pseudo: a pseudonym of the vehicle at file creation.
        attack_code: 0 benign, 16 GridSybil, 17 DataReplaySybil, 18 DoSRandomSybil, 19 DoSDisruptiveSybil.
        start: run start second.
        sim: run id (matches the ground-truth file suffix).
    """

    vehicle: int
    pseudo: int
    attack_code: int
    start: int
    sim: int

    @classmethod
    def parse(cls, name: str) -> TraceFileName:
        """Parses a file name; raises TraceFormatError when it is not a trace file."""
        match = _TRACE_RE.match(name)
        if match is None:
            raise TraceFormatError(f"not a traceJSON file name: {name!r}")
        return cls(*(int(g) for g in match.groups()))


@dataclasses.dataclass
class ReceiverStream:
    """All BSMs one receiver heard, in reception order, with its own position at each reception.

    Shapes: n = number of received BSMs; vectors are (n, 2) float64 planar (x, y), metres / m/s / m/s^2.

    Attributes:
        receiver: vehicle id of the receiver.
        receiver_attack_code: attack code of the receiver itself (from its file name).
        rcv_time: (n,) reception time, s.
        send_time: (n,) send time, s.
        sender: (n,) int64 true sending vehicle (never used by the checks; labels only).
        pseudo: (n,) int64 sender pseudonym as broadcast (what the detector sees).
        message_id: (n,) int64.
        pos, spd, acl, hed: claimed position, velocity, acceleration, heading (unit vector).
        pos_conf, spd_conf: claimed confidence vectors (`pos_noise`, `spd_noise`, the BSM confidence fields), raw.
        hed_conf: claimed heading confidence (`hed_noise`), raw.
        rx_pos, rx_spd, rx_hed: the receiver's own latest GPS fix (type 2) at or before rcv_time.
        rx_pos_conf: that fix's position confidence (`pos_noise`).
        label: (n,) int8, 1 misbehaving sender, 0 genuine, UNKNOWN_LABEL when the sender has no trace file.
    """

    receiver: int
    receiver_attack_code: int
    rcv_time: np.ndarray
    send_time: np.ndarray
    sender: np.ndarray
    pseudo: np.ndarray
    message_id: np.ndarray
    pos: np.ndarray
    spd: np.ndarray
    acl: np.ndarray
    hed: np.ndarray
    pos_conf: np.ndarray
    spd_conf: np.ndarray
    hed_conf: np.ndarray
    rx_pos: np.ndarray
    rx_spd: np.ndarray
    rx_hed: np.ndarray
    rx_pos_conf: np.ndarray
    label: np.ndarray

    def __len__(self) -> int:
        return int(self.rcv_time.shape[0])


class RunIndex:
    """Attack code of every vehicle of one run folder, from its trace file names.

    Attributes:
        attack_code: vehicle id -> attack code.
        files: receiver trace files of the run.
    """

    def __init__(self, attack_code: dict[int, int], files: list[Path] | None = None) -> None:
        self.attack_code = attack_code
        self.files = files or []

    @classmethod
    def scan(cls, run_dir: Path) -> RunIndex:
        """Builds the index from the trace file names of `run_dir`; raises TraceFormatError on conflicting codes."""
        attack_code: dict[int, int] = {}
        files: list[Path] = []
        for path in sorted(run_dir.iterdir()):
            if not _TRACE_RE.match(path.name):
                continue
            meta = TraceFileName.parse(path.name)
            previous = attack_code.setdefault(meta.vehicle, meta.attack_code)
            if previous != meta.attack_code:
                raise TraceFormatError(
                    f"vehicle {meta.vehicle} has codes {previous} and {meta.attack_code} in {run_dir}"
                )
            files.append(path)
        return cls(attack_code, files)

    def labels(self, senders: np.ndarray) -> np.ndarray:
        """(n,) int8 label per sender id: 1 attacker, 0 genuine, UNKNOWN_LABEL when the vehicle has no file."""
        out = np.full(senders.shape, UNKNOWN_LABEL, dtype=np.int8)
        for i, sender in enumerate(senders.tolist()):
            code = self.attack_code.get(sender)
            if code is not None:
                out[i] = 1 if code != 0 else 0
        return out


class TraceReader:
    """Reads one receiver trace file into a `ReceiverStream`."""

    def __init__(self, run_index: RunIndex) -> None:
        self._run_index = run_index

    def read(self, path: Path) -> ReceiverStream:
        """Parses the JSON-lines file at `path`; raises TraceFormatError on an unknown record type."""
        meta = TraceFileName.parse(path.name)
        own: list[dict] = []
        received: list[dict] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                kind = record.get("type")
                if kind == 2:
                    own.append(record)
                elif kind == 3:
                    received.append(record)
                else:
                    raise TraceFormatError(f"record type {kind!r} in {path}")
        return self._build(meta, own, received)

    def _build(self, meta: TraceFileName, own: list[dict], received: list[dict]) -> ReceiverStream:
        received.sort(key=lambda r: r["rcvTime"])  # stable; files are already in reception order
        own.sort(key=lambda r: r["rcvTime"])
        n = len(received)
        rcv_time = np.array([r["rcvTime"] for r in received], dtype=np.float64)
        sender = np.array([r["sender"] for r in received], dtype=np.int64)
        own_time = np.array([r["rcvTime"] for r in own], dtype=np.float64)
        own_pos, own_spd, own_hed, own_conf = (_vec(own, key) for key in ("pos", "spd", "hed", "pos_noise"))
        if own:
            # latest own fix at or before the reception; messages before the first fix use the first fix
            idx = np.clip(np.searchsorted(own_time, rcv_time, side="right") - 1, 0, len(own) - 1)
            rx_pos, rx_spd, rx_hed, rx_conf = own_pos[idx], own_spd[idx], own_hed[idx], own_conf[idx]
        else:
            rx_pos = rx_spd = rx_hed = rx_conf = np.full((n, 2), np.nan)
        return ReceiverStream(
            receiver=meta.vehicle,
            receiver_attack_code=meta.attack_code,
            rcv_time=rcv_time,
            send_time=np.array([r["sendTime"] for r in received], dtype=np.float64),
            sender=sender,
            pseudo=np.array([r["senderPseudo"] for r in received], dtype=np.int64),
            message_id=np.array([r["messageID"] for r in received], dtype=np.int64),
            pos=_vec(received, "pos"),
            spd=_vec(received, "spd"),
            acl=_vec(received, "acl"),
            hed=_vec(received, "hed"),
            pos_conf=_vec(received, "pos_noise"),
            spd_conf=_vec(received, "spd_noise"),
            hed_conf=_vec(received, "hed_noise"),
            rx_pos=rx_pos,
            rx_spd=rx_spd,
            rx_hed=rx_hed,
            rx_pos_conf=rx_conf,
            label=self._run_index.labels(sender),
        )


def _vec(records: list[dict], key: str) -> np.ndarray:
    """(n, 2) float64 planar part of a 3-vector field."""
    if not records:
        return np.zeros((0, 2), dtype=np.float64)
    return np.array([r[key][:2] for r in records], dtype=np.float64)


class DatasetLayout:
    """Finds subsets (`<Family>_<group>` folders), their runs and trace files under the raw dataset root."""

    def __init__(self, root: Path) -> None:
        if not root.is_dir():
            raise FileNotFoundError(f"raw dataset folder not found: {root}")
        self.root = root

    def subsets(self) -> list[str]:
        """Sorted subset folder names."""
        return sorted(p.name for p in self.root.iterdir() if p.is_dir() and "_" in p.name)

    def runs(self, subset: str) -> list[Path]:
        """Run folders of one subset; raises FileNotFoundError when the subset is missing."""
        folder = self.root / subset
        if not folder.is_dir():
            raise FileNotFoundError(f"subset folder not found: {folder}")
        return sorted(p for p in folder.iterdir() if p.is_dir())

    def trace_files(self, subset: str) -> Iterator[tuple[Path, Path]]:
        """Yields (run_dir, trace_file) for every receiver of the subset."""
        for run_dir in self.runs(subset):
            for path in sorted(run_dir.iterdir()):
                if _TRACE_RE.match(path.name):
                    yield run_dir, path

    @staticmethod
    def has_ground_truth(run_dir: Path) -> bool:
        """True when the run folder holds its `traceGroundTruthJSON-*.json`."""
        return any(_GROUND_TRUTH_RE.match(p.name) for p in run_dir.iterdir())
