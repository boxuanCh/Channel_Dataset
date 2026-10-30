"""Chunked realization persistence, streaming, shuffling, and replay."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Iterator
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np

from channel_dataset.backends import RandomStream
from channel_dataset.channels.wireless.requests import ChannelSamplingRequest
from channel_dataset.channels.wireless.types import CIRBatch
from channel_dataset.core import ChannelModel, ExecutionContext, RealizationMetadata

_MANIFEST = "manifest.json"
_AXES = {
    "coefficients": ["batch", "rx", "rx_antenna", "tx", "tx_antenna", "path", "time"],
    "delays_s": ["batch", "rx", "tx", "path"],
    "path_mask": ["batch", "rx", "tx", "path"],
}


def _numpy_array(value: Any, backend_name: str) -> np.ndarray:
    if backend_name == "numpy":
        return np.asarray(value)
    return value.detach().cpu().numpy()


def _write_chunks(
    root: Path,
    name: str,
    value: Any,
    backend_name: str,
    chunk_size: int,
) -> dict[str, Any]:
    array = _numpy_array(value, backend_name)
    files = []
    for index, start in enumerate(range(0, array.shape[0], chunk_size)):
        filename = f"{name}.{index:06d}.npy"
        target = root / filename
        np.save(target, array[start : start + chunk_size], allow_pickle=False)
        digest = sha256(target.read_bytes()).hexdigest()
        files.append({"file": filename, "sha256": digest})
    return {"dtype": array.dtype.str, "shape": list(array.shape), "chunks": files}


def save_cir(path: str | Path, cir: CIRBatch, *, chunk_size: int = 64) -> Path:
    """Store a CIR in a versioned, chunked directory.

    Parameters
    ----------
    path : path-like
        New or empty destination directory.
    cir : CIRBatch
        Realization to persist.
    chunk_size : int, default=64
        Maximum number of batch entries in each ``.npy`` chunk.

    Returns
    -------
    pathlib.Path
        Resolved destination path.

    Notes
    -----
    Metadata remains human-readable JSON. Arrays use non-pickled NPY chunks
    with a SHA-256 checksum per chunk. Existing nonempty directories are never
    overwritten.
    """

    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    root = Path(path).resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"destination is not empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    arrays = {
        "coefficients": _write_chunks(
            root, "coefficients", cir.coefficients, cir.metadata.backend, chunk_size
        ),
        "delays_s": _write_chunks(
            root, "delays_s", cir.delays_s, cir.metadata.backend, chunk_size
        ),
    }
    if cir.path_mask is not None:
        arrays["path_mask"] = _write_chunks(
            root, "path_mask", cir.path_mask, cir.metadata.backend, chunk_size
        )
    manifest = {
        "schema_version": 1,
        "kind": "CIRBatch",
        "axes": _AXES,
        "arrays": arrays,
        "metadata": cir.metadata.to_document(),
    }
    (root / _MANIFEST).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return root


def _validate_manifest(document: Any) -> dict[str, Any]:
    if not isinstance(document, dict):
        raise TypeError("realization manifest must be an object")
    expected = {"schema_version", "kind", "axes", "arrays", "metadata"}
    if set(document) != expected:
        raise ValueError("realization manifest has unknown or missing fields")
    if document["schema_version"] != 1:
        raise ValueError("unsupported realization schema version")
    if document["kind"] != "CIRBatch" or document["axes"] != _AXES:
        raise ValueError("unsupported realization kind or axis contract")
    arrays = document["arrays"]
    if not isinstance(arrays, dict) or not {"coefficients", "delays_s"} <= set(arrays):
        raise ValueError("realization arrays are incomplete")
    if set(arrays) - {"coefficients", "delays_s", "path_mask"}:
        raise ValueError("realization contains unknown arrays")
    return document


def _read_chunks(root: Path, description: Any, execution: ExecutionContext) -> Any:
    if not isinstance(description, dict) or set(description) != {"dtype", "shape", "chunks"}:
        raise ValueError("invalid array descriptor")
    chunks = []
    for entry in description["chunks"]:
        if not isinstance(entry, dict) or set(entry) != {"file", "sha256"}:
            raise ValueError("invalid chunk descriptor")
        source = root / entry["file"]
        content = source.read_bytes()
        if sha256(content).hexdigest() != entry["sha256"]:
            raise ValueError(f"array chunk checksum mismatch: {entry['file']}")
        chunks.append(np.load(source, allow_pickle=False))
    if not chunks:
        raise ValueError("array descriptor contains no chunks")
    value = np.concatenate(chunks, axis=0)
    if list(value.shape) != description["shape"] or value.dtype.str != description["dtype"]:
        raise ValueError("array shape or dtype does not match manifest")
    return execution.array_backend.array(
        value, dtype=value.dtype.name, device=execution.device
    )


def load_cir(path: str | Path, execution: ExecutionContext) -> CIRBatch:
    """Load a saved CIR on either supported execution backend."""

    root = Path(path).resolve()
    document = _validate_manifest(json.loads((root / _MANIFEST).read_text(encoding="utf-8")))
    arrays = document["arrays"]
    metadata_document = dict(document["metadata"])
    metadata_document["backend"] = execution.backend
    metadata_document["device"] = execution.device
    metadata_document["precision"] = execution.precision
    metadata = RealizationMetadata(**metadata_document)
    coefficients = _read_chunks(root, arrays["coefficients"], execution)
    delays = _read_chunks(root, arrays["delays_s"], execution)
    mask = None
    if "path_mask" in arrays:
        mask = _read_chunks(root, arrays["path_mask"], execution)
    return CIRBatch(coefficients, delays, metadata, mask)


class StreamingCIRDataset(Iterable[CIRBatch]):
    """Repeatable finite or cyclic stream produced by an index sampler."""

    def __init__(
        self, sampler: Callable[[int], CIRBatch], length: int, *, repeat: bool = False
    ) -> None:
        if length <= 0:
            raise ValueError("length must be positive")
        self.sampler = sampler
        self.length = int(length)
        self.repeat = bool(repeat)

    def __iter__(self) -> Iterator[CIRBatch]:
        while True:
            for index in range(self.length):
                yield self.sampler(index)
            if not self.repeat:
                return


def shuffle_stream(
    values: Iterable[CIRBatch],
    buffer_size: int,
    random_stream: RandomStream[Any],
) -> Iterator[CIRBatch]:
    """Yield a deterministic bounded-buffer shuffle of a CIR stream."""

    if buffer_size <= 0:
        raise ValueError("buffer_size must be positive")
    stream = random_stream.fork("wireless.dataset.shuffle")
    iterator = iter(values)
    buffer = []
    for value in iterator:
        if len(buffer) < buffer_size:
            buffer.append(value)
            continue
        index = int(stream.integers((1,), 0, len(buffer))[0])
        yield buffer[index]
        buffer[index] = value
    while buffer:
        index = int(stream.integers((1,), 0, len(buffer))[0])
        yield buffer.pop(index)


class ReplayChannel(ChannelModel[ChannelSamplingRequest, CIRBatch]):
    """Replay stored CIR realizations sequentially as a channel model."""

    def __init__(self, realizations: Iterable[CIRBatch], *, repeat: bool = False) -> None:
        self.realizations = tuple(realizations)
        if not self.realizations:
            raise ValueError("at least one realization is required")
        self.repeat = repeat
        self._index = 0

    @property
    def family(self) -> str:
        """str: Replay model family."""

        return "wireless"

    @property
    def name(self) -> str:
        """str: Replay model name."""

        return "replay"

    def realize(self, request: ChannelSamplingRequest) -> CIRBatch:
        """Return the next compatible stored realization."""

        if self._index >= len(self.realizations):
            if not self.repeat:
                raise StopIteration("replay dataset is exhausted")
            self._index = 0
        value = self.realizations[self._index]
        self._index += 1
        if value.metadata.backend != request.execution.backend:
            raise ValueError("replay realization backend must match request")
        if value.shape[0] != request.batch_size or value.shape[-1] != request.num_time_samples:
            raise ValueError("replay realization dimensions must match request")
        return value


__all__ = [
    "ReplayChannel",
    "StreamingCIRDataset",
    "load_cir",
    "save_cir",
    "shuffle_stream",
]
