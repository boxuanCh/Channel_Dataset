import json

import pytest

from channel_dataset.channels.wireless import load_cir, save_cir


def test_corrupt_chunk_is_rejected(tmp_path, dataset_cir, execution_context):
    destination = save_cir(tmp_path / "realization", dataset_cir)
    chunk = next(destination.glob("coefficients.*.npy"))
    chunk.write_bytes(chunk.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        load_cir(destination, execution_context)


def test_unknown_schema_version_is_rejected(tmp_path, dataset_cir, execution_context):
    destination = save_cir(tmp_path / "realization", dataset_cir)
    path = destination / "manifest.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    document["schema_version"] = 2
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="schema version"):
        load_cir(destination, execution_context)
