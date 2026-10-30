from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from channel_dataset.channels.wireless.types import CIRBatch
from channel_dataset.core import RealizationMetadata


def _metadata(backend_name: str) -> RealizationMetadata:
    return RealizationMetadata(
        model_family="wireless",
        model_name="test",
        schema_version=1,
        backend=backend_name,
        precision="double",
        device="cpu",
        stream_id=f"{backend_name}:1:root",
        config_digest="0" * 64,
        resolved_configuration={},
        software_version="test",
    )


def test_valid_cir_contract(backend, backend_name) -> None:
    coefficients = backend.ones((2, 3, 2, 4, 2, 5, 6), dtype="complex128")
    delays = backend.zeros((2, 3, 4, 5), dtype="float64")
    mask = backend.ones((2, 3, 4, 5), dtype="bool")
    cir = CIRBatch(coefficients, delays, _metadata(backend_name), mask)
    assert cir.shape == (2, 3, 2, 4, 2, 5, 6)


def test_cir_rejects_dimension_mismatch(backend, backend_name) -> None:
    coefficients = backend.ones((1, 1, 1, 1, 1, 2, 1), dtype="complex128")
    delays = backend.zeros((1, 1, 1, 1), dtype="float64")
    with pytest.raises(ValueError, match="does not match"):
        CIRBatch(coefficients, delays, _metadata(backend_name))


def test_cir_rejects_negative_delay(backend, backend_name) -> None:
    coefficients = backend.ones((1, 1, 1, 1, 1, 1, 1), dtype="complex128")
    delays = backend.full((1, 1, 1, 1), -1.0, dtype="float64")
    with pytest.raises(ValueError, match="nonnegative"):
        CIRBatch(coefficients, delays, _metadata(backend_name))


def test_cir_rejects_metadata_backend_mismatch(backend, backend_name) -> None:
    other = "torch" if backend_name == "numpy" else "numpy"
    coefficients = backend.ones((1, 1, 1, 1, 1, 1, 1), dtype="complex128")
    delays = backend.zeros((1, 1, 1, 1), dtype="float64")
    with pytest.raises(ValueError, match="metadata"):
        CIRBatch(coefficients, delays, replace(_metadata(backend_name), backend=other))


def test_cir_selects_multi_link_view(backend, backend_name) -> None:
    coefficients = backend.ones((2, 3, 2, 4, 2, 5, 6), dtype="complex128")
    delays = backend.zeros((2, 3, 4, 5), dtype="float64")
    cir = CIRBatch(coefficients, delays, _metadata(backend_name))
    selected = cir.select_link(1, 2, 3)
    assert selected.shape == (1, 1, 2, 1, 2, 5, 6)
    if backend_name == "torch":
        assert (
            coefficients.untyped_storage().data_ptr()
            == selected.coefficients.untyped_storage().data_ptr()
        )
    else:
        assert np.shares_memory(coefficients, selected.coefficients)
