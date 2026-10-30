import pytest

from channel_dataset.channels.wireless import CIRBatch


@pytest.fixture
def dataset_cir(backend, realization_metadata):
    coefficients = backend.ones((3, 1, 1, 1, 1, 2, 2), dtype="complex128")
    coefficients = coefficients * backend.reshape(
        backend.array([1.0, 2.0, 3.0], dtype="float64"), (3, 1, 1, 1, 1, 1, 1)
    )
    delays = backend.broadcast_to(
        backend.reshape(backend.array([0.0, 1.0e-6], dtype="float64"), (1, 1, 1, 2)),
        (3, 1, 1, 2),
    )
    mask = backend.ones((3, 1, 1, 2), dtype="bool")
    return CIRBatch(coefficients, delays, realization_metadata, mask)
