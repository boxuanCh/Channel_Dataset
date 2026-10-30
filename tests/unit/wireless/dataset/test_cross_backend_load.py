import importlib.util

import pytest

from channel_dataset.channels.wireless import load_cir, save_cir
from channel_dataset.core import ExecutionContext


def test_saved_realization_loads_on_other_backend(tmp_path, dataset_cir):
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")
    target = "torch" if dataset_cir.metadata.backend == "numpy" else "numpy"
    destination = save_cir(tmp_path / "realization", dataset_cir)
    loaded = load_cir(destination, ExecutionContext(target, "double", "cpu"))
    target_backend = ExecutionContext(target, "double", "cpu").array_backend
    assert loaded.metadata.backend == target
    assert target_backend.shape(loaded.coefficients) == dataset_cir.shape
    assert float(target_backend.to_scalar(loaded.coefficients[2, 0, 0, 0, 0, 0, 0].real)) == 3.0
