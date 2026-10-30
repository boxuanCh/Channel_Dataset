from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from channel_dataset.backends import registry


def test_numpy_backend_import_does_not_import_torch() -> None:
    root = Path(__file__).resolve().parents[2]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(root / "src")
    command = (
        "import sys; import channel_dataset.backends.numpy_backend; "
        "assert 'torch' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", command], env=environment, check=True)


@pytest.mark.torch
def test_default_backend_is_torch_when_installed() -> None:
    pytest.importorskip("torch")
    assert registry.get_backend().name == "torch"


def test_missing_default_torch_has_actionable_error(monkeypatch: pytest.MonkeyPatch) -> None:
    registry._CACHE.pop("torch", None)

    def fail_import(name: str):
        raise ImportError(name)

    monkeypatch.setattr(registry, "import_module", fail_import)
    with pytest.raises(ImportError, match="wireless-torch"):
        registry.get_backend()
