from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from channel_dataset.core import PathCluster, PropagationPath


def test_path_rejects_negative_delay() -> None:
    with pytest.raises(ValueError, match="delay_s"):
        PropagationPath(path_id="p0", complex_gain=1 + 0j, delay_s=-1.0)


def test_path_is_immutable() -> None:
    path = PropagationPath(path_id="p0", complex_gain=1 + 0j, delay_s=0.0)
    with pytest.raises(FrozenInstanceError):
        path.delay_s = 1.0  # type: ignore[misc]


def test_cluster_rejects_duplicate_path_ids() -> None:
    path = PropagationPath(path_id="p0", complex_gain=1 + 0j, delay_s=0.0)
    with pytest.raises(ValueError, match="unique"):
        PathCluster(cluster_id="c0", paths=(path, path))
