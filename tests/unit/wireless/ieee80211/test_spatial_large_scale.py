from __future__ import annotations

import numpy as np
import pytest

from channel_dataset.channels.wireless.ieee80211.large_scale import tgn_breakpoint_path_loss_db
from channel_dataset.channels.wireless.ieee80211.spatial import laplacian_ula_covariance


def _as_numpy(value):
    if type(value).__module__.partition(".")[0] == "torch":
        return value.detach().cpu().numpy()
    return value


@pytest.mark.ieee80211
@pytest.mark.tgn
def test_laplacian_covariance_is_hermitian_psd(execution_context) -> None:
    covariance = laplacian_ula_covariance(4, 0.5, 0.3, 0.4, execution_context)
    values = _as_numpy(covariance)
    assert np.allclose(values, values.conj().T, atol=1e-12)
    assert np.allclose(np.diag(values), 1.0, atol=1e-12)
    assert np.linalg.eigvalsh(values).min() >= -1e-11


@pytest.mark.ieee80211
@pytest.mark.tgn
def test_tgn_path_loss_is_continuous_at_breakpoint(backend) -> None:
    distances = backend.array([9.999999, 10.0, 10.000001], dtype="float64")
    loss = _as_numpy(tgn_breakpoint_path_loss_db(distances, 5.25e9, 10.0, backend))
    assert loss[0] == pytest.approx(loss[1], abs=2e-6)
    assert loss[2] == pytest.approx(loss[1], abs=2e-6)
    assert np.all(np.diff(loss) > 0)
