from __future__ import annotations

from math import sqrt

import numpy as np
import pytest

from channel_dataset.channels.wireless.tr38901 import TappedDelayLine, TDLConfig


def _model(profile: str, *, delay_spread_s=100e-9, k_factor_db=None):
    fixed = profile in {"TDLA30", "TDLB100", "TDLC300", "TDLC60"}
    carrier = 28e9 if profile == "TDLC60" else 3.5e9
    return TappedDelayLine(
        TDLConfig(
            profile=profile, standard_version="16.1.0",
            delay_spread_s=None if fixed else delay_spread_s,
            carrier_frequency_hz=carrier, bandwidth_hz=100e6, speed_mps=0.0,
            num_rx=1, num_rx_antennas=1, num_tx=1, num_tx_antennas=1,
            num_sinusoids=16, k_factor_db=k_factor_db, correlation_mode="none",
            rx_correlation_coefficient=None, tx_correlation_coefficient=None,
            joint_correlation_matrix=None,
        )
    )


@pytest.mark.parametrize("profile", ["TDL-A", "TDL-B", "TDL-C", "TDL-D", "TDL-E"])
def test_scalable_profiles_have_requested_rms_delay_spread(profile) -> None:
    desired = 300e-9
    model = _model(profile, delay_spread_s=desired)
    delays = np.asarray(model.path_delays_s)
    powers = np.asarray(model.path_powers_linear)
    mean = float(np.sum(delays * powers))
    rms = sqrt(float(np.sum(powers * (delays - mean) ** 2)))
    assert sum(powers) == pytest.approx(1.0)
    assert rms == pytest.approx(desired, rel=1e-2)


@pytest.mark.parametrize(
    ("profile", "last_delay"),
    [("TDLA30", 290e-9), ("TDLB100", 480e-9), ("TDLC300", 2595e-9), ("TDLC60", 520e-9)],
)
def test_fixed_profiles_keep_tabulated_delays(profile, last_delay) -> None:
    model = _model(profile)
    assert model.path_delays_s[0] == 0.0
    assert model.path_delays_s[-1] == pytest.approx(last_delay)
    assert sum(model.path_powers_linear) == pytest.approx(1.0)


def test_desired_k_factor_sets_specular_to_diffuse_ratio() -> None:
    model = _model("TDL-D", k_factor_db=6.0)
    powers = model.path_powers_linear
    los = powers[model.profile.component_types.index("los")]
    diffuse = sum(
        power
        for power, kind in zip(powers, model.profile.component_types, strict=True)
        if kind == "rayleigh"
    )
    assert 10 * np.log10(los / diffuse) == pytest.approx(6.0)
