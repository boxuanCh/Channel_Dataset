import inspect

import pytest

from channel_dataset.backends import create_random_stream, get_backend
from channel_dataset.backends.numpy_backend import NumpyBackend, NumpyRandomStream
from channel_dataset.channels.wireless import (
    compute_link_geometry,
    create_scenario,
    load_large_scale_parameters,
    load_scenario_geometry,
    register_wireless_models,
    reverse_ray_batch,
    select_paths,
)
from channel_dataset.channels.wireless.antennas import (
    global_to_local_angles,
    load_antenna_layout,
    local_field_to_global,
    local_to_global_angles,
    parse_antenna_layout,
    rotation_matrix,
    spherical_basis,
)
from channel_dataset.channels.wireless.fading import parse_rayleigh_config
from channel_dataset.channels.wireless.operators import apply_ofdm, apply_time
from channel_dataset.channels.wireless.tr38901 import (
    load_cdl_profile,
    load_cdl_ray_offsets,
    load_tdl_profile,
    parse_cdl_config,
    parse_tdl_config,
    register_cdl_model,
    register_tdl_model,
)
from channel_dataset.channels.wireless.transforms import (
    cir_to_ofdm,
    cir_to_time,
    normalize_response,
    subcarrier_frequencies,
    time_lag_bounds,
    time_lags,
)
from channel_dataset.config import load_config, parse_config_document


@pytest.mark.parametrize(
    "function",
    [
        create_random_stream,
        get_backend,
        select_paths,
        compute_link_geometry,
        reverse_ray_batch,
        load_scenario_geometry,
        create_scenario,
        load_large_scale_parameters,
        register_wireless_models,
        spherical_basis,
        rotation_matrix,
        global_to_local_angles,
        local_to_global_angles,
        local_field_to_global,
        load_antenna_layout,
        parse_antenna_layout,
        parse_rayleigh_config,
        apply_ofdm,
        apply_time,
        cir_to_ofdm,
        cir_to_time,
        normalize_response,
        subcarrier_frequencies,
        time_lag_bounds,
        time_lags,
        load_cdl_profile,
        load_cdl_ray_offsets,
        parse_cdl_config,
        register_cdl_model,
        load_tdl_profile,
        parse_tdl_config,
        register_tdl_model,
        load_config,
        parse_config_document,
    ],
)
def test_public_functions_have_numpy_style_sections(function) -> None:
    doc = inspect.getdoc(function)
    assert doc is not None
    if inspect.signature(function).parameters:
        assert "Parameters\n----------" in doc
    assert "Returns\n-------" in doc or function.__name__.startswith("register_")


def test_concrete_backends_inherit_protocol_documentation() -> None:
    assert "Create a native array filled with zeros" in inspect.getdoc(NumpyBackend.zeros)
    assert "continuous uniform distribution" in inspect.getdoc(NumpyRandomStream.uniform)
