"""Convert continuous path delays into a sampled time-domain response."""

from __future__ import annotations

from math import isfinite
from numbers import Real

from channel_dataset.backends import get_backend
from channel_dataset.channels.wireless.geometric_cdl.types import DynamicCIRBatch
from channel_dataset.channels.wireless.transforms.time_lags import time_lag_bounds
from channel_dataset.channels.wireless.types import CIRBatch, TimeChannelResponse


def cir_to_time(
    cir: CIRBatch | DynamicCIRBatch,
    sampling_frequency_hz: float,
    *,
    lag_min: int | None = None,
    lag_max: int | None = None,
    sinc_half_width: int = 6,
    interpolation: str = "sinc",
) -> TimeChannelResponse:
    r"""Convert a path-based CIR to a discrete-time channel response.

    Parameters
    ----------
    cir : CIRBatch
        Fixed path coefficients and physical delays.
    sampling_frequency_hz : float
        Positive, finite sampling frequency in hertz.
    lag_min, lag_max : int, optional
        Inclusive output sample-lag bounds. Either provide both bounds or omit
        both to derive them using :func:`time_lag_bounds`.
    sinc_half_width : int, default=6
        Truncation margin used only when bounds are derived automatically.
    interpolation : {"sinc"}, default="sinc"
        Fractional-delay interpolation kernel. The current implementation uses
        normalized sinc interpolation.

    Returns
    -------
    TimeChannelResponse
        Response with axes ``[batch, rx, rx_ant, tx, tx_ant, output_time,
        lag]``. ``output_time`` equals the CIR time dimension.

    Raises
    ------
    TypeError
        If scalar arguments have incompatible types.
    ValueError
        If the frequency or bounds are invalid, or if an unsupported kernel is
        requested.

    Notes
    -----
    For integer lag :math:`\ell`, the transform evaluates

    .. math::

       h[t,\ell] = \sum_p a_p(t)\,\operatorname{sinc}
       (\ell - \tau_p f_s).

    Invalid masked paths contribute zero. Native backend operations preserve
    PyTorch gradients through both path coefficients and delays.
    """

    if isinstance(sampling_frequency_hz, bool) or not isinstance(
        sampling_frequency_hz, Real
    ):
        raise TypeError("sampling_frequency_hz must be a real number")
    if not isfinite(float(sampling_frequency_hz)) or sampling_frequency_hz <= 0:
        raise ValueError("sampling_frequency_hz must be finite and positive")
    if interpolation != "sinc":
        raise ValueError("interpolation must be 'sinc'")
    if (lag_min is None) != (lag_max is None):
        raise ValueError("lag_min and lag_max must be provided together")

    backend = get_backend(cir.metadata.backend)
    if lag_min is None:
        max_delay_s = float(backend.to_scalar(backend.max(cir.delays_s)))
        lag_min, lag_max = time_lag_bounds(
            max_delay_s,
            float(sampling_frequency_hz),
            sinc_half_width=sinc_half_width,
        )
    else:
        if isinstance(lag_min, bool) or not isinstance(lag_min, int):
            raise TypeError("lag_min must be an integer")
        if isinstance(lag_max, bool) or not isinstance(lag_max, int):
            raise TypeError("lag_max must be an integer")
        if lag_max < lag_min:
            raise ValueError("lag_max must be greater than or equal to lag_min")

    b, r, ra, t, ta, p, time = cir.shape
    lag_count = lag_max - lag_min + 1
    lag_values = backend.arange(
        lag_count,
        dtype=cir.metadata.precision == "single" and "float32" or "float64",
        device=cir.metadata.device,
    ) + lag_min
    lags = backend.reshape(lag_values, (1, 1, 1, 1, 1, 1, 1, lag_count))
    if isinstance(cir, DynamicCIRBatch):
        dynamic_delays = backend.transpose(cir.delays_s, (0, 1, 2, 4, 3))
        delays = backend.reshape(dynamic_delays, (b, r, 1, t, 1, time, p, 1))
    else:
        delays = backend.reshape(cir.delays_s, (b, r, 1, t, 1, 1, p, 1))
    kernel = backend.sinc(lags - delays * float(sampling_frequency_hz))
    coefficients = backend.transpose(cir.coefficients, (0, 1, 2, 3, 4, 6, 5))
    coefficients = backend.reshape(coefficients, (b, r, ra, t, ta, time, p, 1))
    terms = coefficients * kernel
    if cir.path_mask is not None:
        if isinstance(cir, DynamicCIRBatch):
            dynamic_mask = backend.transpose(cir.path_mask, (0, 1, 2, 4, 3))
            mask = backend.reshape(dynamic_mask, (b, r, 1, t, 1, time, p, 1))
        else:
            mask = backend.reshape(cir.path_mask, (b, r, 1, t, 1, 1, p, 1))
        terms = terms * mask
    response = backend.sum(terms, axis=-2)
    return TimeChannelResponse(response, cir.metadata, lag_min)
