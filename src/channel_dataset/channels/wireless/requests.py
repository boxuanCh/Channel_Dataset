"""Typed requests for sampling batched wireless channels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from channel_dataset.backends import RandomStream
from channel_dataset.core import ExecutionContext


@dataclass(frozen=True, slots=True)
class ChannelSamplingRequest:
    """Per-call controls for sampling a batched wireless realization.

    Parameters
    ----------
    batch_size : int
        Positive number of independent batch entries.
    num_time_samples : int
        Positive number of coefficient time samples.
    sampling_frequency_hz : float
        Positive coefficient sampling frequency in hertz.
    execution : ExecutionContext
        Backend, precision, and device controls.
    random_stream : RandomStream
        Injected stream using the same backend and device as ``execution``.
    request_id : str, optional
        Nonempty caller identifier recorded in realization provenance.

    Raises
    ------
    ValueError
        If dimensions/frequency are not positive, the identifier is empty, or
        execution and random-stream backend/device values differ.
    TypeError
        If integer dimensions are not integers.
    """

    batch_size: int
    num_time_samples: int
    sampling_frequency_hz: float
    execution: ExecutionContext
    random_stream: RandomStream[Any]
    request_id: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.batch_size, bool) or not isinstance(self.batch_size, int):
            raise TypeError("batch_size must be an integer")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if isinstance(self.num_time_samples, bool) or not isinstance(
            self.num_time_samples, int
        ):
            raise TypeError("num_time_samples must be an integer")
        if self.num_time_samples <= 0:
            raise ValueError("num_time_samples must be a positive integer")
        if self.sampling_frequency_hz <= 0:
            raise ValueError("sampling_frequency_hz must be positive")
        if self.random_stream.backend_name != self.execution.backend:
            raise ValueError("random stream and execution context use different backends")
        if self.random_stream.device != self.execution.device:
            raise ValueError("random stream and execution context use different devices")
        if self.request_id is not None and not self.request_id:
            raise ValueError("request_id must be non-empty when provided")
