"""Backend-neutral interfaces implemented by concrete channel models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

RequestT = TypeVar("RequestT")
RealizationT = TypeVar("RealizationT")


class ChannelModel(ABC, Generic[RequestT, RealizationT]):
    """Generic interface for configured channel-realization generators.

    Type Parameters
    ---------------
    RequestT
        Typed per-call sampling request.
    RealizationT
        Fixed realization produced by :meth:`realize`.

    Notes
    -----
    Signal application is intentionally absent. A model samples reusable state;
    independent operators transform or apply that state to signals.
    """

    @property
    @abstractmethod
    def family(self) -> str:
        """Return the channel family key used by configuration and registries."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the model key within its family."""

    @abstractmethod
    def realize(self, request: RequestT) -> RealizationT:
        """Sample a fixed, reusable realization.

        Parameters
        ----------
        request : RequestT
            Typed request containing per-call dimensions, execution controls,
            and an injected random stream.

        Returns
        -------
        RealizationT
            Realization that can be transformed or applied repeatedly without
            resampling.
        """
