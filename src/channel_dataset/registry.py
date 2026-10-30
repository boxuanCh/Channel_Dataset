"""Model registration and construction without family-to-family coupling."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from channel_dataset.config import ConfigEnvelope
from channel_dataset.core import ChannelModel

ParameterParser = Callable[[Mapping[str, Any]], object]
ModelBuilder = Callable[[object], ChannelModel[Any, Any]]


@dataclass(frozen=True, slots=True)
class ModelRegistration:
    """Typed parsing and construction boundary for one model.

    Parameters
    ----------
    parse_parameters : callable
        Converts an unstructured parameter mapping into a typed configuration.
    build : callable
        Constructs a :class:`ChannelModel` from that typed configuration.
    """

    parse_parameters: ParameterParser
    build: ModelBuilder


class ModelRegistry:
    """Map stable ``(family, name)`` keys to typed model builders.

    Notes
    -----
    Parsing occurs before construction, so model constructors never receive raw
    configuration mappings.
    """

    def __init__(self) -> None:
        self._registrations: dict[tuple[str, str], ModelRegistration] = {}

    def register(
        self,
        family: str,
        name: str,
        parse_parameters: ParameterParser,
        builder: ModelBuilder,
    ) -> None:
        """Register a model parser and builder.

        Parameters
        ----------
        family, name : str
            Stable configuration key.
        parse_parameters : callable
            Strict parser from a parameter mapping to a typed configuration.
        builder : callable
            Constructor accepting the typed parser result.

        Raises
        ------
        ValueError
            If the key is already registered.
        """
        key = (family, name)
        if key in self._registrations:
            raise ValueError(f"model is already registered: {family}/{name}")
        self._registrations[key] = ModelRegistration(parse_parameters, builder)

    def create(self, config: ConfigEnvelope) -> ChannelModel[Any, Any]:
        """Construct a model from a validated configuration envelope.

        Parameters
        ----------
        config : ConfigEnvelope
            Envelope whose family, name, and parameters select the registration.

        Returns
        -------
        ChannelModel
            Configured model built from typed parameters.

        Raises
        ------
        KeyError
            If no model is registered for the envelope key.
        ValueError, TypeError
            If model-specific parameter validation fails.
        """
        key = (config.family, config.name)
        try:
            registration = self._registrations[key]
        except KeyError as error:
            raise KeyError(f"unknown channel model: {config.family}/{config.name}") from error
        typed_config = registration.parse_parameters(config.parameters)
        return registration.build(typed_config)

    def available_models(self) -> tuple[tuple[str, str], ...]:
        """Return registered model keys in stable order.

        Returns
        -------
        tuple of tuple of str
            Lexicographically sorted ``(family, name)`` keys.
        """

        return tuple(sorted(self._registrations))
