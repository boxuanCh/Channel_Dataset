"""A sampled, reusable channel state."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from channel_dataset.core.paths import PathCluster, PropagationPath


def _freeze_mapping_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _freeze_mapping_value(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_mapping_value(item) for item in value)
    return value


def _plain_mapping_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _plain_mapping_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain_mapping_value(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class RealizationMetadata:
    """Reproducibility and configuration provenance for an array realization.

    Parameters
    ----------
    model_family, model_name : str
        Stable model registry identity.
    schema_version : int
        Positive persistent metadata schema version.
    backend : {"torch", "numpy"}
        Backend owning realization arrays.
    precision : {"single", "double"}
        Declared array precision.
    device : str
        Canonical array device.
    stream_id : str
        Root seed and stream-lineage identifier.
    config_digest : str
        SHA-256 digest of resolved physical model settings.
    resolved_configuration : mapping
        Read-only physical model settings used for generation.
    software_version : str
        Package version that produced the realization.
    request_id : str, optional
        Caller-supplied request identifier.
    root_seed : int, optional
        Root random seed when available separately from ``stream_id``.
    model_version, standard_version : str, optional
        Physical model and external-standard versions used for generation.
    """

    model_family: str
    model_name: str
    schema_version: int
    backend: str
    precision: str
    device: str
    stream_id: str
    config_digest: str
    resolved_configuration: Mapping[str, Any]
    software_version: str
    request_id: str | None = None
    root_seed: int | None = None
    model_version: str | None = None
    standard_version: str | None = None

    def __post_init__(self) -> None:
        if self.schema_version <= 0:
            raise ValueError("schema_version must be positive")
        if not self.model_family or not self.model_name:
            raise ValueError("model identity must be non-empty")
        if self.backend not in {"numpy", "torch"}:
            raise ValueError("backend must be 'numpy' or 'torch'")
        if self.precision not in {"single", "double"}:
            raise ValueError("precision must be 'single' or 'double'")
        if len(self.config_digest) != 64:
            raise ValueError("config_digest must be a SHA-256 hexadecimal digest")
        try:
            int(self.config_digest, 16)
        except ValueError as error:
            raise ValueError("config_digest must be a SHA-256 hexadecimal digest") from error
        if self.root_seed is not None and self.root_seed < 0:
            raise ValueError("root_seed must be nonnegative when provided")
        if self.model_version is not None and not self.model_version:
            raise ValueError("model_version must be non-empty when provided")
        if self.standard_version is not None and not self.standard_version:
            raise ValueError("standard_version must be non-empty when provided")
        object.__setattr__(
            self, "resolved_configuration", _freeze_mapping_value(self.resolved_configuration)
        )

    def to_document(self) -> dict[str, Any]:
        """Return JSON-serializable realization provenance.

        Returns
        -------
        dict
            Metadata fields with the resolved configuration converted back to
            ordinary dictionaries and lists.
        """

        return {
            "model_family": self.model_family,
            "model_name": self.model_name,
            "schema_version": self.schema_version,
            "backend": self.backend,
            "precision": self.precision,
            "device": self.device,
            "stream_id": self.stream_id,
            "config_digest": self.config_digest,
            "resolved_configuration": _plain_mapping_value(self.resolved_configuration),
            "software_version": self.software_version,
            "request_id": self.request_id,
            "root_seed": self.root_seed,
            "model_version": self.model_version,
            "standard_version": self.standard_version,
        }


@dataclass(frozen=True, slots=True)
class ChannelRealization:
    """Fixed scalar-path realization sampled from a configured model.

    Parameters
    ----------
    model_family, model_name : str
        Stable model identity.
    paths : tuple of PropagationPath, optional
        Top-level paths.
    clusters : tuple of PathCluster, optional
        Clustered paths.
    metadata : mapping, optional
        Read-only realization provenance.
    """

    model_family: str
    model_name: str
    paths: tuple[PropagationPath, ...] = ()
    clusters: tuple[PathCluster, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "paths", tuple(self.paths))
        object.__setattr__(self, "clusters", tuple(self.clusters))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    def all_paths(self) -> tuple[PropagationPath, ...]:
        """Return all scalar paths as one immutable sequence.

        Returns
        -------
        tuple of PropagationPath
            Top-level paths followed by paths from each cluster in cluster
            order.
        """

        return self.paths + tuple(path for cluster in self.clusters for path in cluster.paths)
