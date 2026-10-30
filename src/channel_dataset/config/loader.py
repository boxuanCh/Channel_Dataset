"""Load the common configuration envelope from YAML or JSON."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from hashlib import sha256
from math import isfinite
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import yaml

from channel_dataset.core.context import ExecutionContext

from .profiles import resolve_model_profile

if TYPE_CHECKING:
    from channel_dataset.core.realization import RealizationMetadata


def _freeze_json(value: Any, field_name: str) -> Any:
    if isinstance(value, Mapping):
        frozen: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{field_name} keys must be strings")
            frozen[key] = _freeze_json(item, field_name)
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item, field_name) for item in value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not isfinite(value):
            raise ValueError(f"{field_name} must contain only finite numbers")
        return value
    raise TypeError(f"{field_name} must contain only JSON-compatible values")


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class ExecutionConfig:
    """Validated execution settings kept separate from model physics.

    Parameters
    ----------
    backend : {"torch", "numpy"}
        Numerical backend name.
    precision : {"single", "double"}
        Real/complex precision pair used by public arrays.
    device : str
        Backend-native device string. NumPy accepts only ``"cpu"``.
    seed : int
        Nonnegative root seed used to construct a request random stream.

    Raises
    ------
    TypeError
        If a textual field is not a string or ``seed`` is not an integer.
    ValueError
        If a value is outside its supported domain.
    """

    backend: str
    precision: str
    device: str
    seed: int

    def __post_init__(self) -> None:
        if not isinstance(self.backend, str):
            raise TypeError("execution.backend must be a string")
        if not isinstance(self.precision, str):
            raise TypeError("execution.precision must be a string")
        if not isinstance(self.device, str):
            raise TypeError("execution.device must be a string")
        ExecutionContext(self.backend, self.precision, self.device)
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("execution.seed must be an integer")
        if self.seed < 0:
            raise ValueError("execution.seed must be a nonnegative integer")

    def context(self) -> ExecutionContext:
        """Create the corresponding immutable execution context.

        Returns
        -------
        ExecutionContext
            Context containing backend, precision, and device. The seed is
            deliberately excluded and remains available on this object.
        """

        return ExecutionContext(self.backend, self.precision, self.device)

    def to_document(self) -> dict[str, Any]:
        """Return a JSON-serializable execution mapping.

        Returns
        -------
        dict
            Backend, precision, device, and seed values.
        """

        return {
            "backend": self.backend,
            "precision": self.precision,
            "device": self.device,
            "seed": self.seed,
        }


@dataclass(frozen=True, slots=True)
class ConfigEnvelope:
    """Validated common configuration envelope.

    Parameters
    ----------
    schema_version : int
        Persistent configuration schema version.
    family, name : str
        Stable model registry key.
    parameters : mapping
        Model-owned physical parameters. A registered parser converts this
        mapping into a typed immutable configuration before construction.
    execution : ExecutionConfig, optional
        Backend, precision, device, and seed settings.
    metadata : mapping, optional
        Non-physical experiment labels.
    profile : str, optional
        Source profile reference. ``parameters`` always contains its fully
        resolved values plus envelope overrides.

    Notes
    -----
    ``parameters`` and ``metadata`` are copied into read-only mapping proxies.
    Execution settings are excluded from :attr:`model_digest`.
    """

    schema_version: int
    family: str
    name: str
    parameters: Mapping[str, Any]
    execution: ExecutionConfig | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    profile: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or self.schema_version != 1:
            raise ValueError(f"unsupported schema_version: {self.schema_version!r}")
        if not isinstance(self.family, str) or not self.family:
            raise ValueError("model.family must be a non-empty string")
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("model.name must be a non-empty string")
        if not isinstance(self.parameters, Mapping):
            raise TypeError("model.parameters must be a mapping")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping")
        if self.profile is not None and (not isinstance(self.profile, str) or not self.profile):
            raise ValueError("model.profile must be a non-empty string")
        object.__setattr__(self, "parameters", _freeze_json(self.parameters, "model.parameters"))
        object.__setattr__(self, "metadata", _freeze_json(self.metadata, "metadata"))

    @property
    def model_digest(self) -> str:
        """Return a stable digest of the physical model configuration.

        Returns
        -------
        str
            Lowercase SHA-256 hexadecimal digest of schema version, model
            family/name, and model parameters with canonical key ordering.

        Raises
        ------
        ValueError
            If a parameter contains a non-finite float that JSON cannot
            serialize under the canonical representation.
        TypeError
            If parameters contain values unsupported by JSON.
        """

        document = {
            "schema_version": self.schema_version,
            "model": {
                "family": self.family,
                "name": self.name,
                "parameters": _thaw_json(self.parameters),
            },
        }
        payload = json.dumps(
            document, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        )
        return sha256(payload.encode("utf-8")).hexdigest()

    def to_document(self) -> dict[str, Any]:
        """Return a portable, fully resolved configuration document.

        Returns
        -------
        dict
            JSON-serializable schema-version-1 envelope. Profile references
            are intentionally replaced by resolved parameters so the result
            remains reproducible if a resource later moves.
        """

        document: dict[str, Any] = {
            "schema_version": self.schema_version,
            "model": {
                "family": self.family,
                "name": self.name,
                "parameters": _thaw_json(self.parameters),
            },
        }
        if self.execution is not None:
            document["execution"] = self.execution.to_document()
        if self.metadata:
            document["metadata"] = _thaw_json(self.metadata)
        return document

    @classmethod
    def from_realization_metadata(cls, metadata: RealizationMetadata) -> ConfigEnvelope:
        """Reconstruct the physical configuration recorded by a realization.

        Parameters
        ----------
        metadata : RealizationMetadata
            Provenance containing model identity, schema version, resolved
            parameters, and their digest.

        Returns
        -------
        ConfigEnvelope
            Model-only envelope with no execution settings.

        Raises
        ------
        ValueError
            If the reconstructed configuration does not match the recorded
            digest, indicating corrupt or inconsistent provenance.
        """

        envelope = cls(
            schema_version=metadata.schema_version,
            family=metadata.model_family,
            name=metadata.model_name,
            parameters=metadata.resolved_configuration,
        )
        if envelope.model_digest != metadata.config_digest:
            raise ValueError("realization configuration does not match config_digest")
        return envelope


def _parse_execution(value: Any) -> ExecutionConfig | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise TypeError("execution must be a mapping")
    expected = {"backend", "precision", "device", "seed"}
    unknown = set(value) - expected
    if unknown:
        raise ValueError(f"unknown execution fields: {sorted(unknown)}")
    missing = expected - set(value)
    if missing:
        raise ValueError(f"missing execution fields: {sorted(missing)}")
    return ExecutionConfig(
        backend=value["backend"],
        precision=value["precision"],
        device=value["device"],
        seed=value["seed"],
    )


def parse_config_document(
    document: Any, *, base_directory: str | Path | None = None
) -> ConfigEnvelope:
    """Validate an already-decoded YAML or JSON configuration document.

    Parameters
    ----------
    document : mapping
        Decoded configuration with ``schema_version`` and ``model`` fields.
        Optional top-level fields are ``execution`` and ``metadata``.
    base_directory : path-like, optional
        Base for relative ``file:`` profile references. It is supplied
        automatically by :func:`load_config`.

    Returns
    -------
    ConfigEnvelope
        Strict, immutable representation of the common envelope.

    Raises
    ------
    TypeError
        If the root, model, parameters, execution, or metadata values have
        incompatible types.
    ValueError
        If required values are missing, fields are unknown, or schema/execution
        values are invalid.
    """

    if not isinstance(document, dict):
        raise TypeError("configuration root must be a mapping")
    unknown_root_fields = set(document) - {"schema_version", "model", "execution", "metadata"}
    if unknown_root_fields:
        raise ValueError(f"unknown root fields: {sorted(unknown_root_fields)}")
    schema_version = document.get("schema_version")
    model = document.get("model")
    if isinstance(schema_version, bool) or schema_version != 1:
        raise ValueError(f"unsupported schema_version: {schema_version!r}")
    if not isinstance(model, dict):
        raise TypeError("model must be a mapping")
    unknown_model_fields = set(model) - {"family", "name", "parameters", "profile"}
    if unknown_model_fields:
        raise ValueError(f"unknown model fields: {sorted(unknown_model_fields)}")

    family = model.get("family")
    name = model.get("name")
    profile = model.get("profile")
    has_parameters = "parameters" in model
    parameters = model.get("parameters", {})
    execution = _parse_execution(document.get("execution"))
    metadata = document.get("metadata", {})
    if not isinstance(family, str) or not family:
        raise ValueError("model.family must be a non-empty string")
    if not isinstance(name, str) or not name:
        raise ValueError("model.name must be a non-empty string")
    if not has_parameters and profile is None:
        raise ValueError("model must define parameters, profile, or both")
    if not isinstance(parameters, dict):
        raise TypeError("model.parameters must be a mapping")
    if profile is not None and (not isinstance(profile, str) or not profile):
        raise ValueError("model.profile must be a non-empty string")
    if not isinstance(metadata, dict):
        raise TypeError("metadata must be a mapping")

    if profile is not None:
        parameters = resolve_model_profile(
            profile,
            family=family,
            name=name,
            overrides=parameters,
            base_directory=base_directory,
        )

    return ConfigEnvelope(
        schema_version=schema_version,
        family=family,
        name=name,
        parameters=parameters,
        execution=execution,
        metadata=metadata,
        profile=profile,
    )


def load_config(path: str | Path) -> ConfigEnvelope:
    """Load and validate a YAML or JSON configuration file.

    Parameters
    ----------
    path : str or pathlib.Path
        File ending in ``.yaml``, ``.yml``, or ``.json``.

    Returns
    -------
    ConfigEnvelope
        Validated common envelope. Model-specific parameter validation occurs
        when the model registry constructs the model.

    Raises
    ------
    OSError
        If the file cannot be opened.
    ValueError
        If the extension or document values are unsupported.
    TypeError
        If decoded fields have incompatible types.

    Notes
    -----
    YAML is parsed with :func:`yaml.safe_load`; arbitrary constructors are not
    enabled.
    """

    config_path = Path(path).resolve()
    suffix = config_path.suffix.lower()
    if suffix not in {".yaml", ".yml", ".json"}:
        raise ValueError(f"unsupported configuration format: {suffix}")
    with config_path.open("r", encoding="utf-8") as stream:
        if suffix in {".yaml", ".yml"}:
            document = yaml.safe_load(stream)
        elif suffix == ".json":
            document = json.load(stream)
    return parse_config_document(document, base_directory=config_path.parent)
