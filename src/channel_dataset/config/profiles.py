"""Typed model profiles, inheritance, and deterministic overrides."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from channel_dataset.config.resources import _load_resource


def _merge_parameters(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge mappings while replacing every non-mapping value."""

    merged = dict(base)
    for key, value in override.items():
        inherited = merged.get(key)
        if isinstance(inherited, Mapping) and isinstance(value, Mapping):
            merged[key] = _merge_parameters(inherited, value)
        else:
            merged[key] = value
    return merged


@dataclass(frozen=True, slots=True)
class ModelProfile:
    """Validated model-parameter profile.

    Parameters
    ----------
    schema_version : int
        Profile schema version. Phase 3 supports version 1.
    family, name : str
        Model registry identity to which the profile applies.
    parameters : mapping
        Parameter values contributed by this profile.
    extends : str, optional
        Parent resource reference resolved before these parameters.
    """

    schema_version: int
    family: str
    name: str
    parameters: Mapping[str, Any]
    extends: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "parameters", MappingProxyType(dict(self.parameters)))


def _parse_profile(document: Any) -> ModelProfile:
    if not isinstance(document, dict):
        raise TypeError("profile root must be a mapping")
    expected = {"schema_version", "family", "name", "parameters", "extends"}
    unknown = set(document) - expected
    if unknown:
        raise ValueError(f"unknown profile fields: {sorted(unknown)}")
    required = {"schema_version", "family", "name", "parameters"}
    missing = required - set(document)
    if missing:
        raise ValueError(f"missing profile fields: {sorted(missing)}")
    if isinstance(document["schema_version"], bool) or document["schema_version"] != 1:
        raise ValueError(f"unsupported profile schema_version: {document['schema_version']!r}")
    family = document["family"]
    name = document["name"]
    parameters = document["parameters"]
    extends = document.get("extends")
    if not isinstance(family, str) or not family:
        raise ValueError("profile.family must be a non-empty string")
    if not isinstance(name, str) or not name:
        raise ValueError("profile.name must be a non-empty string")
    if not isinstance(parameters, dict):
        raise TypeError("profile.parameters must be a mapping")
    if extends is not None and (not isinstance(extends, str) or not extends):
        raise ValueError("profile.extends must be a non-empty string")
    return ModelProfile(1, family, name, parameters, extends)


def resolve_model_profile(
    reference: str,
    *,
    family: str,
    name: str,
    overrides: Mapping[str, Any] | None = None,
    base_directory: str | Path | None = None,
) -> dict[str, Any]:
    """Resolve profile inheritance and apply model-level overrides.

    Parameters
    ----------
    reference : str
        Package or file resource reference for the leaf profile.
    family, name : str
        Expected model identity. Every profile in the chain must match it.
    overrides : mapping, optional
        Highest-precedence parameter values from the model envelope.
    base_directory : path-like, optional
        Base for relative ``file:`` references.

    Returns
    -------
    dict
        Fully resolved parameters, independent of the source resources.

    Raises
    ------
    ValueError
        If a profile has the wrong identity, contains unknown fields, or an
        inheritance cycle is detected.
    FileNotFoundError
        If a referenced resource does not exist.

    Notes
    -----
    Parents are resolved first. Child and envelope mappings recursively merge;
    scalar values and sequences replace inherited values. Resolution therefore
    depends only on explicit references and content, never mapping insertion
    order or the current working directory.
    """

    visiting: list[str] = []

    def visit(current_reference: str) -> dict[str, Any]:
        loaded = _load_resource(current_reference, base_directory=base_directory)
        if loaded.identity in visiting:
            cycle = " -> ".join((*visiting, loaded.identity))
            raise ValueError(f"profile inheritance cycle: {cycle}")
        profile = _parse_profile(loaded.document)
        if (profile.family, profile.name) != (family, name):
            raise ValueError(
                "profile model identity does not match envelope: "
                f"{profile.family}/{profile.name} != {family}/{name}"
            )
        visiting.append(loaded.identity)
        try:
            inherited = {} if profile.extends is None else visit(profile.extends)
            return _merge_parameters(inherited, profile.parameters)
        finally:
            visiting.pop()

    resolved = visit(reference)
    return _merge_parameters(resolved, overrides or {})
