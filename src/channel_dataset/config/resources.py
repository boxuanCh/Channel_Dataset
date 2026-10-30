"""Deterministic loading of configuration resources and external files."""

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from importlib import resources
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

RESOURCE_PACKAGE = "channel_dataset.resources"


@dataclass(frozen=True, slots=True)
class LoadedResource:
    """Decoded configuration resource together with its canonical identity.

    Parameters
    ----------
    reference : str
        User-facing resource reference.
    identity : str
        Canonical identity used for profile-cycle detection.
    document : dict
        Decoded YAML or JSON mapping.
    checksum : str
        SHA-256 checksum of the original resource bytes.
    """

    reference: str
    identity: str
    document: dict[str, Any]
    checksum: str


def _package_path(reference: str) -> PurePosixPath:
    path_text = reference.removeprefix("package://")
    if not path_text or "\\" in path_text:
        raise ValueError("package resource references must use non-empty POSIX paths")
    path = PurePosixPath(path_text)
    if path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise ValueError("package resource references cannot escape the resource package")
    return path


def _resolve_file_path(reference: str, base_directory: str | Path | None) -> Path:
    path_text = reference.removeprefix("file:")
    path = Path(path_text)
    if not path.is_absolute():
        if base_directory is None:
            raise ValueError("relative file resources require base_directory")
        path = Path(base_directory) / path
    return path.resolve()


def _decode_document(text: str, suffix: str, identity: str) -> dict[str, Any]:
    if suffix in {".yaml", ".yml"}:
        document = yaml.safe_load(text)
    elif suffix == ".json":
        document = json.loads(text)
    else:
        raise ValueError(f"unsupported resource format: {suffix}")
    if not isinstance(document, dict):
        raise TypeError(f"configuration resource must contain a mapping: {identity}")
    return document


def _load_resource(
    reference: str, *, base_directory: str | Path | None = None
) -> LoadedResource:
    if not isinstance(reference, str):
        raise TypeError("resource reference must be a string")
    if not reference:
        raise ValueError("resource reference must be non-empty")

    if reference.startswith("file:") or Path(reference).is_absolute():
        path = _resolve_file_path(reference, base_directory)
        raw = path.read_bytes()
        identity = f"file:{path.as_posix()}"
        suffix = path.suffix.lower()
    else:
        path = _package_path(reference)
        resource = resources.files(RESOURCE_PACKAGE).joinpath(*path.parts)
        if not resource.is_file():
            raise FileNotFoundError(f"package resource does not exist: package://{path}")
        raw = resource.read_bytes()
        identity = f"package://{path}"
        suffix = path.suffix.lower()

    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"configuration resource is not UTF-8: {identity}") from error
    return LoadedResource(
        reference=reference,
        identity=identity,
        document=_decode_document(text, suffix, identity),
        checksum=sha256(raw).hexdigest(),
    )


def load_resource_document(
    reference: str, *, base_directory: str | Path | None = None
) -> dict[str, Any]:
    """Load a YAML or JSON document from an installed resource or file.

    Parameters
    ----------
    reference : str
        ``package://`` reference relative to ``channel_dataset.resources``, a
        package-relative path such as ``"profiles/wireless/rayleigh.yaml"``,
        an absolute path, or a ``file:`` reference. Relative file references
        are resolved only against ``base_directory``, never the process CWD.
    base_directory : path-like, optional
        Directory used by relative ``file:`` references.

    Returns
    -------
    dict
        Decoded resource mapping.

    Raises
    ------
    FileNotFoundError
        If the resource does not exist.
    TypeError
        If the reference is not textual or the document root is not a mapping.
    ValueError
        If the reference escapes the package, lacks a required base directory,
        has an unsupported extension, or is not UTF-8.
    """

    return _load_resource(reference, base_directory=base_directory).document


def resource_checksum(reference: str, *, base_directory: str | Path | None = None) -> str:
    """Return the SHA-256 checksum of a package or file resource.

    Parameters
    ----------
    reference : str
        Resource reference accepted by :func:`load_resource_document`.
    base_directory : path-like, optional
        Directory used by relative ``file:`` references.

    Returns
    -------
    str
        Lowercase hexadecimal SHA-256 checksum of the source bytes.
    """

    return _load_resource(reference, base_directory=base_directory).checksum
