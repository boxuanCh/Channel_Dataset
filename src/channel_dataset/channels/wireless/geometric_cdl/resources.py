"""Version-aware standard-resource resolution for geometric CDL."""

from __future__ import annotations

from typing import Any

from channel_dataset.config import load_resource_document, resource_checksum

_ROOT = "standards/wireless/tr_38_901"
_VERSIONS = {"16.1.0": "v16_1", "19.2.0": "v19_2"}
_ALIASED = {
    "cdl_a.yaml", "cdl_b.yaml", "cdl_c.yaml", "cdl_d.yaml", "cdl_e.yaml",
    "scenario_geometry.yaml", "scenario_large_scale.yaml", "scenario_rays.yaml",
}


def load_geometric_resource(standard_version: str, name: str) -> dict[str, Any]:
    """Load a versioned resource, verifying explicit v19.2 aliases."""

    try:
        namespace = _VERSIONS[standard_version]
    except KeyError as error:
        raise ValueError(f"unsupported standard version: {standard_version}") from error
    if not isinstance(name, str) or not name or "/" in name or "\\" in name:
        raise ValueError("resource name must be one package-local filename")
    if standard_version == "16.1.0" or name == "geometric_evolution.yaml":
        return load_resource_document(f"{_ROOT}/{namespace}/{name}")
    if name not in _ALIASED:
        raise ValueError(f"resource is not declared for v19.2: {name}")
    manifest = load_resource_document(f"{_ROOT}/v19_2/resource_manifest.yaml")
    aliases = manifest.get("aliases")
    if not isinstance(aliases, dict) or name not in aliases:
        raise ValueError(f"v19.2 manifest is missing alias: {name}")
    alias = aliases[name]
    if not isinstance(alias, dict) or set(alias) != {"target", "target_sha256"}:
        raise ValueError(f"invalid v19.2 alias descriptor: {name}")
    target = f"{_ROOT}/v16_1/{name}"
    if resource_checksum(target) != alias["target_sha256"]:
        raise ValueError(f"v19.2 alias checksum mismatch: {name}")
    return load_resource_document(target)


__all__ = ["load_geometric_resource"]
