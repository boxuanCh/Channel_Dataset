import json
from importlib.resources import files

import jsonschema

from channel_dataset.channels.wireless import save_cir


def test_manifest_validates_against_packaged_schema(tmp_path, dataset_cir):
    destination = save_cir(tmp_path / "realization", dataset_cir)
    manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
    schema_path = files("channel_dataset.resources").joinpath(
        "schema/cir-realization-v1.schema.json"
    )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    jsonschema.validate(manifest, schema)
    assert manifest["metadata"]["stream_id"] == dataset_cir.metadata.stream_id
