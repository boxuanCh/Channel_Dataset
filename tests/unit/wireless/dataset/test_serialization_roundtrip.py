from channel_dataset.channels.wireless import load_cir, save_cir


def test_roundtrip_preserves_arrays_and_metadata(tmp_path, dataset_cir, execution_context, backend):
    destination = save_cir(tmp_path / "realization", dataset_cir, chunk_size=2)
    loaded = load_cir(destination, execution_context)
    assert loaded.shape == dataset_cir.shape
    assert loaded.metadata.config_digest == dataset_cir.metadata.config_digest
    assert backend.to_scalar(
        backend.max(backend.abs(loaded.coefficients - dataset_cir.coefficients))
    ) == 0.0
    assert backend.to_scalar(backend.max(backend.abs(loaded.delays_s - dataset_cir.delays_s))) == 0.0


def test_batch_axis_is_chunked(tmp_path, dataset_cir):
    destination = save_cir(tmp_path / "realization", dataset_cir, chunk_size=1)
    assert len(list(destination.glob("coefficients.*.npy"))) == 3
    assert len(list(destination.glob("delays_s.*.npy"))) == 3
