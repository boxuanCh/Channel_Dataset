from channel_dataset.channels.wireless import StreamingCIRDataset, shuffle_stream


def test_shuffle_is_seeded_and_preserves_items(backend, dataset_cir):
    dataset = StreamingCIRDataset(lambda index: dataset_cir.select_link(index, 0, 0), 3)
    first = list(shuffle_stream(dataset, 2, backend.random_stream(5)))
    second = list(shuffle_stream(dataset, 2, backend.random_stream(5)))
    extract = lambda values: [float(value.coefficients[0, 0, 0, 0, 0, 0, 0].real) for value in values]
    assert extract(first) == extract(second)
    assert sorted(extract(first)) == [1.0, 2.0, 3.0]
