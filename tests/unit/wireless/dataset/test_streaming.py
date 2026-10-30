from itertools import islice

from channel_dataset.channels.wireless import StreamingCIRDataset


def test_finite_stream_exhausts_and_repeats_deterministically(dataset_cir):
    finite = StreamingCIRDataset(lambda index: dataset_cir.select_link(index, 0, 0), 3)
    assert len(list(finite)) == 3
    cyclic = StreamingCIRDataset(
        lambda index: dataset_cir.select_link(index, 0, 0), 3, repeat=True
    )
    values = list(islice(cyclic, 5))
    assert [float(value.coefficients[0, 0, 0, 0, 0, 0, 0].real) for value in values] == [1, 2, 3, 1, 2]
