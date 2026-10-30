import pytest

from benchmarks.benchmark_wireless import benchmark


@pytest.mark.performance
def test_representative_cpu_benchmark_smoke(backend_name):
    elapsed = benchmark(backend_name, iterations=1)
    assert elapsed > 0.0
