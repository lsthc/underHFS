from underhfs.benchmarks import run_microbenchmarks


def test_microbenchmarks_return_cpu_results():
    results = run_microbenchmarks(size=2, iterations=1, warmup=1, include_cuda=False)
    payload = [result.to_dict() for result in results]
    assert [item["name"] for item in payload] == ["add", "matmul"]
    assert all(item["backend"] == "cpu" for item in payload)
    assert all(item["ops_per_second"] > 0 for item in payload)
