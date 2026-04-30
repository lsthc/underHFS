from underhfs.native import probe, status


def test_native_contract_probe_when_available():
    state = status()
    if not state.available:
        assert state.reason
        return
    result = probe()
    assert isinstance(result["cuda_enabled"], bool)
    assert result["add"] == [6.0, 8.0, 10.0, 12.0]
    assert result["matmul"] == [19.0, 22.0, 43.0, 50.0]
    assert result["sum"] == [134.0]
