from underhfs import tensor
from underhfs.cli import main
from underhfs.diagnostics import doctor
from underhfs.tensor import DType


def test_inplace_autograd_version_guard():
    x = tensor(2.0, requires_grad=True)
    y = x * x
    x.add_(1.0)
    try:
        y.backward()
    except RuntimeError as exc:
        assert "in-place mutation" in str(exc)
    else:
        raise AssertionError("expected in-place mutation guard to fail")


def test_dtype_mismatch_fails_explicitly():
    left = tensor([1.0, 2.0], dtype=DType.FP32)
    right = tensor([1.0, 2.0], dtype=DType.FP16)
    try:
        _ = left + right
    except RuntimeError as exc:
        assert "different dtypes" in str(exc)
    else:
        raise AssertionError("expected dtype mismatch to fail explicitly")


def test_doctor_report_and_cli():
    report = doctor()
    assert report.python
    assert "git" in report.tools
    assert isinstance(report.native_cudnn, bool)
    assert isinstance(report.native_nccl, bool)
    assert "onnx" in report.optional_dependencies
    assert "grpcio" in report.optional_dependencies
    assert "opencv-python" in report.optional_dependencies
    assert "websockets" in report.optional_dependencies
    assert isinstance(report.cuda_devices, list)
    assert "vram" in report.memory_budgets or not report.cuda_visible
    payload = report.to_dict()
    assert "native_cudnn" in payload
    assert "optional_dependencies" in payload
    assert main(["doctor"]) == 0
