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
    assert main(["doctor"]) == 0
