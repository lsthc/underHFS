from __future__ import annotations

import argparse
import json
from pathlib import Path

from underhfs import __version__
from underhfs.cuda import device_count, is_available
from underhfs.native import status as native_status
from underhfs.testing import run_test_functions


def _cmd_init(args: argparse.Namespace) -> int:
    root = Path(args.path)
    root.mkdir(parents=True, exist_ok=True)
    (root / "underhfs.yaml").write_text("runtime:\n  device: cuda:0\n", encoding="utf-8")
    print(f"initialized {root}")
    return 0


def _cmd_bench(args: argparse.Namespace) -> int:
    native = native_status()
    payload = {
        "underhfs": __version__,
        "cuda_visible": is_available(),
        "cuda_device_count": device_count(),
        "native_core": native.available,
        "native_reason": native.reason,
        "note": "native CUDA microbenchmarks are pending CUDA Toolkit installation",
    }
    print(json.dumps(payload, indent=2))
    return 0


def _cmd_test(_: argparse.Namespace) -> int:
    modules = [
        "tests.test_tensor_autograd",
        "tests.test_nn_optim",
        "tests.test_runtime_surface",
        "tests.test_losses_serialization",
    ]
    passed = run_test_functions(modules)
    for name in passed:
        print(f"{name} OK")
    print(f"{len(passed)} tests passed")
    return 0


def _not_implemented(name: str):
    def run(_: argparse.Namespace) -> int:
        print(f"underhfs {name}: command surface is reserved; implementation is pending")
        return 2

    return run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="underhfs")
    parser.add_argument("--version", action="version", version=f"underhfs {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    init = sub.add_parser("init")
    init.add_argument("path", nargs="?", default=".")
    init.set_defaults(func=_cmd_init)

    bench = sub.add_parser("bench")
    bench.set_defaults(func=_cmd_bench)

    test = sub.add_parser("test")
    test.set_defaults(func=_cmd_test)

    for name in ("train", "serve", "dataset", "checkpoint", "export"):
        command = sub.add_parser(name)
        command.set_defaults(func=_not_implemented(name))

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
