from __future__ import annotations

import argparse
import json
from pathlib import Path

from underhfs import __version__
from underhfs.benchmarks import run_benchmark_suite, run_memory_benchmark, run_microbenchmarks
from underhfs.cuda import allocator_stats, device_count, devices, is_available, memory_budgets, stream_stats
from underhfs.datasets import inspect_text_dataset, write_sample_text_dataset
from underhfs.diagnostics import doctor
from underhfs.functional import cross_entropy
from underhfs.native import status as native_status
from underhfs.nn import TransformerLM
from underhfs.optim import SGD
from underhfs.serialization import (
    export_onnx,
    export_manifest,
    load_binary_state_dict,
    load_checkpoint,
    save_binary_state_dict,
    save_checkpoint,
)
from underhfs.serve import ServeConfig, serve, serve_http
from underhfs.tensor import tensor
from underhfs.text import ByteTokenizer
from underhfs.testing import run_test_functions


def _cmd_init(args: argparse.Namespace) -> int:
    root = Path(args.path)
    root.mkdir(parents=True, exist_ok=True)
    (root / "underhfs.yaml").write_text("runtime:\n  device: cuda:0\n", encoding="utf-8")
    print(f"initialized {root}")
    return 0


def _cmd_bench(args: argparse.Namespace) -> int:
    native = native_status()
    results = run_microbenchmarks(
        size=args.size,
        iterations=args.iterations,
        warmup=args.warmup,
        include_cuda=not args.no_cuda,
    )
    cuda_runtime = _cuda_runtime_report(native.cuda_enabled)
    payload = {
        "underhfs": __version__,
        "suite": run_benchmark_suite(
            size=max(2, min(args.size, 16)),
            iterations=max(1, min(args.iterations, 5)),
            warmup=max(0, min(args.warmup, 2)),
            include_cuda=not args.no_cuda,
            include_oracle=not args.no_oracle,
        ).to_dict()
        if args.suite
        else None,
        "cuda_visible": is_available(),
        "cuda_device_count": device_count(),
        "cuda_devices": [device.to_dict() for device in devices()],
        "memory_budgets": {tier.value: size for tier, size in memory_budgets().items()},
        "native_core": native.available,
        "native_reason": native.reason,
        "cuda_runtime": cuda_runtime,
        "results": [result.to_dict() for result in results],
        "memory": run_memory_benchmark().to_dict(),
    }
    print(json.dumps(payload, indent=2))
    return 0


def _cuda_runtime_report(cuda_enabled: bool) -> dict:
    if not cuda_enabled:
        return {}
    try:
        return {"allocator": allocator_stats(), "stream": stream_stats()}
    except RuntimeError:
        return {}


def _cmd_doctor(_: argparse.Namespace) -> int:
    print(json.dumps(doctor().to_dict(), indent=2))
    return 0


def _cmd_test(_: argparse.Namespace) -> int:
    modules = [
        "tests.test_tensor_autograd",
        "tests.test_nn_optim",
        "tests.test_runtime_surface",
        "tests.test_losses_serialization",
        "tests.test_transformer_lm",
        "tests.test_cli_train",
        "tests.test_benchmarks",
        "tests.test_text_generation",
        "tests.test_cli_fullstack",
        "tests.test_runtime_memory_planner",
        "tests.test_guardrails",
        "tests.test_native_contract",
        "tests.test_project_starters",
    ]
    passed = run_test_functions(modules)
    for name in passed:
        print(f"{name} OK")
    print(f"{len(passed)} tests passed")
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    if not args.smoke:
        print("underhfs train currently supports --smoke for the built-in tiny LM training path")
        return 2

    model = TransformerLM(
        vocab_size=args.vocab_size,
        max_seq_len=args.seq_len,
        features=args.features,
        hidden_features=args.hidden_features,
        layers=args.layers,
    )
    opt = SGD(model.parameters(), lr=args.lr)
    tokens = tensor([i % args.vocab_size for i in range(args.seq_len)])
    targets = tensor([(i + 1) % args.vocab_size for i in range(args.seq_len)])
    losses: list[float] = []

    for _ in range(args.steps):
        opt.zero_grad()
        logits = model(tokens)
        loss = cross_entropy(logits, targets)
        losses.append(loss.item())
        loss.backward()
        opt.step()

    print(
        json.dumps(
            {
                "mode": "smoke",
                "steps": args.steps,
                "initial_loss": losses[0] if losses else None,
                "final_loss": losses[-1] if losses else None,
                "losses": losses,
            },
            indent=2,
        )
    )
    return 0


def _tiny_lm_from_args(args: argparse.Namespace) -> TransformerLM:
    return TransformerLM(
        vocab_size=args.vocab_size,
        max_seq_len=args.seq_len,
        features=args.features,
        hidden_features=args.hidden_features,
        layers=args.layers,
    )


def _cmd_checkpoint(args: argparse.Namespace) -> int:
    path = Path(args.path)
    if args.action == "save-smoke":
        model = _tiny_lm_from_args(args)
        save_checkpoint(
            path,
            state=model.state_dict(),
            metadata={
                "model": "TransformerLM",
                "vocab_size": args.vocab_size,
                "seq_len": args.seq_len,
                "features": args.features,
                "hidden_features": args.hidden_features,
                "layers": args.layers,
            },
        )
        print(json.dumps({"saved": str(path), "parameters": len(model.state_dict())}, indent=2))
        return 0
    if args.action == "inspect":
        payload = load_checkpoint(path)
        print(json.dumps({"path": str(path), "metadata": payload["metadata"], "tensors": len(payload["state"])}, indent=2))
        return 0
    if args.action == "save-binary-smoke":
        model = _tiny_lm_from_args(args)
        save_binary_state_dict(path, model.state_dict())
        print(json.dumps({"saved": str(path), "format": "binary", "parameters": len(model.state_dict())}, indent=2))
        return 0
    if args.action == "inspect-binary":
        state = load_binary_state_dict(path)
        print(json.dumps({"path": str(path), "format": "binary", "tensors": len(state)}, indent=2))
        return 0
    print(f"unsupported checkpoint action: {args.action}")
    return 2


def _cmd_dataset(args: argparse.Namespace) -> int:
    if args.sample:
        report = write_sample_text_dataset(args.path)
    else:
        report = inspect_text_dataset(args.path)
    print(json.dumps(report.to_dict(), indent=2))
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    if not args.smoke and not args.http_smoke:
        print("underhfs serve currently supports --smoke and --http-smoke verification paths")
        return 2
    tokenizer = ByteTokenizer()
    model = TransformerLM(vocab_size=256, max_seq_len=8, features=4, hidden_features=8, layers=1)
    server = serve(lambda payload: tokenizer.decode(model.generate(tokenizer.encode(payload["prompt"]), args.max_new_tokens)))
    if args.http_smoke:
        http = serve_http(
            lambda payload: {"text": server.predict(payload)},
            ServeConfig(host=args.host, port=args.port),
        )
        try:
            http.start()
            print(json.dumps({"url": http.url, "health": "/health", "predict": "/predict"}, indent=2))
        finally:
            http.close()
        return 0
    print(json.dumps({"prompt": args.prompt, "response": server.predict({"prompt": args.prompt})}, indent=2))
    return 0


def _cmd_export(args: argparse.Namespace) -> int:
    model = _tiny_lm_from_args(args)
    if args.format == "onnx":
        actual_format = export_onnx(
            args.path,
            model_name="TransformerLM",
            state=model.state_dict(),
            inputs={"token_ids": {"shape": [args.seq_len], "dtype": "int64"}},
        )
        print(json.dumps({"exported": args.path, "format": actual_format, "model": "TransformerLM"}, indent=2))
        return 0
    export_manifest(
        args.path,
        model_name="TransformerLM",
        state=model.state_dict(),
        inputs={"token_ids": {"shape": [args.seq_len], "dtype": "int64"}},
    )
    print(json.dumps({"exported": args.path, "format": "manifest", "model": "TransformerLM"}, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="underhfs")
    parser.add_argument("--version", action="version", version=f"underhfs {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    init = sub.add_parser("init")
    init.add_argument("path", nargs="?", default=".")
    init.set_defaults(func=_cmd_init)

    bench = sub.add_parser("bench")
    bench.add_argument("--size", type=int, default=32)
    bench.add_argument("--iterations", type=int, default=20)
    bench.add_argument("--warmup", type=int, default=3)
    bench.add_argument("--no-cuda", action="store_true")
    bench.add_argument("--suite", action="store_true")
    bench.add_argument("--no-oracle", action="store_true")
    bench.set_defaults(func=_cmd_bench)

    doctor_cmd = sub.add_parser("doctor")
    doctor_cmd.set_defaults(func=_cmd_doctor)

    test = sub.add_parser("test")
    test.set_defaults(func=_cmd_test)

    train = sub.add_parser("train")
    train.add_argument("--smoke", action="store_true")
    train.add_argument("--steps", type=int, default=3)
    train.add_argument("--seq-len", type=int, default=4)
    train.add_argument("--vocab-size", type=int, default=8)
    train.add_argument("--features", type=int, default=4)
    train.add_argument("--hidden-features", type=int, default=8)
    train.add_argument("--layers", type=int, default=1)
    train.add_argument("--lr", type=float, default=1e-3)
    train.set_defaults(func=_cmd_train)

    checkpoint = sub.add_parser("checkpoint")
    checkpoint.add_argument("action", choices=("save-smoke", "inspect", "save-binary-smoke", "inspect-binary"))
    checkpoint.add_argument("path")
    checkpoint.add_argument("--seq-len", type=int, default=4)
    checkpoint.add_argument("--vocab-size", type=int, default=8)
    checkpoint.add_argument("--features", type=int, default=4)
    checkpoint.add_argument("--hidden-features", type=int, default=8)
    checkpoint.add_argument("--layers", type=int, default=1)
    checkpoint.set_defaults(func=_cmd_checkpoint)

    dataset = sub.add_parser("dataset")
    dataset.add_argument("path")
    dataset.add_argument("--sample", action="store_true")
    dataset.set_defaults(func=_cmd_dataset)

    serve_cmd = sub.add_parser("serve")
    serve_cmd.add_argument("--smoke", action="store_true")
    serve_cmd.add_argument("--http-smoke", action="store_true")
    serve_cmd.add_argument("--host", default="127.0.0.1")
    serve_cmd.add_argument("--port", type=int, default=0)
    serve_cmd.add_argument("--prompt", default="hi")
    serve_cmd.add_argument("--max-new-tokens", type=int, default=2)
    serve_cmd.set_defaults(func=_cmd_serve)

    export = sub.add_parser("export")
    export.add_argument("path")
    export.add_argument("--format", choices=("manifest", "onnx"), default="manifest")
    export.add_argument("--seq-len", type=int, default=4)
    export.add_argument("--vocab-size", type=int, default=8)
    export.add_argument("--features", type=int, default=4)
    export.add_argument("--hidden-features", type=int, default=8)
    export.add_argument("--layers", type=int, default=1)
    export.set_defaults(func=_cmd_export)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
