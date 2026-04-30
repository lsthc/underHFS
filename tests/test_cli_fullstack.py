from pathlib import Path

from underhfs.cli import main
from underhfs.serialization import load_checkpoint, load_manifest


def test_checkpoint_dataset_and_serve_cli(tmp_path=None):
    root = Path(".underhfs-test") if tmp_path is None else tmp_path
    root.mkdir(exist_ok=True)
    checkpoint = root / "tiny.uhfs.json"
    dataset = root / "sample.txt"
    manifest = root / "tiny.export.json"

    assert main(["checkpoint", "save-smoke", str(checkpoint)]) == 0
    payload = load_checkpoint(checkpoint)
    assert payload["metadata"]["model"] == "TransformerLM"
    assert payload["state"]
    assert main(["checkpoint", "inspect", str(checkpoint)]) == 0
    assert main(["dataset", str(dataset), "--sample"]) == 0
    assert dataset.exists()
    assert main(["bench", "--size", "2", "--iterations", "1", "--warmup", "0", "--no-cuda"]) == 0
    assert main(["serve", "--smoke", "--prompt", "hi"]) == 0
    assert main(["export", str(manifest)]) == 0
    exported = load_manifest(manifest)
    assert exported["model"] == "TransformerLM"
    assert exported["parameters"]

    if tmp_path is None:
        checkpoint.unlink()
        dataset.unlink()
        manifest.unlink()
        root.rmdir()
