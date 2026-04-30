from underhfs.cli import main


def test_train_smoke_cli():
    assert main(["train", "--smoke", "--steps", "1"]) == 0
