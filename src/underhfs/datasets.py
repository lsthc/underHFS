from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TextDatasetReport:
    path: str
    bytes: int
    characters: int
    lines: int
    unique_bytes: int

    def to_dict(self) -> dict[str, int | str]:
        return {
            "path": self.path,
            "bytes": self.bytes,
            "characters": self.characters,
            "lines": self.lines,
            "unique_bytes": self.unique_bytes,
        }


def inspect_text_dataset(path: str | Path) -> TextDatasetReport:
    source = Path(path)
    raw = source.read_bytes()
    text = raw.decode("utf-8", errors="ignore")
    return TextDatasetReport(
        path=str(source),
        bytes=len(raw),
        characters=len(text),
        lines=0 if not text else text.count("\n") + (0 if text.endswith("\n") else 1),
        unique_bytes=len(set(raw)),
    )


def write_sample_text_dataset(path: str | Path) -> TextDatasetReport:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("underHFS trains beyond the box.\n", encoding="utf-8")
    return inspect_text_dataset(target)
