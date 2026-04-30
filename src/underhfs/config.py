from __future__ import annotations

from dataclasses import asdict, dataclass, field

from underhfs.compile import CompilePolicy
from underhfs.cuda import RuntimePolicy
from underhfs.distributed import DistributedPolicy


@dataclass
class UnderHFSConfig:
    runtime: RuntimePolicy = field(default_factory=RuntimePolicy)
    compile: CompilePolicy = field(default_factory=CompilePolicy)
    distributed: DistributedPolicy = field(default_factory=DistributedPolicy)

    def to_dict(self) -> dict:
        return asdict(self)
