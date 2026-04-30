from __future__ import annotations

from time import perf_counter

from underhfs import tensor
from underhfs.nn import Linear
from underhfs.optim import SGD


def run() -> None:
    x = tensor([[1.0, 2.0, 3.0, 4.0]], requires_grad=True)
    model = Linear(4, 4)
    opt = SGD(model.parameters(), lr=1e-3)
    start = perf_counter()
    steps = 100
    for _ in range(steps):
        opt.zero_grad()
        y = model(x)
        loss = (y * y).mean()
        loss.backward()
        opt.step()
    elapsed = perf_counter() - start
    print({"steps": steps, "seconds": elapsed, "steps_per_second": steps / elapsed})


if __name__ == "__main__":
    run()
