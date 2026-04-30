from __future__ import annotations

from time import perf_counter

from underhfs import tensor
from underhfs.functional import cross_entropy
from underhfs.nn import TransformerLM
from underhfs.optim import SGD


def run() -> None:
    model = TransformerLM(vocab_size=16, max_seq_len=8, features=8, hidden_features=16, layers=1)
    opt = SGD(model.parameters(), lr=1e-3)
    tokens = tensor([0, 1, 2, 3, 4, 5, 6, 7])
    targets = tensor([1, 2, 3, 4, 5, 6, 7, 8])
    steps = 10
    start = perf_counter()
    last_loss = 0.0
    for _ in range(steps):
        opt.zero_grad()
        loss = cross_entropy(model(tokens), targets)
        last_loss = loss.item()
        loss.backward()
        opt.step()
    elapsed = perf_counter() - start
    print(
        {
            "steps": steps,
            "seconds": elapsed,
            "steps_per_second": steps / elapsed,
            "last_loss": last_loss,
        }
    )


if __name__ == "__main__":
    run()
