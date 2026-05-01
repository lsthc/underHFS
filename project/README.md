# underHFS Projects

This directory contains separate product-direction starters built on top of the
underHFS framework.

| Project | Purpose | Smoke entrypoint |
| --- | --- | --- |
| `text2image_text2pixel` | Text-conditioned pixel/image tensor generation | `train_text2pixel.py` |
| `text2world` | Text-conditioned simulated world-state generation | `train_text2world.py` |
| `liveSee` | Streaming frame perception and future live video learning | `train_livesee_stream.py` |
| `progamer_rl` | Game policy learning and future RL/self-play | `train_progamer_policy.py` |

Each starter is intentionally tiny, deterministic, and runnable on the portable
underHFS runtime. They are not examples; they are project seeds for growing the
separate model families.
