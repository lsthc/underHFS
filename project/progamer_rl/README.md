# underHFS progamer / RL starter

This project starts the game-learning branch of underHFS.

The smoke task trains a tiny policy network on a 3x3 gridworld oracle. It is a
supervised imitation-learning seed now, and can grow toward self-play,
reinforcement learning, replay buffers, and distributed rollout workers.

Run:

```powershell
$env:PYTHONPATH = "src"
python project\progamer_rl\train_progamer_policy.py
```
