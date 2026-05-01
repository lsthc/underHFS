# underHFS liveSee live world starter

This project starts the AI live world generation branch of underHFS.

The liveSee target is an AI-generated world that can be viewed and explored in
real time. A tiny underHFS model learns a text-conditioned tile world, then a
standard-library Python web server lets you walk through it with WASD.

Train the world generator smoke:

```powershell
$env:PYTHONPATH = "src"
python project\liveSee\train_live_world.py
```

Play locally:

```powershell
$env:PYTHONPATH = "src"
python project\liveSee\play_live_world.py
```
