# underHFS liveSee Minecraft-like world starter

This project starts the AI live world generation branch of underHFS.

The liveSee target is an AI-generated Minecraft-like voxel world that can be
viewed and explored in real time. A tiny underHFS model learns text-conditioned
surface blocks, the world generator builds chunked block columns, and a
standard-library Python web server serves a browser canvas where you move with
WASD and turn with Q/E or arrow keys.

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
