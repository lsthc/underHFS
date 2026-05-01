# underHFS text2world starter

This project starts the world-state generation branch of underHFS.

The smoke task maps a text command into a compact simulated world vector:

- agent x/y
- goal x/y
- resource level
- hazard level

It is a tiny supervised world model now, and gives the later text2world stack a
place to grow into scene graphs, physics state, or simulator control.

Run:

```powershell
$env:PYTHONPATH = "src"
python project\text2world\train_text2world.py
```
