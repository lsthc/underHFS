# underHFS liveSee starter

This project starts the real-time streaming perception branch of underHFS.

The smoke task trains a tiny frame classifier on synthetic stream features.
The serving layer already has file, OpenCV, FFmpeg, WebSocket, HTTP, and gRPC
paths; this project is the model-training side that can later consume those
frames directly.

Run:

```powershell
$env:PYTHONPATH = "src"
python project\liveSee\train_livesee_stream.py
```
