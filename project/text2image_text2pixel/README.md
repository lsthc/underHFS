# underHFS text2image / text2pixel starter

This project starts the pixel-generation branch of underHFS.

The first smoke task trains a tiny text-conditioned MLP that maps a prompt into
a 4x4 grayscale pixel tensor. It is intentionally small enough to run on the
portable underHFS runtime while keeping the same shape of a future
text-conditioned image decoder:

- text prompt encoder
- conditional pixel decoder
- MSE reconstruction objective
- artifact metrics and generated pixel tensor

Run:

```powershell
$env:PYTHONPATH = "src"
python project\text2image_text2pixel\train_text2pixel.py
```
