<div align="center">
  <img src="BetaGo.png" alt="BetaGo" width="720">
</div>

# BetaGo underHFS Test Project

BetaGo trains a small 9x9 Baduk policy/value agent using only underHFS runtime
components. It is a test project, not a professional-strength Go engine.

The training data is generated from deterministic rule-aware and strategy-aware
positions. The labels combine tactical capture/atari defense with connection,
extension, corner, side, center, and influence heuristics. The report records the
exact scope and limitations so the result is not overstated.

## Run

```powershell
python project\BetaGo\train_betago_agent.py
```

Outputs are written to:

- `project/BetaGo/artifacts/betago_agent_state.json`
- `project/BetaGo/artifacts/metrics.json`
- `project/BetaGo/reports/report.md`
