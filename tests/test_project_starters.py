from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load(path: str):
    module_path = ROOT / path
    spec = importlib.util.spec_from_file_location(module_path.stem, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_text2pixel_project_smoke():
    module = _load("project/text2image_text2pixel/train_text2pixel.py")
    report = module.run_smoke(steps=2)
    assert report["project"] == "text2image_text2pixel"
    assert report["pixel_shape"] == [1, 4, 4]
    assert len(report["generated_pixels"]) == 4


def test_text2world_project_smoke():
    module = _load("project/text2world/train_text2world.py")
    report = module.run_smoke(steps=2)
    assert report["project"] == "text2world"
    assert report["schema"] == ["agent_x", "agent_y", "goal_x", "goal_y", "resource", "hazard"]
    assert len(report["world_vector"]) == 6


def test_livesee_project_smoke():
    module = _load("project/liveSee/train_live_world.py")
    report = module.run_smoke(steps=2)
    assert report["project"] == "liveSee"
    assert report["task"] == "ai_live_world_generation_and_play"
    assert report["controls"] == "WASD"
    assert "@" in report["ascii_view"]


def test_livesee_play_server_state_and_wasd_move():
    module = _load("project/liveSee/play_live_world.py")
    session = module.LiveWorldSession(steps=1)
    before = session.state()
    after = session.move("d")
    assert before["player"] == {"x": 0, "y": 0}
    assert after["player"] == {"x": 1, "y": 0}
    assert len(after["tiles"]) == 11


def test_progamer_project_smoke():
    module = _load("project/progamer_rl/train_progamer_policy.py")
    report = module.run_smoke(steps=2)
    assert report["project"] == "progamer_rl"
    assert len(report["policy"]) == 9
    assert len(report["oracle"]) == 9
