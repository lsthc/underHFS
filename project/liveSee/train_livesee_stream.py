from train_live_world import run_smoke


if __name__ == "__main__":
    import json

    print(json.dumps(run_smoke(write_artifacts=True), indent=2))
