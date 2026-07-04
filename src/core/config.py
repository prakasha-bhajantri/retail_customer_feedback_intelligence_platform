from pathlib import Path
import yaml
import os

ROOT_DIR = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT_DIR / "configs"


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def merge(a, b):
    for k, v in b.items():
        if isinstance(v, dict):
            a[k] = merge(a.get(k, {}), v)
        else:
            a[k] = v
    return a


base = load_yaml(CONFIG_DIR / "base.yaml")

env = os.getenv("APP_ENV", "dev")

override = load_yaml(CONFIG_DIR / f"{env}.yaml")

config = merge(base, override)

# ---------------------------------------------------
# Resolve project-relative paths
# ---------------------------------------------------

config["models"]["sentiment"] = str(
    ROOT_DIR / config["models"]["sentiment"]
)

config["models"]["ner"] = str(
    ROOT_DIR / config["models"]["ner"]
)