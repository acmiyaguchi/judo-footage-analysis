from pathlib import Path


def ensure_path(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_parent(path):
    path = Path(path)
    ensure_path(path.parent)
    return path
