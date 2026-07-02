import json
from pathlib import Path
from typing import Any

import dill


def ensure_directory(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_object(file_path: str | Path, obj: Any) -> None:
    file_path = Path(file_path)
    ensure_directory(file_path.parent)
    with file_path.open("wb") as handle:
        dill.dump(obj, handle)


def load_object(file_path: str | Path) -> Any:
    with Path(file_path).open("rb") as handle:
        return dill.load(handle)


def save_json(file_path: str | Path, payload: dict) -> None:
    file_path = Path(file_path)
    ensure_directory(file_path.parent)
    with file_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def load_json(file_path: str | Path) -> dict:
    with Path(file_path).open("r", encoding="utf-8") as handle:
        return json.load(handle)
