import json
from pathlib import Path
from typing import Dict


def save_json(file_path: Path, data: Dict) -> None:
    """Save data results as JSON.

    Args:
        file_path: path to save data
        data: results to save
    """
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)


def load_json(file_path: Path) -> dict:
    """Load JSON file from filepath.

    Args:
        file_path: path to JSON

    Returns:
        dict object of loaded JSON file
    """
    with open(file_path, "r") as f:
        data = json.load(f)

    return data


def make_dir(dir: Path | str) -> Path:
    """Make directory.

    Args:
        dir: desired directory to initialize

    Returns:
        Initialized directory path
    """
    if isinstance(dir, str):
        dir = Path(dir)
    dir.mkdir(exist_ok=True, parents=True)

    return dir
