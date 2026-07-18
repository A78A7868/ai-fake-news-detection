"""
utils.py — shared helpers used across the pipeline.
"""

import sys
from pathlib import Path


def check_raw_data(raw_dir: str | Path = "data/raw") -> tuple[Path, Path]:
    """
    Verify Fake.csv and True.csv exist in raw_dir.
    Exits with a friendly message if either is missing — avoids cryptic FileNotFoundError
    downstream.
    """
    raw = Path(raw_dir)
    fake_path = raw / "Fake.csv"
    true_path = raw / "True.csv"

    missing = []
    if not fake_path.exists():
        missing.append(str(fake_path))
    if not true_path.exists():
        missing.append(str(true_path))

    if missing:
        print("\n[ERROR] The following required data files are missing:")
        for p in missing:
            print(f"  - {p}")
        print(
            "\nPlease download Fake.csv and True.csv from the Kaggle dataset "
            "'Fake and Real News Dataset' and place them in the data/raw/ folder."
        )
        sys.exit(1)

    return fake_path, true_path


def ensure_dirs(*dirs: str | Path) -> None:
    """Create directories (including parents) if they don't exist."""
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
