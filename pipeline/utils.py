"""Shared utilities for the adaptation pipeline."""

from __future__ import annotations

import csv
import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)


def delete_run_dir(save_dir: str) -> None:
    path = Path(save_dir)
    if not path.exists():
        return
    try:
        shutil.rmtree(path)
    except Exception as e:
        logger.warning(f"Failed to delete run dir {path}: {e}")


def append_csv_row(csv_path: str, row: list) -> None:
    """Append one row to a CSV file."""
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)
