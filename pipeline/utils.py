"""Shared utilities for the adaptation pipeline."""

from __future__ import annotations

import csv
import heapq
import logging
import shutil
from pathlib import Path
from typing import List, Tuple

logger = logging.getLogger(__name__)


class TopKWeightsTracker:
    """Keeps the top-k training runs by score, copying their full Ultralytics run dirs.

    Uses a min-heap so the worst-scoring kept run can be evicted in O(log k)
    when a better run arrives.
    """

    def __init__(self, weights_dir: str, k: int):
        self.weights_dir = Path(weights_dir)
        self.k = k
        self._heap: List[Tuple[float, str]] = []  # (score, run_name)

    def consider(self, score: float, run_name: str, src_dir: str) -> bool:
        """Copy src_dir to weights_dir/run_name if score qualifies for top-k."""
        if not Path(src_dir).exists():
            logger.warning(f"TopKWeightsTracker: source dir not found: {src_dir}")
            return False

        if len(self._heap) < self.k:
            heapq.heappush(self._heap, (score, run_name))
            self._copy(run_name, src_dir)
            return True

        if score > self._heap[0][0]:
            _, evicted_name = heapq.heapreplace(self._heap, (score, run_name))
            self._delete(evicted_name)
            self._copy(run_name, src_dir)
            return True

        return False

    def _copy(self, run_name: str, src_dir: str) -> None:
        self.weights_dir.mkdir(parents=True, exist_ok=True)
        dst = self.weights_dir / run_name
        try:
            shutil.copytree(src_dir, dst, dirs_exist_ok=True)
            logger.info(f"  Top-k weights saved: {dst}")
        except Exception as e:
            logger.warning(f"  Failed to copy weights for {run_name}: {e}")

    def _delete(self, run_name: str) -> None:
        dst = self.weights_dir / run_name
        if dst.exists():
            try:
                shutil.rmtree(dst)
                logger.info(f"  Top-k evicted: {dst}")
            except Exception as e:
                logger.warning(f"  Failed to evict {run_name}: {e}")


def cleanup_run_artifacts(save_dir: str) -> None:
    """Delete all Ultralytics run artifacts except train.log."""
    path = Path(save_dir)
    if not path.exists():
        return
    for item in path.iterdir():
        if item.name == "train.log":
            continue
        try:
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
        except Exception as e:
            logger.warning(f"Cleanup failed for {item}: {e}")


def append_csv_row(csv_path: str, row: list) -> None:
    """Append one row to a CSV file."""
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)
