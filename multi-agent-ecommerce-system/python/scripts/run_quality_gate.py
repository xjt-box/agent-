"""Run deterministic quality checks and print a concise summary."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_FILES = [
    PROJECT_ROOT / "tests" / "test_ab_test.py",
    PROJECT_ROOT / "tests" / "test_catalog_repository.py",
    PROJECT_ROOT / "tests" / "test_product_rec_catalog.py",
    PROJECT_ROOT / "tests" / "test_inventory_catalog.py",
    PROJECT_ROOT / "tests" / "test_recommendation_eval.py",
    PROJECT_ROOT / "tests" / "test_recommendation_run_store.py",
    PROJECT_ROOT / "tests" / "test_graph_checkpoint.py",
]


def main() -> int:
    passed = 0
    failed = 0

    for test_file in TEST_FILES:
        result = subprocess.run([sys.executable, str(test_file)], cwd=PROJECT_ROOT)
        if result.returncode == 0:
            passed += 1
            print(f"[PASS] {test_file.name}")
        else:
            failed += 1
            print(f"[FAIL] {test_file.name} (exit={result.returncode})")

    print(f"Quality gate summary: passed={passed} failed={failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())