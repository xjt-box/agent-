"""Direct tests for the SQLite catalog and inventory repository."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from services.catalog_repository import CatalogRepository


def create_repository(directory: Path) -> CatalogRepository:
    database_path = directory / "catalog.db"
    repository = CatalogRepository(f"sqlite:///{database_path}")
    try:
        repository.initialize()
    except Exception:
        repository.close()
        raise
    return repository


def test_seed_is_idempotent() -> None:
    with tempfile.TemporaryDirectory() as directory:
        repository = create_repository(Path(directory))
        try:
            assert repository.initialize() == 0
            assert len(repository.list_products()) == 6
        finally:
            repository.close()


def test_stock_update_is_persisted() -> None:
    with tempfile.TemporaryDirectory() as directory:
        repository = create_repository(Path(directory))
        try:
            repository.set_stock("P001", 0)
            assert repository.get_stock(["P001", "P999"]) == {"P001": 0}
        finally:
            repository.close()


def test_invalid_stock_is_rejected() -> None:
    with tempfile.TemporaryDirectory() as directory:
        repository = create_repository(Path(directory))
        try:
            try:
                repository.set_stock("P001", -1)
            except ValueError as error:
                assert "cannot be negative" in str(error)
            else:
                raise AssertionError("Negative stock was accepted.")
        finally:
            repository.close()


if __name__ == "__main__":
    test_seed_is_idempotent()
    test_stock_update_is_persisted()
    test_invalid_stock_is_rejected()
    print("All catalog repository tests passed!")