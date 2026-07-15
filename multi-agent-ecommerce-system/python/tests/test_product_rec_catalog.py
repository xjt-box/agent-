"""Tests that product recall reads candidates from SQLite catalog data."""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agents.product_rec_agent import ProductRecAgent
from models.schemas import UserProfile
from services.catalog_repository import CatalogRepository, SeedProduct


def test_recall_uses_catalog_repository() -> None:
    with tempfile.TemporaryDirectory() as directory:
        database_path = Path(directory) / "catalog.db"
        repository = CatalogRepository(f"sqlite:///{database_path}")
        repository.initialize(
            [
                SeedProduct("DB001", "Database Phone", "phones", 1000, "Demo", "S01", 5, ["new"]),
                SeedProduct("DB002", "Database Tablet", "tablets", 800, "Demo", "S02", 8, ["study"]),
            ]
        )
        agent = ProductRecAgent.__new__(ProductRecAgent)
        agent.catalog_repository = repository
        agent.vector_store = None
        try:
            profile = UserProfile(user_id="user", preferred_categories=["tablets"])
            products = asyncio.run(agent._recall(profile, 10))
            assert [product.product_id for product in products] == ["DB002", "DB001"]
        finally:
            repository.close()


if __name__ == "__main__":
    test_recall_uses_catalog_repository()
    print("Product recommendation catalog test passed!")