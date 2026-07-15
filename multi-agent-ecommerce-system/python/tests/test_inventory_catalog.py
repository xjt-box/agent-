"""Integration tests for SQLite-backed inventory filtering."""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agents.inventory_agent import InventoryAgent
from models.schemas import (
    MarketingCopyResult,
    ProductRecResult,
    RecommendationRequest,
    UserProfile,
    UserProfileResult,
)
from orchestrator.supervisor import AgentBundle, SupervisorOrchestrator
from services.catalog_repository import CatalogRepository, SeedProduct


class StaticProfileAgent:
    async def run(self, **kwargs: Any) -> UserProfileResult:
        return UserProfileResult(profile=UserProfile(user_id=kwargs["user_id"]))


class StaticProductAgent:
    def __init__(self, products):
        self.products = products

    async def run(self, **kwargs: Any) -> ProductRecResult:
        return ProductRecResult(products=self.products[:kwargs["num_items"]])


class StaticMarketingAgent:
    async def run(self, **kwargs: Any) -> MarketingCopyResult:
        return MarketingCopyResult(
            copies=[
                {"product_id": product.product_id, "copy": f"Recommend {product.name}."}
                for product in kwargs["products"]
            ]
        )


def test_out_of_stock_product_is_removed_from_recommendations() -> None:
    with tempfile.TemporaryDirectory() as directory:
        database_path = Path(directory) / "catalog.db"
        repository = CatalogRepository(f"sqlite:///{database_path}")
        repository.initialize(
            [
                SeedProduct("DB001", "Available Phone", "phones", 1000, "Demo", "S01", 3, ["new"]),
                SeedProduct("DB002", "Sold Out Tablet", "tablets", 800, "Demo", "S02", 0, ["study"]),
            ]
        )
        inventory_agent = InventoryAgent(catalog_repository=repository)
        products = repository.list_products()
        supervisor = SupervisorOrchestrator(
            agents=AgentBundle(
                user_profile=StaticProfileAgent(),
                product_rec=StaticProductAgent(products),
                marketing_copy=StaticMarketingAgent(),
                inventory=inventory_agent,
            )
        )
        try:
            response = asyncio.run(
                supervisor.recommend(RecommendationRequest(user_id="inventory_test", num_items=2))
            )
            assert [product.product_id for product in response.products] == ["DB001"]
            assert response.marketing_copies[0]["product_id"] == "DB001"
        finally:
            repository.close()


if __name__ == "__main__":
    test_out_of_stock_product_is_removed_from_recommendations()
    print("Inventory catalog integration test passed!")