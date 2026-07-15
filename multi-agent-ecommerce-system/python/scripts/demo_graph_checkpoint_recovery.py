"""Run an end-to-end FastAPI demo of LangGraph checkpoint recovery."""

from __future__ import annotations

import os
import sys
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi.testclient import TestClient

import main
from models.schemas import InventoryResult, MarketingCopyResult, Product, ProductRecResult, UserProfile, UserProfileResult
from orchestrator import graph as recommendation_graph


PRODUCT = Product(product_id="DEMO-001", name="Demo Phone", category="phones", price=3999, stock=10)


class DemoUserProfileAgent:
    async def run(self, **kwargs: Any) -> UserProfileResult:
        return UserProfileResult(profile=UserProfile(user_id=kwargs["user_id"]))


class DemoProductRecAgent:
    async def run(self, **kwargs: Any) -> ProductRecResult:
        return ProductRecResult(products=[PRODUCT])


class DemoInventoryAgent:
    async def run(self, **kwargs: Any) -> InventoryResult:
        return InventoryResult(available_products=[PRODUCT.product_id])


class DemoMarketingCopyAgent:
    async def run(self, **kwargs: Any) -> MarketingCopyResult:
        return MarketingCopyResult(copies=[{"product_id": PRODUCT.product_id, "copy": "Demo recommendation."}])


def main_demo() -> None:
    original_agents = (
        recommendation_graph.user_profile_agent,
        recommendation_graph.product_rec_agent,
        recommendation_graph.inventory_agent,
        recommendation_graph.marketing_copy_agent,
    )
    recommendation_graph.user_profile_agent = DemoUserProfileAgent()
    recommendation_graph.product_rec_agent = DemoProductRecAgent()
    recommendation_graph.inventory_agent = DemoInventoryAgent()
    recommendation_graph.marketing_copy_agent = DemoMarketingCopyAgent()

    try:
        with TestClient(main.app) as client:
            first_response = client.post(
                "/api/v1/recommend/graph",
                json={"user_id": "demo-user", "num_items": 1},
            )
            first_response.raise_for_status()
            first_result = first_response.json()
            run_id = first_result["request_id"]

            resumed_response = client.post(
                "/api/v1/recommend/graph",
                json={"user_id": "demo-user", "num_items": 1, "resume_run_id": run_id},
            )
            resumed_response.raise_for_status()
            resumed_result = resumed_response.json()

        assert resumed_result["request_id"] == run_id
        assert resumed_result["products"] == first_result["products"]
        print(f"First graph run ID: {run_id}")
        print(f"Recovered graph run ID: {resumed_result['request_id']}")
        print(f"Recommended product: {first_result['products'][0]['product_id']}")
    finally:
        (
            recommendation_graph.user_profile_agent,
            recommendation_graph.product_rec_agent,
            recommendation_graph.inventory_agent,
            recommendation_graph.marketing_copy_agent,
        ) = original_agents


if __name__ == "__main__":
    main_demo()