"""Deterministic regression tests for the recommendation orchestration flow."""

from __future__ import annotations

import asyncio
import inspect
import os
import sys
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.schemas import (
    InventoryResult,
    MarketingCopyResult,
    Product,
    ProductRecResult,
    RecommendationRequest,
    UserProfile,
    UserProfileResult,
)
from orchestrator.supervisor import AgentBundle, SupervisorOrchestrator
from eval_contracts import RecommendationExpectation, assert_recommendation_contract
from recommendation_cases import GOLDEN_RECOMMENDATION_CASES


PRODUCTS = [
    Product(product_id="P001", name="Phone", category="electronics", price=1000, stock=10),
    Product(product_id="P002", name="Headphones", category="electronics", price=200, stock=5),
]
REQUIRED_AGENT_RESULTS = {"user_profile", "product_rec", "marketing_copy", "inventory"}


class FakeUserProfileAgent:
    async def run(self, **kwargs: Any) -> UserProfileResult:
        return UserProfileResult(profile=UserProfile(user_id=kwargs["user_id"]))


class FakeProductRecAgent:
    async def run(self, **kwargs: Any) -> ProductRecResult:
        if kwargs["user_profile"] is None:
            return ProductRecResult(products=PRODUCTS)
        return ProductRecResult(products=list(reversed(PRODUCTS))[:kwargs["num_items"]])


class FakeInventoryAgent:
    def __init__(self, available_product_ids: list[str], success: bool = True):
        self.available_product_ids = available_product_ids
        self.success = success

    async def run(self, **kwargs: Any) -> InventoryResult:
        return InventoryResult(
            success=self.success,
            error=None if self.success else "inventory service unavailable",
            available_products=self.available_product_ids,
        )


class FakeMarketingCopyAgent:
    async def run(self, **kwargs: Any) -> MarketingCopyResult:
        copies = [
            {"product_id": product.product_id, "copy": f"Try {product.name}."}
            for product in kwargs["products"]
        ]
        return MarketingCopyResult(copies=copies)


def build_supervisor(
    available_product_ids: list[str], inventory_success: bool = True
) -> SupervisorOrchestrator:
    return SupervisorOrchestrator(
        agents=AgentBundle(
            user_profile=FakeUserProfileAgent(),
            product_rec=FakeProductRecAgent(),
            marketing_copy=FakeMarketingCopyAgent(),
            inventory=FakeInventoryAgent(available_product_ids, inventory_success),
        )
    )


def test_golden_recommendation_cases() -> None:
    for golden_case in GOLDEN_RECOMMENDATION_CASES:
        response = asyncio.run(
            build_supervisor(golden_case.available_product_ids).recommend(
                golden_case.case.request
            )
        )
        assert_recommendation_contract(response, golden_case.case.expectation)
        print(f"[PASS] {golden_case.case.name}")

class FailingProductRecAgent:
    async def run(self, **kwargs: Any) -> ProductRecResult:
        raise RuntimeError("product recall unavailable")


def test_agent_failure_propagates() -> None:
    supervisor = SupervisorOrchestrator(
        agents=AgentBundle(
            user_profile=FakeUserProfileAgent(),
            product_rec=FailingProductRecAgent(),
            marketing_copy=FakeMarketingCopyAgent(),
            inventory=FakeInventoryAgent([]),
        )
    )

    try:
        asyncio.run(supervisor.recommend(RecommendationRequest(user_id="user_failure")))
    except RuntimeError as error:
        assert str(error) == "product recall unavailable"
    else:
        raise AssertionError("Agent failure did not propagate.")



def test_inventory_failure_returns_no_unverified_products() -> None:
    response = asyncio.run(
        build_supervisor(["P001"], inventory_success=False).recommend(
            RecommendationRequest(user_id="user_inventory_failure", num_items=1)
        )
    )

    assert response.products == []
    assert response.marketing_copies == []
    assert "inventory_unavailable" in response.degradation_reasons
def test_contract_rejects_unknown_copy_product() -> None:
    response = asyncio.run(build_supervisor(["P002"]).recommend(
        RecommendationRequest(user_id="user_004", num_items=1)
    ))
    response.marketing_copies = [{"product_id": "P999", "copy": "Invalid."}]

    try:
        assert_recommendation_contract(
            response,
            RecommendationExpectation(max_products=1),
        )
    except AssertionError as error:
        assert "unknown product IDs" in str(error)
    else:
        raise AssertionError("Unknown marketing copy product ID was accepted.")


def test_contract_rejects_empty_copy() -> None:
    response = asyncio.run(build_supervisor(["P002"]).recommend(
        RecommendationRequest(user_id="user_empty_copy", num_items=1)
    ))
    response.marketing_copies[0]["copy"] = ""

    try:
        assert_recommendation_contract(response, RecommendationExpectation(max_products=1))
    except AssertionError as error:
        assert "non-empty copy text" in str(error)
    else:
        raise AssertionError("Empty marketing copy was accepted.")


def test_constructor_remains_backward_compatible() -> None:
    parameters = inspect.signature(SupervisorOrchestrator).parameters
    assert parameters["ab_engine"].default is None
    assert parameters["agents"].default is None


if __name__ == "__main__":
    test_golden_recommendation_cases()
    test_agent_failure_propagates()
    test_inventory_failure_returns_no_unverified_products()
    test_contract_rejects_unknown_copy_product()
    test_contract_rejects_empty_copy()
    test_constructor_remains_backward_compatible()
    print("All recommendation evaluation tests passed!")