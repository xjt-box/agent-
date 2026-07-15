"""Regression tests for recommendation checkpoint persistence and recovery."""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from pathlib import Path
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
from services.run_store import RecommendationRunStore


PRODUCT = Product(product_id="P001", name="Phone", category="electronics", price=1000, stock=10)


class CountingUserProfileAgent:
    def __init__(self):
        self.calls = 0

    async def run(self, **kwargs: Any) -> UserProfileResult:
        self.calls += 1
        return UserProfileResult(profile=UserProfile(user_id=kwargs["user_id"]))


class CountingProductRecAgent:
    def __init__(self):
        self.calls = 0

    async def run(self, **kwargs: Any) -> ProductRecResult:
        self.calls += 1
        return ProductRecResult(products=[PRODUCT])


class CountingInventoryAgent:
    def __init__(self):
        self.calls = 0

    async def run(self, **kwargs: Any) -> InventoryResult:
        self.calls += 1
        return InventoryResult(available_products=[PRODUCT.product_id])


class CountingMarketingCopyAgent:
    def __init__(self):
        self.calls = 0

    async def run(self, **kwargs: Any) -> MarketingCopyResult:
        self.calls += 1
        return MarketingCopyResult(
            copies=[{"product_id": product.product_id, "copy": "Recommended."} for product in kwargs["products"]]
        )


def build_supervisor(store: RecommendationRunStore) -> tuple[SupervisorOrchestrator, tuple[Any, ...]]:
    agents = (
        CountingUserProfileAgent(),
        CountingProductRecAgent(),
        CountingMarketingCopyAgent(),
        CountingInventoryAgent(),
    )
    return (
        SupervisorOrchestrator(
            agents=AgentBundle(
                user_profile=agents[0],
                product_rec=agents[1],
                marketing_copy=agents[2],
                inventory=agents[3],
            ),
            run_store=store,
        ),
        agents,
    )


def test_store_round_trip() -> None:
    with tempfile.TemporaryDirectory() as directory:
        store = RecommendationRunStore(f"sqlite:///{Path(directory) / 'runs.db'}")
        request = RecommendationRequest(user_id="store_user")
        store.start("run-store", request)
        store.save_checkpoint("run-store", "phase1_completed", {"value": "saved"})
        checkpoint = store.load_checkpoint("run-store")
        assert checkpoint is not None
        assert checkpoint.user_id == "store_user"
        assert checkpoint.stage == "phase1_completed"
        assert checkpoint.state == {"value": "saved"}
        store.close()


def test_phase1_resume_skips_completed_agents() -> None:
    with tempfile.TemporaryDirectory() as directory:
        store = RecommendationRunStore(f"sqlite:///{Path(directory) / 'runs.db'}")
        request = RecommendationRequest(user_id="resume_user", resume_run_id="phase1-run", num_items=1)
        profile_result = UserProfileResult(profile=UserProfile(user_id=request.user_id))
        rec_result = ProductRecResult(products=[PRODUCT])
        store.start("phase1-run", RecommendationRequest(user_id=request.user_id, num_items=1))
        store.save_checkpoint(
            "phase1-run",
            "phase1_completed",
            SupervisorOrchestrator._phase1_state(profile_result, rec_result, profile_result.profile, [PRODUCT], []),
        )
        supervisor, agents = build_supervisor(store)
        response = asyncio.run(supervisor.recommend(request))
        assert response.products == [PRODUCT]
        assert agents[0].calls == 0
        assert agents[1].calls == 1
        assert agents[3].calls == 1
        assert agents[2].calls == 1
        store.close()


def test_phase2_resume_only_runs_marketing_copy() -> None:
    with tempfile.TemporaryDirectory() as directory:
        store = RecommendationRunStore(f"sqlite:///{Path(directory) / 'runs.db'}")
        request = RecommendationRequest(user_id="resume_user", resume_run_id="phase2-run", num_items=1)
        profile_result = UserProfileResult(profile=UserProfile(user_id=request.user_id))
        rec_result = ProductRecResult(products=[PRODUCT])
        inventory_result = InventoryResult(available_products=[PRODUCT.product_id])
        store.start("phase2-run", RecommendationRequest(user_id=request.user_id, num_items=1))
        store.save_checkpoint(
            "phase2-run",
            "phase2_completed",
            SupervisorOrchestrator._phase2_state(
                profile_result,
                rec_result,
                rec_result,
                inventory_result,
                profile_result.profile,
                [PRODUCT],
                [],
            ),
        )
        supervisor, agents = build_supervisor(store)
        response = asyncio.run(supervisor.recommend(request))
        assert response.products == [PRODUCT]
        assert agents[0].calls == 0
        assert agents[1].calls == 0
        assert agents[3].calls == 0
        assert agents[2].calls == 1
        store.close()


def test_unknown_or_foreign_run_is_rejected() -> None:
    with tempfile.TemporaryDirectory() as directory:
        store = RecommendationRunStore(f"sqlite:///{Path(directory) / 'runs.db'}")
        supervisor, _ = build_supervisor(store)
        try:
            asyncio.run(supervisor.recommend(RecommendationRequest(user_id="user", resume_run_id="missing")))
        except ValueError as error:
            assert "not found" in str(error)
        else:
            raise AssertionError("Unknown run ID was accepted.")

        store.start("owned-run", RecommendationRequest(user_id="owner"))
        try:
            asyncio.run(supervisor.recommend(RecommendationRequest(user_id="other", resume_run_id="owned-run")))
        except ValueError as error:
            assert "does not belong" in str(error)
        else:
            raise AssertionError("Foreign run ID was accepted.")
        store.close()


if __name__ == "__main__":
    test_store_round_trip()
    test_phase1_resume_skips_completed_agents()
    test_phase2_resume_only_runs_marketing_copy()
    test_unknown_or_foreign_run_is_rejected()
    print("All recommendation run store tests passed!")