"""Supervisor orchestration with resumable recommendation checkpoints."""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass
from typing import Any, Protocol

import structlog

from agents import InventoryAgent, MarketingCopyAgent, ProductRecAgent, UserProfileAgent
from config import get_settings
from models.schemas import (
    AgentResult,
    InventoryResult,
    MarketingCopyResult,
    Product,
    ProductRecResult,
    RecommendationRequest,
    RecommendationResponse,
    UserProfile,
    UserProfileResult,
)
from services.ab_test import ABTestEngine
from services.run_store import RecommendationCheckpoint, RecommendationRunStore


logger = structlog.get_logger()


class AgentRunner(Protocol):
    """Minimal agent contract used by the supervisor."""

    async def run(self, **kwargs: Any) -> AgentResult:
        ...


@dataclass(frozen=True)
class AgentBundle:
    """Agent dependencies required by a recommendation run."""

    user_profile: AgentRunner
    product_rec: AgentRunner
    marketing_copy: AgentRunner
    inventory: AgentRunner


class SupervisorOrchestrator:
    """Coordinates recommendation agents and persists phase checkpoints."""

    def __init__(
        self,
        ab_engine: ABTestEngine | None = None,
        agents: AgentBundle | None = None,
        run_store: RecommendationRunStore | None = None,
    ):
        if agents:
            self.user_profile_agent = agents.user_profile
            self.product_rec_agent = agents.product_rec
            self.marketing_copy_agent = agents.marketing_copy
            self.inventory_agent = agents.inventory
        else:
            self.user_profile_agent = UserProfileAgent()
            self.product_rec_agent = ProductRecAgent()
            self.marketing_copy_agent = MarketingCopyAgent()
            self.inventory_agent = InventoryAgent()
        self.ab_engine = ab_engine or ABTestEngine()
        self.run_store = run_store or RecommendationRunStore(get_settings().database_url)

    async def recommend(self, request: RecommendationRequest) -> RecommendationResponse:
        run_id, checkpoint = self._prepare_run(request)
        if checkpoint and checkpoint.status == "completed":
            return RecommendationResponse.model_validate(checkpoint.state["response"])

        start = time.perf_counter()
        experiment = self.ab_engine.assign(request.user_id)
        stage = checkpoint.stage if checkpoint else "started"
        try:
            if checkpoint and checkpoint.stage == "phase2_completed":
                phase_state = checkpoint.state
                profile_result = self._restore_result("user_profile", phase_state["profile_result"])
                rec_result = self._restore_result("product_rec", phase_state["rec_result"])
                rerank_result = self._restore_result("product_rec", phase_state["rerank_result"])
                inventory_result = self._restore_result("inventory", phase_state["inventory_result"])
                user_profile = self._restore_profile(phase_state.get("user_profile"))
                final_products = self._restore_products(phase_state["final_products"])
                degradation_reasons = list(phase_state["degradation_reasons"])
            else:
                if checkpoint and checkpoint.stage == "phase1_completed":
                    phase_state = checkpoint.state
                    profile_result = self._restore_result("user_profile", phase_state["profile_result"])
                    rec_result = self._restore_result("product_rec", phase_state["rec_result"])
                    user_profile = self._restore_profile(phase_state.get("user_profile"))
                    raw_products = self._restore_products(phase_state["raw_products"])
                    degradation_reasons = list(phase_state["degradation_reasons"])
                else:
                    profile_result, rec_result = await asyncio.gather(
                        self.user_profile_agent.run(user_id=request.user_id, context=request.context),
                        self.product_rec_agent.run(user_profile=None, num_items=request.num_items * 2),
                    )
                    user_profile = getattr(profile_result, "profile", None)
                    raw_products = getattr(rec_result, "products", [])
                    degradation_reasons = []
                    if not profile_result.success:
                        degradation_reasons.append("user_profile_unavailable")
                    if not rec_result.success:
                        degradation_reasons.append("product_recall_unavailable")
                    stage = "phase1_completed"
                    self.run_store.save_checkpoint(
                        run_id,
                        stage,
                        self._phase1_state(profile_result, rec_result, user_profile, raw_products, degradation_reasons),
                    )

                rerank_result, inventory_result = await asyncio.gather(
                    self.product_rec_agent.run(user_profile=user_profile, num_items=request.num_items),
                    self.inventory_agent.run(products=raw_products),
                )
                ranked_products = getattr(rerank_result, "products", raw_products)
                if not rerank_result.success:
                    degradation_reasons.append("product_rerank_unavailable")
                if not inventory_result.success:
                    degradation_reasons.append("inventory_unavailable")
                    final_products: list[Product] = []
                else:
                    available_ids = set(getattr(inventory_result, "available_products", []))
                    final_products = [
                        product for product in ranked_products if product.product_id in available_ids
                    ][:request.num_items]
                    if ranked_products and not final_products:
                        degradation_reasons.append("no_available_products")
                stage = "phase2_completed"
                self.run_store.save_checkpoint(
                    run_id,
                    stage,
                    self._phase2_state(
                        profile_result,
                        rec_result,
                        rerank_result,
                        inventory_result,
                        user_profile,
                        final_products,
                        degradation_reasons,
                    ),
                )

            copy_result = await self.marketing_copy_agent.run(
                user_profile=user_profile,
                products=final_products,
            )
            copies = getattr(copy_result, "copies", [])
            if not copy_result.success:
                degradation_reasons.append("marketing_copy_unavailable")
            total_latency = (time.perf_counter() - start) * 1000
            response = RecommendationResponse(
                request_id=run_id,
                user_id=request.user_id,
                products=final_products,
                marketing_copies=copies,
                experiment_group=experiment.get("group", "control"),
                agent_results={
                    "user_profile": profile_result,
                    "product_rec": rerank_result,
                    "marketing_copy": copy_result,
                    "inventory": inventory_result,
                },
                degradation_reasons=degradation_reasons,
                total_latency_ms=total_latency,
            )
            self.run_store.complete(run_id, response)
            logger.info("supervisor.complete", request_id=run_id, product_count=len(final_products))
            return response
        except Exception as error:
            self.run_store.fail(run_id, stage, str(error))
            raise

    def _prepare_run(
        self,
        request: RecommendationRequest,
    ) -> tuple[str, RecommendationCheckpoint | None]:
        if request.resume_run_id:
            checkpoint = self.run_store.load_checkpoint(request.resume_run_id)
            if checkpoint is None:
                raise ValueError(f"Recommendation run was not found: {request.resume_run_id}")
            if checkpoint.user_id != request.user_id:
                raise ValueError("Recommendation run does not belong to this user.")
            if checkpoint.stage not in {"phase1_completed", "phase2_completed", "completed"}:
                raise ValueError(f"Recommendation run cannot resume from stage: {checkpoint.stage}")
            return checkpoint.run_id, checkpoint

        run_id = str(uuid.uuid4())
        self.run_store.start(run_id, request)
        logger.info("supervisor.start", request_id=run_id, user_id=request.user_id, scene=request.scene)
        return run_id, None

    @staticmethod
    def _phase1_state(
        profile_result: AgentResult,
        rec_result: AgentResult,
        user_profile: UserProfile | None,
        raw_products: list[Product],
        degradation_reasons: list[str],
    ) -> dict[str, Any]:
        return {
            "profile_result": profile_result.model_dump(mode="json"),
            "rec_result": rec_result.model_dump(mode="json"),
            "user_profile": user_profile.model_dump(mode="json") if user_profile else None,
            "raw_products": [product.model_dump(mode="json") for product in raw_products],
            "degradation_reasons": degradation_reasons,
        }

    @classmethod
    def _phase2_state(
        cls,
        profile_result: AgentResult,
        rec_result: AgentResult,
        rerank_result: AgentResult,
        inventory_result: AgentResult,
        user_profile: UserProfile | None,
        final_products: list[Product],
        degradation_reasons: list[str],
    ) -> dict[str, Any]:
        state = cls._phase1_state(profile_result, rec_result, user_profile, [], degradation_reasons)
        state.update(
            {
                "rerank_result": rerank_result.model_dump(mode="json"),
                "inventory_result": inventory_result.model_dump(mode="json"),
                "final_products": [product.model_dump(mode="json") for product in final_products],
            }
        )
        return state

    @staticmethod
    def _restore_profile(value: dict[str, Any] | None) -> UserProfile | None:
        return UserProfile.model_validate(value) if value else None

    @staticmethod
    def _restore_products(values: list[dict[str, Any]]) -> list[Product]:
        return [Product.model_validate(value) for value in values]

    @staticmethod
    def _restore_result(agent_name: str, value: dict[str, Any]) -> AgentResult:
        result_type = {
            "user_profile": UserProfileResult,
            "product_rec": ProductRecResult,
            "inventory": InventoryResult,
            "marketing_copy": MarketingCopyResult,
        }[agent_name]
        return result_type.model_validate(value)