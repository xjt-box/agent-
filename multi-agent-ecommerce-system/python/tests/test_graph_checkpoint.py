"""Regression test for LangGraph SQLite checkpoints."""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import aiosqlite
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from models.schemas import InventoryResult, MarketingCopyResult, Product, ProductRecResult, UserProfile, UserProfileResult
from orchestrator import graph as recommendation_graph


PRODUCT = Product(product_id="P001", name="Phone", category="electronics", price=1000, stock=10)


class FakeProfileAgent:
    def __init__(self) -> None:
        self.calls = 0

    async def run(self, **kwargs: Any) -> UserProfileResult:
        self.calls += 1
        return UserProfileResult(profile=UserProfile(user_id=kwargs["user_id"]))


class FakeProductAgent:
    def __init__(self) -> None:
        self.calls = 0

    async def run(self, **kwargs: Any) -> ProductRecResult:
        self.calls += 1
        return ProductRecResult(products=[PRODUCT])


class FakeInventoryAgent:
    def __init__(self) -> None:
        self.calls = 0

    async def run(self, **kwargs: Any) -> InventoryResult:
        self.calls += 1
        return InventoryResult(available_products=[PRODUCT.product_id])


class FakeCopyAgent:
    def __init__(self) -> None:
        self.calls = 0

    async def run(self, **kwargs: Any) -> MarketingCopyResult:
        self.calls += 1
        return MarketingCopyResult(copies=[{"product_id": PRODUCT.product_id, "copy": "Recommended."}])


async def test_graph_checkpoint_round_trip() -> None:
    original_agents = (
        recommendation_graph.user_profile_agent,
        recommendation_graph.product_rec_agent,
        recommendation_graph.inventory_agent,
        recommendation_graph.marketing_copy_agent,
    )
    fake_agents = (FakeProfileAgent(), FakeProductAgent(), FakeInventoryAgent(), FakeCopyAgent())
    recommendation_graph.user_profile_agent = fake_agents[0]
    recommendation_graph.product_rec_agent = fake_agents[1]
    recommendation_graph.inventory_agent = fake_agents[2]
    recommendation_graph.marketing_copy_agent = fake_agents[3]

    try:
        with tempfile.TemporaryDirectory() as directory:
            database_path = str(Path(directory) / "graph-checkpoints.db")
            connection = await aiosqlite.connect(database_path)
            saver = AsyncSqliteSaver(connection, serde=_serializer())
            try:
                graph = recommendation_graph.build_recommendation_graph(checkpointer=saver)
                config = {"configurable": {"thread_id": "graph-run"}}
                result = await graph.ainvoke(
                    {
                        "request_id": "graph-run",
                        "user_id": "graph-user",
                        "scene": "homepage",
                        "num_items": 1,
                        "context": {},
                    },
                    config=config,
                )
                checkpoint = await saver.aget_tuple(config)
                calls_after_first_run = tuple(agent.calls for agent in fake_agents)
                resumed = await graph.ainvoke(None, config=config)

                assert checkpoint is not None
                assert result["request_id"] == "graph-run"
                assert [product.product_id for product in result["final_products"]] == ["P001"]
                assert resumed["marketing_copies"] == result["marketing_copies"]
                assert tuple(agent.calls for agent in fake_agents) == calls_after_first_run
            finally:
                await connection.close()
    finally:
        (
            recommendation_graph.user_profile_agent,
            recommendation_graph.product_rec_agent,
            recommendation_graph.inventory_agent,
            recommendation_graph.marketing_copy_agent,
        ) = original_agents


def _serializer() -> JsonPlusSerializer:
    return JsonPlusSerializer(
        allowed_msgpack_modules=[
            ("models.schemas", "Product"),
            ("models.schemas", "UserProfile"),
            ("models.schemas", "UserProfileResult"),
            ("models.schemas", "ProductRecResult"),
            ("models.schemas", "InventoryResult"),
            ("models.schemas", "MarketingCopyResult"),
        ]
    )

def main() -> None:
    asyncio.run(test_graph_checkpoint_round_trip())
    print("LangGraph SQLite checkpoint test passed!")


if __name__ == "__main__":
    main()