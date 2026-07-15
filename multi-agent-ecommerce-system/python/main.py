from __future__ import annotations

import os
import sys
import uuid
from contextlib import asynccontextmanager

import aiosqlite
import structlog
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

sys.path.insert(0, os.path.dirname(__file__))

from config import get_settings
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from models.schemas import RecommendationRequest, RecommendationResponse
from orchestrator.graph import build_recommendation_graph
from orchestrator.supervisor import SupervisorOrchestrator
from services.ab_test import ABTestEngine
from services.metrics import MetricsCollector


logger = structlog.get_logger()
settings = get_settings()

ab_engine = ABTestEngine()
metrics_collector = MetricsCollector()
supervisor = SupervisorOrchestrator(ab_engine=ab_engine)
rec_graph = None
graph_checkpointer = None
graph_checkpointer_connection = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global rec_graph, graph_checkpointer, graph_checkpointer_connection
    checkpoint_path = _checkpoint_database_path(settings.database_url)
    graph_checkpointer_connection = await aiosqlite.connect(checkpoint_path)
    graph_checkpointer = AsyncSqliteSaver(
        graph_checkpointer_connection,
        serde=_checkpoint_serializer(),
    )
    rec_graph = build_recommendation_graph(checkpointer=graph_checkpointer)
    logger.info("app.startup", model=settings.llm_model, checkpoint_path=checkpoint_path)
    try:
        yield
    finally:
        await graph_checkpointer_connection.close()
        rec_graph = None
        graph_checkpointer = None
        graph_checkpointer_connection = None
        logger.info("app.shutdown")

def _checkpoint_database_path(database_url: str) -> str:
    if not database_url.startswith("sqlite:///"):
        raise ValueError("LangGraph checkpointing requires a SQLite database URL.")
    return database_url.removeprefix("sqlite:///")
def _checkpoint_serializer() -> JsonPlusSerializer:
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


app = FastAPI(
    title="Multi-Agent E-Commerce Recommendation System",
    description="User profile agent, product recommendation agent, marketing copy agent, and inventory decision agent.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def home():
    frontend_path = os.path.join(os.path.dirname(__file__), "frontend", "index.html")
    return FileResponse(frontend_path)


@app.get("/health")
async def health():
    return {"status": "healthy", "model": settings.llm_model}


@app.post("/api/v1/recommend", response_model=RecommendationResponse)
async def recommend(request: RecommendationRequest):
    response = await supervisor.recommend(request)
    _collect_metrics(response)
    return response


@app.post("/api/v1/recommend/graph")
async def recommend_via_graph(request: RecommendationRequest):
    if not rec_graph or not graph_checkpointer:
        raise HTTPException(status_code=503, detail="Graph is not initialized.")

    run_id = request.resume_run_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": run_id}}
    if request.resume_run_id:
        checkpoint = await graph_checkpointer.aget_tuple(config)
        if checkpoint is None:
            raise HTTPException(status_code=404, detail="Recommendation run was not found.")
        snapshot = await rec_graph.aget_state(config)
        if snapshot.values.get("user_id") != request.user_id:
            raise HTTPException(status_code=403, detail="Recommendation run does not belong to this user.")
        result = await rec_graph.ainvoke(None, config=config)
    else:
        state = {
            "request_id": run_id,
            "user_id": request.user_id,
            "scene": request.scene,
            "num_items": request.num_items,
            "context": request.context,
        }
        result = await rec_graph.ainvoke(state, config=config)
    return {
        "request_id": result.get("request_id"),
        "user_id": result.get("user_id"),
        "products": [p.model_dump() for p in result.get("final_products", [])],
        "marketing_copies": result.get("marketing_copies", []),
        "experiment_group": result.get("experiment_group", "control"),
        "degradation_reasons": result.get("degradation_reasons", []),
        "total_latency_ms": round(result.get("total_latency_ms", 0), 1),
    }


@app.get("/api/v1/experiments")
async def get_experiments():
    experiments = {}
    for exp_id, exp in ab_engine.experiments.items():
        experiments[exp_id] = {
            "name": exp.name,
            "enabled": exp.enabled,
            "groups": [
                {
                    "name": group.name,
                    "weight": group.weight,
                    "config": group.config,
                    "successes": group.successes,
                    "failures": group.failures,
                }
                for group in exp.groups
            ],
            "stats": ab_engine.get_stats(exp_id),
        }
    return experiments


@app.get("/api/v1/metrics")
async def get_metrics():
    return {
        "agents": metrics_collector.get_agent_stats(),
        "business": metrics_collector.get_business_stats(),
    }


@app.post("/api/v1/experiments/{experiment_id}/outcome")
async def record_outcome(experiment_id: str, group: str, success: bool):
    ab_engine.record_outcome(experiment_id, group, success)
    return {"status": "recorded"}


def _collect_metrics(response: RecommendationResponse):
    for name, result in response.agent_results.items():
        metrics_collector.record_agent_call(
            agent_name=name,
            success=result.success,
            latency_ms=result.latency_ms,
        )


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
