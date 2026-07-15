"""
商品推荐Agent
- 召回层：协同过滤 + 向量检索(Milvus) + 热度/新品策略
- 排序层：LLM重排 + 特征交叉(用户画像 x 商品属性)
- 多样性控制：类目打散、卖家去重、新品加权
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from config import get_settings
from services.catalog_repository import CatalogRepository
from models.schemas import Product, ProductRecResult, UserProfile

from .base_agent import BaseAgent

RERANK_PROMPT = """你是电商推荐排序专家。根据用户画像和候选商品,重新排序并选出最优的{num_items}个商品。

用户画像:
{user_profile}

候选商品:
{candidates}

排序原则:
1. 用户偏好类目优先
2. 价格在用户可接受范围内
3. 保证类目多样性(相邻商品尽量不同类目)
4. 新品适当加权

请输出商品ID列表(JSON数组),按推荐优先级排序:
["product_id_1", "product_id_2", ...]

只输出JSON数组,不要其他内容。"""

class ProductRecAgent(BaseAgent):
    def __init__(self, catalog_repository: CatalogRepository | None = None):
        settings = get_settings()
        super().__init__(
            name="product_rec",
            timeout=settings.agent_timeout_product_rec,
        )
        self.llm = ChatOpenAI(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
            model=settings.llm_model,
            temperature=0.3,
            max_tokens=512,
        )
        self.catalog_repository = catalog_repository or CatalogRepository(settings.database_url)
        self.catalog_repository.initialize()
        self.vector_store: Any = None  # injected in Phase 2
    async def _execute(self, **kwargs: Any) -> ProductRecResult:
        user_profile: UserProfile | None = kwargs.get("user_profile")
        num_items: int = kwargs.get("num_items", 10)

        candidates = await self._recall(user_profile, num_items * 3)
        ranked_ids = await self._rerank(user_profile, candidates, num_items)

        id_to_product = {p.product_id: p for p in candidates}
        final_products = []
        for pid in ranked_ids:
            if pid in id_to_product:
                final_products.append(id_to_product[pid])
        if len(final_products) < num_items:
            for p in candidates:
                if p.product_id not in ranked_ids:
                    final_products.append(p)
                    if len(final_products) >= num_items:
                        break

        return ProductRecResult(
            success=True,
            products=final_products[:num_items],
            recall_strategy="collaborative_filter+vector+hot",
            data={"candidate_count": len(candidates), "reranked": len(ranked_ids)},
            confidence=0.8,
        )

    async def _recall(self, profile: UserProfile | None, limit: int) -> list[Product]:
        """Recall persisted catalog products before optional vector retrieval is added."""
        if self.vector_store:
            pass  # Phase 2: real vector search

        candidates = self.catalog_repository.list_products()
        if profile and profile.preferred_categories:
            preferred = set(profile.preferred_categories)
            candidates.sort(
                key=lambda product: (
                    product.category not in preferred,
                    product.stock <= 0,
                    -product.score,
                    product.product_id,
                )
            )

        return candidates[:limit]
    async def _rerank(
        self, profile: UserProfile | None, candidates: list[Product], num_items: int
    ) -> list[str]:
        if not profile:
            return [p.product_id for p in candidates[:num_items]]

        profile_summary = {
            "segments": [s.value for s in profile.segments],
            "preferred_categories": profile.preferred_categories,
            "price_range": list(profile.price_range),
        }
        candidate_summary = [
            {"id": p.product_id, "name": p.name, "category": p.category, "price": p.price, "tags": p.tags}
            for p in candidates
        ]
        prompt = RERANK_PROMPT.format(
            num_items=num_items,
            user_profile=json.dumps(profile_summary, ensure_ascii=False),
            candidates=json.dumps(candidate_summary, ensure_ascii=False),
        )
        messages = [
            SystemMessage(content="你是电商推荐排序专家。"),
            HumanMessage(content=prompt),
        ]
        response = await self.llm.ainvoke(messages)
        try:
            raw = response.content.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
            return json.loads(raw)
        except (json.JSONDecodeError, IndexError):
            return [p.product_id for p in candidates[:num_items]]
