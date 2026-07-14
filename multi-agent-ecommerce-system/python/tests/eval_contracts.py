"""Contracts for deterministic recommendation evaluation cases."""

from __future__ import annotations

from dataclasses import dataclass, field

from models.schemas import RecommendationRequest, RecommendationResponse


@dataclass(frozen=True)
class RecommendationExpectation:
    """Expected invariants for a recommendation response."""

    max_products: int
    allowed_product_ids: set[str] | None = None
    expected_copy_product_ids: set[str] | None = None
    required_agent_results: set[str] = field(default_factory=set)


@dataclass(frozen=True)
class RecommendationCase:
    """Input and expected invariants for one deterministic evaluation case."""

    name: str
    request: RecommendationRequest
    expectation: RecommendationExpectation


def assert_recommendation_contract(
    response: RecommendationResponse,
    expectation: RecommendationExpectation,
) -> None:
    """Assert that a recommendation response satisfies its expected invariants."""
    product_ids = [product.product_id for product in response.products]
    product_id_set = set(product_ids)

    assert len(product_ids) <= expectation.max_products, (
        f"Expected at most {expectation.max_products} products, got {len(product_ids)}."
    )
    assert len(product_ids) == len(product_id_set), "Recommended product IDs must be unique."

    if expectation.allowed_product_ids is not None:
        unexpected_ids = product_id_set - expectation.allowed_product_ids
        assert not unexpected_ids, (
            f"Recommended unavailable product IDs: {sorted(unexpected_ids)}."
        )

    copy_product_ids: set[str] = set()
    for copy_item in response.marketing_copies:
        product_id = copy_item.get("product_id")
        assert isinstance(product_id, str) and product_id, (
            "Each marketing copy must include a non-empty product_id."
        )
        copy_text = copy_item.get("copy")
        assert isinstance(copy_text, str) and copy_text.strip(), (
            "Each marketing copy must include non-empty copy text."
        )
        copy_product_ids.add(product_id)

    invalid_copy_ids = copy_product_ids - product_id_set
    assert not invalid_copy_ids, (
        f"Marketing copy references unknown product IDs: {sorted(invalid_copy_ids)}."
    )

    if expectation.expected_copy_product_ids is not None:
        assert copy_product_ids == expectation.expected_copy_product_ids, (
            "Marketing copy product IDs do not match the expectation. "
            f"Expected {sorted(expectation.expected_copy_product_ids)}, "
            f"got {sorted(copy_product_ids)}."
        )

    missing_agent_results = expectation.required_agent_results - set(response.agent_results)
    assert not missing_agent_results, (
        f"Missing agent results: {sorted(missing_agent_results)}."
    )
