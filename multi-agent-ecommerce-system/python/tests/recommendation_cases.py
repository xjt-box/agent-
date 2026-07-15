"""Golden recommendation cases used by deterministic regression tests."""

from __future__ import annotations

from dataclasses import dataclass

from eval_contracts import RecommendationCase, RecommendationExpectation
from models.schemas import RecommendationRequest


REQUIRED_AGENT_RESULTS = {"user_profile", "product_rec", "marketing_copy", "inventory"}


@dataclass(frozen=True)
class GoldenRecommendationCase:
    """A recommendation contract together with the simulated inventory result."""

    case: RecommendationCase
    available_product_ids: list[str]


GOLDEN_RECOMMENDATION_CASES = [
    GoldenRecommendationCase(
        case=RecommendationCase(
            name="all_available_products_receive_matching_copy",
            request=RecommendationRequest(user_id="golden_all", num_items=2),
            expectation=RecommendationExpectation(
                max_products=2,
                allowed_product_ids={"P001", "P002"},
                expected_copy_product_ids={"P001", "P002"},
                required_agent_results=REQUIRED_AGENT_RESULTS,
            ),
        ),
        available_product_ids=["P001", "P002"],
    ),
    GoldenRecommendationCase(
        case=RecommendationCase(
            name="single_available_product_is_retained",
            request=RecommendationRequest(user_id="golden_single", num_items=2),
            expectation=RecommendationExpectation(
                max_products=2,
                allowed_product_ids={"P001"},
                expected_copy_product_ids={"P001"},
                required_agent_results=REQUIRED_AGENT_RESULTS,
            ),
        ),
        available_product_ids=["P001"],
    ),
    GoldenRecommendationCase(
        case=RecommendationCase(
            name="request_limit_is_respected",
            request=RecommendationRequest(user_id="golden_limit", num_items=1),
            expectation=RecommendationExpectation(
                max_products=1,
                allowed_product_ids={"P001", "P002"},
                expected_copy_product_ids={"P002"},
                required_agent_results=REQUIRED_AGENT_RESULTS,
            ),
        ),
        available_product_ids=["P001", "P002"],
    ),
    GoldenRecommendationCase(
        case=RecommendationCase(
            name="zero_requested_items_returns_empty_response",
            request=RecommendationRequest(user_id="golden_empty", num_items=0),
            expectation=RecommendationExpectation(
                max_products=0,
                allowed_product_ids=set(),
                expected_copy_product_ids=set(),
                required_agent_results=REQUIRED_AGENT_RESULTS,
            ),
        ),
        available_product_ids=[],
    ),
    GoldenRecommendationCase(
        case=RecommendationCase(
            name="empty_inventory_returns_no_recommendations",
            request=RecommendationRequest(user_id="golden_fallback", num_items=1),
            expectation=RecommendationExpectation(
                max_products=1,
                allowed_product_ids=set(),
                expected_copy_product_ids=set(),
                required_agent_results=REQUIRED_AGENT_RESULTS,
            ),
        ),
        available_product_ids=[],
    ),
]