from .supervisor import SupervisorOrchestrator


def build_recommendation_graph():
    """Build the LangGraph recommendation pipeline on demand."""
    from .graph import build_recommendation_graph as build_graph

    return build_graph()


__all__ = ["SupervisorOrchestrator", "build_recommendation_graph"]