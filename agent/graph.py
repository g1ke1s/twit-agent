"""
agent/graph.py — Pipeline: 7 nodes.

INPUT → researcher → analyst → angle_finder → writer → critic → validator_gate → formatter → END
         angle_finder: adversarial thesis selection (analyst → non-obvious angle → writer)
"""
from __future__ import annotations

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from agent.schemas import AgentState
from agent.nodes import (
    input_router_node,
    researcher_node,
    analyst_node,
    angle_finder_node,
    writer_node,
    critic_node,
    route_after_critic,
    validator_gate_node,
    route_after_validation,
    formatter_node,
    distributor_node,
)

_compiled_graph = None


def build_graph(checkpointer=None):
    g = StateGraph(AgentState)

    g.add_node("input_router",   input_router_node)
    g.add_node("researcher",     researcher_node)
    g.add_node("analyst",        analyst_node)
    g.add_node("angle_finder",   angle_finder_node)   # PART 1: contrarian thesis
    g.add_node("writer",         writer_node)
    g.add_node("critic",         critic_node)
    g.add_node("validator_gate", validator_gate_node)
    g.add_node("formatter",      formatter_node)
    g.add_node("distributor",    distributor_node)

    # Edges
    g.set_entry_point("input_router")
    g.add_edge("input_router",  "researcher")
    g.add_edge("researcher",    "analyst")
    g.add_edge("analyst",       "angle_finder")   # angle_finder sits between analyst and writer
    g.add_edge("angle_finder",  "writer")
    g.add_edge("writer",        "critic")

    g.add_conditional_edges(
        "critic",
        route_after_critic,
        {"rewrite": "writer", "advance": "validator_gate"},
    )

    g.add_conditional_edges(
        "validator_gate",
        route_after_validation,
        {"approve": "formatter", "reject": "writer"},
    )

    g.add_edge("formatter",   "distributor")
    g.add_edge("distributor", END)

    cp = checkpointer or MemorySaver()
    return g.compile(
        checkpointer=cp,
        interrupt_before=["validator_gate"],
    )


def get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph
