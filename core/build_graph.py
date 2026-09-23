from langgraph.graph import StateGraph, END
from core.graph_state import GraphState
from agents.graph_nodes import supervisor_node, sql_agent_node, rag_node, chart_node, reporting_node
from core.config import MAX_SQL_RETRIES, MAX_CHART_RETRIES


from schemas.models import QueryType


# Routing & Retry Conditions
def route_after_rag(state: GraphState) -> str:
    """General conversation / questions bypass SQL agent completely and go straight to reporting."""
    query_type = state.get("query_type")
    if query_type == QueryType.GENERAL:
        return "reporting"
    return "sql_agent"


def sql_should_retry(state: GraphState) -> str:
    """SQL fail hua toh retry, success toh chart pe jao, zyada retries toh give_up."""
    result = state.get("sql_result")
    if not result:
        return "give_up"
    
    success = result.get("success") if isinstance(result, dict) else getattr(result, "success", False)
    retries = result.get("retries_used", 0) if isinstance(result, dict) else getattr(result, "retries_used", 0)

    if success:
        return "success"
    if retries >= MAX_SQL_RETRIES:
        return "give_up"
    return "retry"


def chart_should_retry(state: GraphState) -> str:
    """Chart code fail hua toh retry, success toh reporting pe jao."""
    result = state.get("chart_result")
    if not result:
        return "give_up"

    success = result.get("success") if isinstance(result, dict) else getattr(result, "success", False)
    retries = result.get("retries_used", 0) if isinstance(result, dict) else getattr(result, "retries_used", 0)

    if success:
        return "success"
    if retries >= MAX_CHART_RETRIES:
        return "give_up"
    return "retry"



def build_graph():
    graph = StateGraph(GraphState)

    # Add nodes
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("rag", rag_node)
    graph.add_node("sql_agent", sql_agent_node)
    graph.add_node("chart_agent", chart_node)
    graph.add_node("reporting", reporting_node)

    # Entry point & initial routing
    graph.set_entry_point("supervisor")
    graph.add_edge("supervisor", "rag")

    # Conditional routing after RAG: GENERAL -> reporting, ANALYTICAL/BOTH -> sql_agent
    graph.add_conditional_edges(
        "rag",
        route_after_rag,
        {"reporting": "reporting", "sql_agent": "sql_agent"}
    )

    # Self-healing loop — SQL retry
    graph.add_conditional_edges(
        "sql_agent",
        sql_should_retry,
        {"retry": "sql_agent", "success": "chart_agent", "give_up": "reporting"}
    )

    # Self-healing loop — Chart retry
    graph.add_conditional_edges(
        "chart_agent",
        chart_should_retry,
        {"retry": "chart_agent", "success": "reporting", "give_up": "reporting"}
    )

    graph.add_edge("reporting", END)

    return graph.compile()


# Compiled graph — ready to invoke
app_graph = build_graph()
