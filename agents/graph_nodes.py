from core.graph_state import GraphState
from core.llm_client import llm
from database.duckdb_manager import DuckDBClient
from database.vector_db import VectorDBManager
from schemas.models import QueryType, SQLResult, ChartResult, FinalResponse, TableResult


from core.graph_state import GraphState
from core.llm_client import llm
from database.duckdb_manager import DuckDBClient
from database.vector_db import VectorDBManager
from schemas.models import QueryType, SQLResult, ChartResult, FinalResponse, TableResult


def supervisor_node(state: GraphState) -> GraphState:
    """Classify user question intent — general / analytical / hybrid"""
    state["query_type"] = llm.classify_query(state["question"], state.get("chat_history"))
    return state


def rag_node(state: GraphState) -> GraphState:
    """Fetch relevant context from Vector DB — column meanings, domain context"""
    vdb = VectorDBManager(state["user_id"])
    results = vdb.search(
        query=state["question"],
        top_k=5,
        doc_id=state.get("doc_id"),
    )
    state["rag_context"] = results
    return state


def sql_agent_node(state: GraphState) -> GraphState:
    """Generate SQL query using schema, RAG context, and chat history, then execute on DuckDB"""
    db = DuckDBClient(state["user_id"])

    # Construct schema context for LLM prompt
    schema_str = db.get_full_context()

    # Format RAG context
    rag_context = state.get("rag_context", [])
    context_str = "\n".join([item.get("text", "") if isinstance(item, dict) else str(item) for item in rag_context]) if rag_context else "No additional context available."

    # Check if this is a retry attempt or first attempt
    prev_result = state.get("sql_result")
    if prev_result and not prev_result.success:
        # Self-healing — fix SQL query using error traceback
        sql = llm.fix_sql(prev_result.sql_query, prev_result.error, schema_str)
        retries = prev_result.retries_used + 1
    else:
        # First attempt — generate SQL query with chat history
        sql = llm.generate_sql(state["question"], schema_str, context_str, state.get("chat_history"))
        retries = 0

    # Execute query on DuckDB
    query_result = db.run_query(sql)
    db.close()

    state["sql_result"] = SQLResult(
        success=query_result["success"],
        sql_query=sql,
        columns=query_result.get("columns"),
        rows=query_result.get("rows"),
        error=query_result.get("error"),
        retries_used=retries,
    )
    return state


def chart_node(state: GraphState) -> GraphState:
    """Generate and execute matplotlib chart code from SQL query results"""
    # Skip chart for GENERAL queries
    if state.get("query_type") == QueryType.GENERAL:
        state["chart_result"] = ChartResult(success=True, chart_base64=None, error="General query - visual chart skipped")
        return state

    sql_res = state.get("sql_result")
    if not sql_res or not sql_res.success or not sql_res.rows:
        state["chart_result"] = ChartResult(success=True, chart_base64=None, error="No SQL data available")
        return state

    # Skip chart if data is a single scalar/count or insufficient data points for visual plotting
    if len(sql_res.rows) < 2 or len(sql_res.columns) < 2:
        state["chart_result"] = ChartResult(success=True, chart_base64=None, error="Insufficient data points for visual chart")
        return state

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import pandas as pd
    import base64
    import io

    # Check if this is a retry attempt or first attempt
    prev_chart = state.get("chart_result")
    if prev_chart and not prev_chart.success:
        code = llm.fix_chart_code(prev_chart.error, prev_chart.error)
        retries = prev_chart.retries_used + 1
    else:
        code = llm.generate_chart_code(sql_res.columns, sql_res.rows[:5])
        retries = 0

    # Execute chart generation code
    try:
        df = pd.DataFrame(sql_res.rows, columns=sql_res.columns)
        plt.clf()
        plt.close('all')
        sns.set_theme(style="whitegrid")
        fig, ax = plt.subplots(figsize=(10, 6))

        # Execute LLM code with seaborn & numpy available
        exec(code, {"df": df, "plt": plt, "pd": pd, "sns": sns, "np": np, "ax": ax, "fig": fig})

        # Convert chart figure to base64 string
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="white")
        plt.close('all')
        buf.seek(0)
        chart_base64 = base64.b64encode(buf.read()).decode("utf-8")

        state["chart_result"] = ChartResult(
            success=True, chart_base64=chart_base64, retries_used=retries
        )
    except Exception as e:
        plt.close('all')
        state["chart_result"] = ChartResult(
            success=False, error=str(e), retries_used=retries
        )

    return state


def reporting_node(state: GraphState) -> GraphState:
    """Synthesize final summary response combining query results and chart context"""
    sql_res = state.get("sql_result")
    chart_res = state.get("chart_result")

    # Fetch schema & dataset context
    db = DuckDBClient(state["user_id"])
    schema_context = db.get_full_context()
    db.close()

    # Combine data preview and schema context
    data_preview = f"DATASET SCHEMA & FEATURES:\n{schema_context}\n\n"
    if sql_res and sql_res.success:
        data_preview += f"SQL QUERY EXECUTED:\n{sql_res.sql_query}\n\nQUERY RESULTS:\nColumns: {sql_res.columns}\nRows Sample: {sql_res.rows[:10]}"
    else:
        data_preview += "NO SQL EXECUTED OR FAILED."

    summary = llm.generate_summary(state["question"], data_preview, state.get("chat_history"))

    # Assemble final response object
    state["final_response"] = FinalResponse(
        summary_text=summary,
        chart_base64=chart_res.chart_base64 if chart_res and chart_res.success else None,
        data_table=TableResult(
            columns=sql_res.columns, rows=sql_res.rows
        ) if sql_res and sql_res.success else None,
        query_type=state["query_type"],
    )
    return state
