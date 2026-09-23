from typing import Optional
from typing_extensions import TypedDict
from schemas.models import QueryType, SQLResult, ChartResult, RAGContextItem, FinalResponse


class GraphState(TypedDict):
    """LangGraph shared state — read and updated by each workflow node."""
    user_id: str
    doc_id: str
    question: str
    table_name: str
    chat_history: Optional[list[dict]]
    query_type: Optional[QueryType]
    rag_context: Optional[list[RAGContextItem]]
    sql_result: Optional[SQLResult]
    chart_result: Optional[ChartResult]
    final_response: Optional[FinalResponse]
