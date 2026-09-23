from pydantic import BaseModel
from typing import Optional, List, Literal
from enum import Enum


class QueryType(str, Enum):
    GENERAL = "general"
    ANALYTICAL = "analytical"
    BOTH = "both"


# Request

class QueryRequest(BaseModel):
    user_id: str
    question: str
    doc_id: Optional[str] = None


#  Supervisor output
class SupervisorDecision(BaseModel):
    query_type: QueryType
    table_name: str


#SQL Agent 

class SQLResult(BaseModel):
    success: bool
    sql_query: Optional[str] = None
    columns: Optional[List[str]] = None
    rows: Optional[List[tuple]] = None
    error: Optional[str] = None
    retries_used: int = 0


# RAG

class RAGContextItem(BaseModel):
    type: Literal["column_meaning", "few_shot"]
    text: str
    score: float
    column: Optional[str] = None
    sql: Optional[str] = None


#  Visualizer / Analytical branch 

class ChartResult(BaseModel):
    success: bool
    chart_base64: Optional[str] = None
    error: Optional[str] = None
    retries_used: int = 0


# ---------- General branch ----------

class TableResult(BaseModel):
    columns: List[str]
    rows: List[tuple]


# ---------- Reporting Agent / Final output ----------

class FinalResponse(BaseModel):
    summary_text: str
    chart_base64: Optional[str] = None
    data_table: Optional[TableResult] = None
    query_type: QueryType
