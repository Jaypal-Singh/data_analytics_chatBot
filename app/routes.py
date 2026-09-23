import os
import re
import uuid
import logging
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from schemas.models import QueryRequest, FinalResponse
from core.build_graph import app_graph
from database.duckdb_manager import DuckDBClient

logger = logging.getLogger(__name__)
router = APIRouter()

# Upload directory
UPLOAD_DIR = "storage/uploads"


def _sanitize_table_name(filename: str) -> str:
    """Generate a safe database table name from a filename."""
    name = os.path.splitext(os.path.basename(filename))[0]
    name = re.sub(r"[^a-zA-Z0-9_]", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name.lower()


@router.post("/upload")
async def upload_file(
    user_id: str = Form(...),
    file: UploadFile = File(...),
):
    """Upload CSV file and create table in user's DuckDB instance."""

    # File type check
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")

    # Safe filename with UUID to prevent overwrites
    safe_filename = f"{uuid.uuid4().hex}_{os.path.basename(file.filename)}"
    user_upload_dir = os.path.join(UPLOAD_DIR, str(user_id))
    os.makedirs(user_upload_dir, exist_ok=True)
    save_path = os.path.join(user_upload_dir, safe_filename)

    # Save file contents
    content = await file.read()
    with open(save_path, "wb") as f:
        f.write(content)

    # Create table in DuckDB
    table_name = _sanitize_table_name(file.filename)
    db = DuckDBClient(user_id)
    try:
        result = db.load_csv(save_path, table_name)
    except Exception as e:
        db.close()
        raise HTTPException(status_code=500, detail=f"Failed to load CSV: {str(e)}")

    schema = db.get_schema(table_name)
    db.close()

    return {
        "doc_id": file.filename,
        "table_name": table_name,
        "columns": schema,
        "status": result,
    }


@router.post("/query", response_model=FinalResponse)
def handle_query(request: QueryRequest):
    """Main endpoint — triggers full LangGraph multi-agent pipeline."""

    # Derive table name from doc_id
    table_name = _sanitize_table_name(request.doc_id) if request.doc_id else None

    # Check if table exists
    db = DuckDBClient(request.user_id)
    tables = db.list_tables()
    db.close()

    if not table_name or table_name not in tables:
        raise HTTPException(
            status_code=400,
            detail=f"Table '{table_name}' not found. Available: {tables}. Please upload the dataset first."
        )

    # Invoke LangGraph pipeline
    initial_state = {
        "user_id": request.user_id,
        "doc_id": request.doc_id or "",
        "question": request.question,
        "table_name": table_name,
        "query_type": None,
        "rag_context": None,
        "sql_result": None,
        "chart_result": None,
        "final_response": None,
    }

    try:
        result = app_graph.invoke(initial_state)
    except Exception as e:
        logger.error("Pipeline failed for user %s: %s", request.user_id, str(e))
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {str(e)}")

    if result.get("final_response") is None:
        raise HTTPException(status_code=500, detail="Pipeline returned no response.")

    return result["final_response"]


@router.get("/tables/{user_id}")
def list_user_tables(user_id: str):
    """List available tables in user's DuckDB instance."""
    db = DuckDBClient(user_id)
    tables = db.list_tables()
    db.close()
    return {"user_id": user_id, "tables": tables}


@router.get("/schema/{user_id}/{table_name}")
def get_table_schema(user_id: str, table_name: str):
    """Return table schema and categorical column samples."""
    db = DuckDBClient(user_id)
    try:
        schema = db.get_schema(table_name)
        samples = db.get_categorical_samples(table_name)
    except ValueError as e:
        db.close()
        raise HTTPException(status_code=404, detail=str(e))
    db.close()
    return {"table_name": table_name, "schema": schema, "samples": samples}
