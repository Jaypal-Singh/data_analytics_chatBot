import os
import re
import json
import uuid
import base64
import io
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Internal Imports
from database.duckdb_manager import DuckDBClient
from database.vector_db import VectorDBManager
from core.build_graph import app_graph
from core.llm_client import llm
from schemas.models import FinalResponse

# ──────────────────────────────────────────────────────────────────────────────
# Disk Persistence Helpers for User Threads
# ──────────────────────────────────────────────────────────────────────────────
THREADS_STORAGE_DIR = os.path.join("storage", "threads")

def _clean_json_val(val):
    """Convert pandas/numpy NaN or invalid floats to JSON null."""
    if pd.isna(val):
        return None
    return val

def load_user_threads_from_disk(user_id: str) -> dict:
    """Disk (JSON) se user ke threads load karo taaki 2 din baad bhi chat history & docs same milein."""
    os.makedirs(THREADS_STORAGE_DIR, exist_ok=True)
    file_path = os.path.join(THREADS_STORAGE_DIR, f"{user_id}.json")
    if os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                for tid, tdata in data.items():
                    for msg in tdata.get("messages", []):
                        if "table_data" in msg and msg["table_data"]:
                            td = msg["table_data"]
                            msg["dataframe"] = pd.DataFrame(data=td.get("data"), columns=td.get("columns"))
                        else:
                            msg["dataframe"] = None
                return data
        except Exception as e:
            print(f"Warning: Failed to parse thread JSON file {file_path}: {e}")

    default_threads = {
        "thread_1": {
            "name": "Chat Thread 1",
            "doc": None,
            "messages": []
        }
    }
    save_user_threads_to_disk(user_id, default_threads)
    return default_threads

def save_user_threads_to_disk(user_id: str, threads_data: dict):
    """User ke sare threads, active doc, aur chat history ko disk (JSON) pe permanently save karo."""
    os.makedirs(THREADS_STORAGE_DIR, exist_ok=True)
    file_path = os.path.join(THREADS_STORAGE_DIR, f"{user_id}.json")
    try:
        clean_threads = {}
        for tid, tdata in threads_data.items():
            clean_msgs = []
            for msg in tdata.get("messages", []):
                df = msg.get("dataframe")
                table_data = None
                if isinstance(df, pd.DataFrame) and not df.empty:
                    clean_rows = [[_clean_json_val(val) for val in row] for row in df.values]
                    table_data = {
                        "columns": list(df.columns),
                        "data": clean_rows
                    }
                clean_msgs.append({
                    "role": msg.get("role"),
                    "content": msg.get("content"),
                    "query_type": msg.get("query_type"),
                    "sql_query": msg.get("sql_query"),
                    "chart_base64": msg.get("chart_base64"),
                    "table_data": table_data
                })
            clean_threads[tid] = {
                "name": tdata.get("name"),
                "doc": tdata.get("doc"),
                "messages": clean_msgs
            }
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(clean_threads, f, indent=2)
    except Exception as e:
        print(f"Error saving threads to disk: {e}")

# ──────────────────────────────────────────────────────────────────────────────
# Page Setup & Styling
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Data Analytics & Doc ChatBot",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.0rem;
        font-weight: 700;
        color: #1E88E5;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 0.95rem;
        color: #666666;
        margin-bottom: 1.2rem;
    }
    .badge {
        display: inline-block;
        padding: 0.25em 0.6em;
        font-size: 75%;
        font-weight: 700;
        border-radius: 10px;
        color: white;
        background-color: #4CAF50;
        margin-bottom: 10px;
    }
    .thread-card {
        padding: 10px;
        border-radius: 8px;
        background-color: #f0f2f6;
        margin-bottom: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def sanitize_table_name(filename: str) -> str:
    """Filename to valid DuckDB table name."""
    name = os.path.splitext(os.path.basename(filename))[0]
    name = re.sub(r"[^a-zA-Z0-9_]", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name.lower() or "table_1"

def extract_text_from_file(file) -> str:
    """Extract text from uploaded text or pdf document."""
    file_name = file.name.lower()
    if file_name.endswith(".txt"):
        return file.read().decode("utf-8", errors="ignore")
    elif file_name.endswith(".pdf"):
        try:
            from pypdf import PdfReader
            reader = PdfReader(file)
            return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
        except Exception as e:
            st.error(f"Error reading PDF: {e}")
            return ""
    elif file_name.endswith(".docx"):
        try:
            import docx
            doc = docx.Document(file)
            return "\n".join([p.text for p in doc.paragraphs])
        except Exception as e:
            st.error(f"Error reading DOCX: {e}")
            return ""
    return ""

def split_text_into_chunks(text: str, chunk_size: int = 500) -> list[str]:
    """Split text into sentences or fixed character chunks."""
    paragraphs = text.split("\n\n")
    chunks = []
    current_chunk = ""
    for p in paragraphs:
        if len(current_chunk) + len(p) <= chunk_size:
            current_chunk += "\n" + p
        else:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            current_chunk = p
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    return chunks

# ──────────────────────────────────────────────────────────────────────────────
# Thread State Management
# ──────────────────────────────────────────────────────────────────────────────
if "threads" not in st.session_state:
    st.session_state.threads = {}  # { user_id: { thread_id: {"name": str, "doc": str, "messages": []} } }

if "active_thread_id" not in st.session_state:
    st.session_state.active_thread_id = None

# ──────────────────────────────────────────────────────────────────────────────
# Sidebar: User & Thread Controls
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/database--v1.png", width=64)
    st.title("Data & Doc Control")

    user_id = st.text_input("User ID / Workspace", value="user123", help="Unique User ID for chat threads & database storage")

    # Load user threads from disk if not present in session state
    if user_id not in st.session_state.threads:
        st.session_state.threads[user_id] = load_user_threads_from_disk(user_id)
        st.session_state.active_thread_id = list(st.session_state.threads[user_id].keys())[0]

    user_threads = st.session_state.threads[user_id]

    # Ensure valid active thread ID
    if st.session_state.active_thread_id not in user_threads:
        st.session_state.active_thread_id = list(user_threads.keys())[0]

    st.divider()

    # ── Thread Selection & Creation ──
    st.subheader("💬 Chat Threads")

    col1, col2 = st.columns([3, 1])
    with col1:
        thread_options = {
            tid: f"🧵 {tdata['name']}" + (f" ({tdata['doc']})" if tdata['doc'] else "")
            for tid, tdata in user_threads.items()
        }
        selected_thread_id = st.selectbox(
            "Select Thread",
            options=list(thread_options.keys()),
            format_func=lambda x: thread_options[x],
            index=list(thread_options.keys()).index(st.session_state.active_thread_id)
        )
        st.session_state.active_thread_id = selected_thread_id

    with col2:
        if st.button("➕ New", help="Create a new chat thread"):
            new_id = f"thread_{uuid.uuid4().hex[:6]}"
            new_count = len(user_threads) + 1
            user_threads[new_id] = {
                "name": f"Chat Thread {new_count}",
                "doc": None,
                "messages": []
            }
            st.session_state.active_thread_id = new_id
            save_user_threads_to_disk(user_id, user_threads)
            st.rerun()

    active_thread = user_threads[st.session_state.active_thread_id]

    # Display active doc badge
    if active_thread["doc"]:
        st.info(f"📄 **Associated File:** `{active_thread['doc']}`")
    else:
        st.caption("ℹ️ No document associated with this thread yet.")

    st.divider()

    # ── Upload File for Active Thread ──
    st.subheader("📁 Upload File for Thread")
    uploaded_file = st.file_uploader(
        "Upload CSV, PDF, TXT or DOCX",
        type=["csv", "txt", "pdf", "docx"],
        accept_multiple_files=False,
        key=f"uploader_{user_id}_{st.session_state.active_thread_id}",
        help="File uploaded here will be linked to the current active chat thread."
    )

    if uploaded_file is not None:
        file_name = uploaded_file.name

        if active_thread["doc"] != file_name:
            upload_dir = os.path.join("storage", "uploads", user_id)
            os.makedirs(upload_dir, exist_ok=True)
            save_path = os.path.join(upload_dir, file_name)

            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            if file_name.lower().endswith(".csv"):
                table_name = sanitize_table_name(file_name)
                db = DuckDBClient(user_id)
                try:
                    res = db.load_csv(save_path, table_name)
                    active_thread["doc"] = file_name
                    active_thread["name"] = f"Chat: {file_name}"
                    save_user_threads_to_disk(user_id, user_threads)
                    st.success(f"✅ CSV linked: `{table_name}`")
                except Exception as e:
                    st.error(f"Failed to load CSV: {e}")
                finally:
                    db.close()
            else:
                with st.spinner("Indexing document..."):
                    raw_text = extract_text_from_file(uploaded_file)
                    if raw_text:
                        chunks = split_text_into_chunks(raw_text)
                        try:
                            vdb = VectorDBManager(user_id)
                            vdb.add_documents(
                                texts=chunks,
                                metadatas=[{"type": "doc_content"}] * len(chunks),
                                doc_id=file_name
                            )
                            active_thread["doc"] = file_name
                            active_thread["name"] = f"Doc: {file_name}"
                            save_user_threads_to_disk(user_id, user_threads)
                            st.success(f"✅ Indexed `{file_name}` ({len(chunks)} chunks)")
                        except Exception as e:
                            st.error(f"Failed to index document: {e}")

    st.divider()

    # ── Loaded Tables Browser ──
    st.subheader("📊 Loaded Tables")
    db = DuckDBClient(user_id)
    tables = db.list_tables()
    db.close()

    if tables:
        selected_table = st.selectbox("Select table to inspect", tables)
        if selected_table and st.button("Preview Schema & Data"):
            db = DuckDBClient(user_id)
            schema = db.get_schema(selected_table)
            st.write("**Schema:**")
            st.json(schema)
            rows_df = db.query_to_df(f'SELECT * FROM "{selected_table}" LIMIT 10')
            st.write("**Sample Rows (First 10):**")
            st.dataframe(rows_df)
            db.close()
    else:
        st.info("No CSV tables loaded yet.")

    st.divider()

    # ── Clear Current Thread History ──
    if st.button("🗑️ Clear Thread Messages"):
        active_thread["messages"] = []
        save_user_threads_to_disk(user_id, user_threads)
        st.rerun()

# ──────────────────────────────────────────────────────────────────────────────
# Main View: Thread Chat & Analytics
# ──────────────────────────────────────────────────────────────────────────────
thread_title = active_thread["name"]
assoc_doc = active_thread["doc"] or "None"

st.markdown(f'<div class="main-header">📊 Chat - {thread_title}</div>', unsafe_allow_html=True)
st.markdown(f'<div class="sub-header">User: <b>{user_id}</b> | Associated Document: <b>{assoc_doc}</b></div>', unsafe_allow_html=True)

# Display existing messages for active thread
current_messages = active_thread["messages"]

for msg in current_messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(msg["content"])
        else:
            if "query_type" in msg and msg["query_type"]:
                st.markdown(f"<span class='badge'>Type: {msg['query_type'].upper()}</span>", unsafe_allow_html=True)

            st.markdown(msg["content"])

            if "sql_query" in msg and msg["sql_query"]:
                with st.expander("🔍 View Generated SQL Query"):
                    st.code(msg["sql_query"], language="sql")

            if "dataframe" in msg and msg["dataframe"] is not None and not msg["dataframe"].empty:
                st.markdown("**📊 Query Result Table:**")
                st.dataframe(msg["dataframe"])

            if "chart_base64" in msg and msg["chart_base64"]:
                st.markdown("**📈 Visual Chart:**")
                image_bytes = base64.b64decode(msg["chart_base64"])
                st.image(image_bytes)

# Chat Input
user_question = st.chat_input(f"Ask a question about {assoc_doc}...")

if user_question:
    # Append user question to active thread
    current_messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    # Process query
    with st.chat_message("assistant"):
        with st.spinner("Analyzing data and generating answer..."):
            db = DuckDBClient(user_id)
            available_tables = db.list_tables()
            db.close()

            # Figure out target table/doc for active thread
            target_doc = active_thread["doc"]
            if target_doc and target_doc.lower().endswith(".csv"):
                target_table = sanitize_table_name(target_doc)
            elif available_tables:
                target_table = available_tables[0]
            else:
                target_table = "default"

            recent_chat_history = current_messages[-6:]

            initial_state = {
                "user_id": user_id,
                "doc_id": target_doc or target_table,
                "question": user_question,
                "table_name": target_table,
                "chat_history": recent_chat_history,
                "query_type": None,
                "rag_context": None,
                "sql_result": None,
                "chart_result": None,
                "final_response": None,
            }

            try:
                result = app_graph.invoke(initial_state)
                final_resp: FinalResponse = result.get("final_response")

                if final_resp:
                    query_type = final_resp.query_type
                    sql_res = result.get("sql_result")
                    chart_res = result.get("chart_result")

                    df_result = None
                    if final_resp.data_table and final_resp.data_table.rows:
                        df_result = pd.DataFrame(
                            final_resp.data_table.rows,
                            columns=final_resp.data_table.columns
                        )

                    sql_query_str = sql_res.sql_query if (sql_res and sql_res.success) else None
                    chart_b64 = final_resp.chart_base64

                    # Render query type badge
                    if query_type:
                        st.markdown(f"<span class='badge'>Type: {query_type.value.upper()}</span>", unsafe_allow_html=True)

                    # Stream text token-by-token with memory!
                    db = DuckDBClient(user_id)
                    schema_context = db.get_full_context()
                    db.close()
                    data_preview = f"DATASET SCHEMA & FEATURES:\n{schema_context}\n\n"
                    if sql_res and sql_res.success:
                        data_preview += f"SQL QUERY EXECUTED:\n{sql_res.sql_query}\n\nQUERY RESULTS:\nColumns: {sql_res.columns}\nRows Sample: {sql_res.rows[:10]}"

                    summary_text = st.write_stream(llm.generate_summary_stream(user_question, data_preview, recent_chat_history))

                    if sql_query_str:
                        with st.expander("🔍 View Generated SQL Query"):
                            st.code(sql_query_str, language="sql")

                    if df_result is not None and not df_result.empty:
                        st.markdown("**📊 Query Result Table:**")
                        st.dataframe(df_result)

                    if chart_b64:
                        st.markdown("**📈 Visual Chart:**")
                        image_bytes = base64.b64decode(chart_b64)
                        st.image(image_bytes)

                    # Save in active thread chat history
                    current_messages.append({
                        "role": "assistant",
                        "content": summary_text,
                        "query_type": query_type.value if query_type else None,
                        "sql_query": sql_query_str,
                        "dataframe": df_result,
                        "chart_base64": chart_b64,
                    })
                    save_user_threads_to_disk(user_id, user_threads)

                else:
                    err_msg = "Could not process request. Please upload a dataset or document for this thread."
                    st.error(err_msg)
                    current_messages.append({"role": "assistant", "content": err_msg})
                    save_user_threads_to_disk(user_id, user_threads)

            except Exception as e:
                err_msg = f"An error occurred while processing: {str(e)}"
                st.error(err_msg)
                current_messages.append({"role": "assistant", "content": err_msg})
                save_user_threads_to_disk(user_id, user_threads)
