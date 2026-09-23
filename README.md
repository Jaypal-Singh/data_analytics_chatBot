# 🤖 Autonomous Data Analytics & Document AI ChatBot

An intelligent, multi-agent AI system for interactive data analytics, automated SQL generation, document RAG (Retrieval-Augmented Generation), and real-time visualization generation built with **LangGraph**, **DuckDB**, **FAISS**, and **Streamlit**.

---

## 🌟 Key Features

- 📊 **Natural Language to SQL (DuckDB Engine)**: Convert natural language questions into optimized SQL queries and execute them against uploaded datasets automatically.
- 🔄 **Self-Healing SQL Execution**: Automatically catches database syntax errors or missing table/column errors, feeds the stack trace back to the LLM agent, and self-corrects up to maximum retry limits.
- 📚 **Document RAG System**: Upload PDFs/Text documents and ask contextual questions. Powered by vector embeddings and semantic search.
- 📈 **Automated Data Visualization**: Automatically detects analytical intent and generates publication-grade charts (Matplotlib / Seaborn) alongside query results.
- ⚡ **Real-Time Token Streaming**: Stream responses word-by-word with `st.write_stream` for a fast and responsive user experience.
- 🧠 **Multi-Turn Chat & Thread Memory**: Maintains per-user thread history (`user_id` + `thread_id`), restoring previous conversations, uploaded datasets, and active documents across sessions.
- 🔀 **Smart Supervisor Router**: Intelligently classifies user queries (`GENERAL`, `ANALYTICAL`, `DOCUMENT`, `HYBRID`) and routes them through specialized processing nodes.

---

## 🏗️ Architecture & Workflow

The system implements an end-to-end multi-agent pipeline designed for both structured (CSV/Excel) and unstructured (PDF/Doc) data analytics:

```mermaid
flowchart TD
    %% Upload / Data Ingestion Branch
    subgraph Data_Ingestion ["1. Data Ingestion & Storage"]
        UP[Upload DOC / CSV]
        UP -->|Unstructured Data| EMB[Embedding Document Content] --> VDB[(Vector DB / ChromaDB / FAISS)]
        UP -->|Structured Data| DSTORE[Store Data Tables] --> DDB[(DuckDB Database)]
    end

    %% Supervisor Agent Router
    UQ[User Question\nGeneral / Analytical / Graph / Table] --> SUP[Supervisor Agent]
    
    subgraph Supervisor_Layer ["2. Supervisor Agent Router"]
        SUP --> SUP_LLM["LLM Router\nIdentify Question Nature & Context"]
        SUP_LLM --> CLASSIFY{"Query Classification"}
        CLASSIFY -->|Analytical / SQL Needed| RAG[RAG Researcher\nFetch Column Meanings & Context]
        CLASSIFY -->|Both Analytical & General| PARALLEL[Parallel Execution Mode]
    end

    %% SQL Generation & Execution Loop
    subgraph SQL_Pipeline ["3. SQL Agent & DuckDB Execution Engine"]
        RAG --> SQL_AGENT[SQL Agent + RAG Context\nGenerate SQL Query]
        SQL_AGENT --> LLM_EXEC[LLM Call with Data & Schema Context]
        LLM_EXEC --> RUN_DUCKDB[Run Query on DuckDB]
        
        %% Self Healing SQL Loop
        RUN_DUCKDB -->|Query Error| ERR_RETRY[Call LLM again + with Error & Context]
        ERR_RETRY --> SQL_AGENT
    end

    RUN_DUCKDB -->|No Error| FETCHED[DuckDB Fetched Data\nQuery Type: Analytical / General / RAG Context]

    %% Parallel Execution Path
    PARALLEL --> PAR_ANALYTICAL[Analytical Branch: LLM Call with Prompt] --> PAR_READ1[Convert into Readable Format]
    PARALLEL --> PAR_GENERAL[General Branch] --> PAR_READ2[Convert into Readable Format]
    PAR_READ1 --> REPORTING
    PAR_READ2 --> REPORTING

    %% Analytical vs General Branching post-SQL
    FETCHED -->|Analytical Query| MATPLOT_PROMPT[LLM Call Prompt:\nGenerate Matplotlib Code]
    FETCHED -->|General Query| GEN_READABLE[Convert into Readable Format]

    %% Python Subprocess & Chart Generation Loop
    subgraph Chart_Pipeline ["4. Visualization Subprocess Engine"]
        MATPLOT_PROMPT --> SUBPROC[Run Subprocess & Execute Python Code\nGenerate Graph Artifact]
        SUBPROC -->|Subprocess Error| ERR_PY[Call LLM with Error] --> MATPLOT_PROMPT
        SUBPROC -->|No Error| IMG_FMT[Make Image Formatted]
    end

    %% Reporting & Synthesis
    IMG_FMT --> REPORTING[Reporting Agent:\nMake Full Summary & Proper Answer]
    GEN_READABLE --> REPORTING
    REPORTING --> UI[Streamlit UI / Token Streaming Output]
```

---

### 🔄 Workflow Step-by-Step Breakdown

1. **Data Ingestion**:
   - **Unstructured Data (Docs/PDFs)**: Processed via embeddings and stored in Vector DB (ChromaDB / FAISS).
   - **Structured Data (CSVs/Excel)**: Loaded into DuckDB for ultra-fast SQL execution.

2. **Supervisor Agent**:
   - Identifies question nature using LLM and context.
   - Determines query output requirements: SQL query, analytical graph, or general text summary.
   - Triggers **Parallel Execution** if user requests both analytical charts and text summaries simultaneously.

3. **RAG Researcher & Self-Healing SQL Agent**:
   - RAG Researcher retrieves relevant column meanings, table schemas, and domain context.
   - SQL Agent generates DuckDB-compliant SQL query.
   - **Self-Healing Loop**: If DuckDB throws a syntax or execution error, the error traceback is sent back to the LLM for automated self-correction.

4. **Visualization Engine (Subprocess Execution)**:
   - For analytical graph queries, the LLM generates Matplotlib/Seaborn python code.
   - Code is executed in an isolated **Subprocess**.
   - If a execution error occurs, the error is fed back to the LLM to self-correct the code until a formatted chart image is generated.

5. **Reporting Agent**:
   - Combines DuckDB analytical results, formatted chart images, and RAG document contexts to synthesize a comprehensive final answer delivered token-by-token on Streamlit UI.

---

## 📁 Repository Structure

```
.
├── app/                  # Application initialization & config
├── agents/               # LangGraph Agent implementations & node logic
├── core/                 # Core Graph state, LLM clients, and workflow builder
│   ├── build_graph.py    # LangGraph execution pipeline setup
│   ├── graph_state.py    # TypedDict state structure
│   └── llm_client.py     # LLM provider wrapper & streaming helper
├── database/             # Storage management
│   ├── duckdb_manager.py # DuckDB client & query executor
│   └── vector_db.py      # FAISS vector store & document indexing
├── prompts/              # System prompts & template definitions
│   ├── chart_prompts.py
│   ├── reporting_prompts.py
│   ├── sql_prompts.py
│   └── supervisor_prompts.py
├── schemas/              # Pydantic data models & state schemas
├── storage/              # Runtime user storage (ignored in Git)
│   ├── threads/          # Per-user thread histories & charts
│   └── uploads/          # Uploaded CSVs, Excel, and PDFs
├── main.py               # CLI entrypoint
├── streamlit_app.py      # Streamlit Web UI
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```

---

## 🚀 Getting Started

### 1. Prerequisites
- **Python**: 3.10 or higher
- **NVIDIA GPU** *(Optional for local LLMs, CPU works fine with cloud API keys)*

### 2. Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/Jaypal-Singh/data_analytics_chatBot.git
cd data_analytics_chatBot

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install required packages
pip install -r requirements.txt
```

### 3. Environment Configuration

Create a `.env` file in the root directory and add your API credentials:

```env
OPENAI_API_KEY=your_openai_api_key_here
# Or GROQ_API_KEY / ANTHROPIC_API_KEY depending on your LLM client configuration
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_api_key_here  # Optional for LangSmith tracking
```

---

## 🖥️ Running the Application

Launch the Streamlit Web Application:

```bash
streamlit run streamlit_app.py --server.port 8501
```

Access the application in your browser at `http://localhost:8501`.

---

## 💡 How to Use

1. **User Identity & Threads**: Enter or select a `User ID` and `Thread ID` in the sidebar to load or isolate conversation state.
2. **Upload Data**: Use the sidebar uploader to add `.csv`, `.xlsx`, or `.pdf` files.
3. **Ask Analytical Questions**:
   - *"What are top 5 highest sales regions?"* -> Triggers DuckDB SQL + Chart node.
   - *"Summarize the uploaded document"* -> Triggers Document RAG node.
   - *"Hi, who are you?"* -> Triggers General conversation node.

---

## 🛡️ License & Contributing

Distributed under the MIT License. Contributions and feature suggestions are welcome via GitHub Pull Requests!
