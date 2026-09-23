import json
import requests
import logging
from langsmith import traceable
from core.config import OLLAMA_CHAT_URL, MODEL_NAME
from schemas.models import QueryType
from prompts.supervisor_prompts import CLASSIFY_SYSTEM_PROMPT
from prompts.sql_prompts import GENERATE_SQL_SYSTEM_PROMPT, FIX_SQL_SYSTEM_PROMPT
from prompts.chart_prompts import GENERATE_CHART_SYSTEM_PROMPT, FIX_CHART_SYSTEM_PROMPT
from prompts.reporting_prompts import SUMMARY_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


class LLMClient:
    def __init__(self, model_name: str = MODEL_NAME):
        self.model_name = model_name

    @traceable(name="Ollama LLM Call")
    def _call(self, system_prompt: str, user_prompt: str) -> str:
        """Ollama API se LLM call karo — response text return hoga."""
        try:
            response = requests.post(
                OLLAMA_CHAT_URL,
                json={
                    "model": self.model_name,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "stream": False,
                },
            )
            response.raise_for_status()
            return response.json()["message"]["content"].strip()
        except requests.RequestException as e:
            logger.error("LLM call failed: %s", str(e))
            raise

    @traceable(name="Ollama LLM Stream")
    def _call_stream(self, system_prompt: str, user_prompt: str):
        """Ollama API se token-by-token stream yield karo."""
        try:
            response = requests.post(
                OLLAMA_CHAT_URL,
                json={
                    "model": self.model_name,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "stream": True,
                },
                stream=True
            )
            response.raise_for_status()
            for line in response.iter_lines():
                if line:
                    data = json.loads(line.decode("utf-8"))
                    chunk = data.get("message", {}).get("content", "")
                    if chunk:
                        yield chunk
        except Exception as e:
            logger.error("LLM streaming failed: %s", str(e))
            yield f" [Error streaming: {str(e)}]"

    def _format_chat_history(self, chat_history: list[dict] = None) -> str:
        """Chat history ko LLM prompt ke liye string format mein covert karo."""
        if not chat_history:
            return "No previous conversation history."
        formatted = []
        for msg in chat_history[-6:]:
            role = "User" if msg.get("role") == "user" else "Assistant"
            content = msg.get("content", "")
            if content:
                # Shorten long past messages if needed
                short_content = content[:300] + ("..." if len(content) > 300 else "")
                formatted.append(f"{role}: {short_content}")
        return "\n".join(formatted)

    # Supervisor is it general or analytical or both
    @traceable(name="Classify Query")
    def classify_query(self, question: str, chat_history: list[dict] = None) -> QueryType:
        history_str = self._format_chat_history(chat_history)
        prompt = f"PAST CONVERSATION:\n{history_str}\n\nCURRENT QUESTION:\n{question}"
        result = self._call(CLASSIFY_SYSTEM_PROMPT, prompt).lower().strip()
        for qt in ("general", "analytical", "both"):
            if qt in result:
                return QueryType(qt)
        return QueryType.GENERAL

    # SQL
    @traceable(name="Generate SQL")
    def generate_sql(self, question: str, schema: str, rag_context: str, chat_history: list[dict] = None) -> str:
        """Schema + context + chat history to generate SQl quiry."""
        history_str = self._format_chat_history(chat_history)
        full_rag = f"{rag_context}\n\nPAST CONVERSATION HISTORY:\n{history_str}"
        system = GENERATE_SQL_SYSTEM_PROMPT.format(schema=schema, rag_context=full_rag)
        sql = self._call(system, question)
        sql = sql.replace("```sql", "").replace("```", "").strip()
        return sql

    # if SQL Query failed thne SQL query fix
    @traceable(name="Fix SQL")
    def fix_sql(self, failed_query: str, error: str, schema: str) -> str:
        system = FIX_SQL_SYSTEM_PROMPT.format(
            failed_query=failed_query, error=error, schema=schema
        )
        sql = self._call(system, "Fix the query above.")
        sql = sql.replace("```sql", "").replace("```", "").strip()
        return sql

    # Chart visualization - matplotlib
    @traceable(name="Generate Chart Code")
    def generate_chart_code(self, columns: list, sample_rows: list) -> str:
      
        system = GENERATE_CHART_SYSTEM_PROMPT.format(
            columns=columns, sample_rows=sample_rows[:3]
        )
        code = self._call(system, "Generate chart code for this data.")
        code = code.replace("```python", "").replace("```", "").strip()
        return code

    @traceable(name="Fix Chart Code")
    def fix_chart_code(self, failed_code: str, error: str) -> str:
        system = FIX_CHART_SYSTEM_PROMPT.format(failed_code=failed_code, error=error)
        code = self._call(system, "Fix the code above.")
        code = code.replace("```python", "").replace("```", "").strip()
        return code

    # Reporting 

    @traceable(name="Generate Summary")
    def generate_summary(self, question: str, data_preview: str, chat_history: list[dict] = None) -> str:
        """Data + chat history se human-readable summary banao."""
        history_str = self._format_chat_history(chat_history)
        user_prompt = f"PAST CONVERSATION:\n{history_str}\n\nQuestion: {question}\nData: {data_preview}"
        return self._call(SUMMARY_SYSTEM_PROMPT, user_prompt)

    @traceable(name="Stream Summary")
    def generate_summary_stream(self, question: str, data_preview: str, chat_history: list[dict] = None):
        """Data + chat history se summary stream karo (token-by-token)."""
        history_str = self._format_chat_history(chat_history)
        user_prompt = f"PAST CONVERSATION:\n{history_str}\n\nQuestion: {question}\nData: {data_preview}"
        return self._call_stream(SUMMARY_SYSTEM_PROMPT, user_prompt)


# Shared instance
llm = LLMClient()
