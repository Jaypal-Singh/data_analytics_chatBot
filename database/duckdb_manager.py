import duckdb
import os
import logging
from core.config import DATABASE_PATH

logger = logging.getLogger(__name__)


class DuckDBClient:
    def __init__(self, user_id: str):
        """create DuckDb Databse for each user"""
        self.user_id = user_id
        db_dir = os.path.join(DATABASE_PATH, str(user_id))
        os.makedirs(db_dir, exist_ok=True)
        self.db_path = os.path.join(db_dir, "analytics.db")
        self.con = duckdb.connect(self.db_path)

    # Load Data into DuckDb and return table name and how many rows
    def load_csv(self, file_path: str, table_name: str) -> str:
       
        # Try 1: DuckDB native read_csv_auto with ignore_errors=true
        try:
            self.con.execute(
                f'CREATE OR REPLACE TABLE "{table_name}" AS '
                f"SELECT * FROM read_csv_auto(?, ignore_errors=true)",
                [file_path]
            )
            count = self.con.execute(f'SELECT COUNT(*) FROM "{table_name}"').fetchone()[0]
            return f"Table '{table_name}' created with {count} rows."
        except Exception as e1:
            logger.warning("DuckDB native read_csv_auto failed: %s. Trying multi-encoding pandas fallback.", str(e1))

        # Try 2: Pandas fallback
        import pandas as pd
        for enc in ['utf-8-sig', 'latin1', 'iso-8859-1', 'cp1252']:
            try:
                df = pd.read_csv(file_path, encoding=enc, encoding_errors='replace', low_memory=False)
                self.con.execute(f'CREATE OR REPLACE TABLE "{table_name}" AS SELECT * FROM df')
                count = self.con.execute(f'SELECT COUNT(*) FROM "{table_name}"').fetchone()[0]
                return f"Table '{table_name}' created with {count} rows (encoding: {enc})."
            except Exception:
                continue

        # Try 3: Last resort pandas python engine with skipped bad lines
        df = pd.read_csv(file_path, engine='python', on_bad_lines='skip', encoding_errors='ignore')
        self.con.execute(f'CREATE OR REPLACE TABLE "{table_name}" AS SELECT * FROM df')
        count = self.con.execute(f'SELECT COUNT(*) FROM "{table_name}"').fetchone()[0]
        return f"Table '{table_name}' created with {count} rows."

    # return Schema & Samples for LLM column name + data type
    def get_schema(self, table_name: str) -> list[dict]:
        tables = self.list_tables()
        if table_name not in tables:
            raise ValueError(f"Table '{table_name}' not found.")
        result = self.con.execute(f'DESCRIBE "{table_name}"').fetchall()
        return [{"column": row[0], "type": row[1]} for row in result]

    #return distinct value of categorial column for LLM prompt
    def get_categorical_samples(self, table_name: str, max_distinct: int = 10) -> dict:
        schema = self.get_schema(table_name)
        samples = {}
        for col in schema:
            if col["type"] in ("VARCHAR", "TEXT"):
                distinct = self.con.execute(
                    f'SELECT DISTINCT "{col["column"]}" FROM "{table_name}" LIMIT ?',
                    [max_distinct]
                ).fetchall()
                samples[col["column"]] = [v[0] for v in distinct if v[0] is not None]
        return samples

    # Run SQL Query and return result
    def run_query(self, sql: str) -> dict:
        try:
            result = self.con.execute(sql)
            columns = [desc[0] for desc in result.description]
            rows = result.fetchall()
            return {"success": True, "columns": columns, "rows": rows}
        except Exception as e:
            logger.error("Query error for user %s: %s", self.user_id, str(e))
            return {"success": False, "error": str(e)}

    # Run SQL Query and return pandas DataFrame
    def query_to_df(self, sql: str):
        import pandas as pd
        try:
            return self.con.execute(sql).df()
        except Exception as e:
            logger.error("Error converting query to df for user %s: %s", self.user_id, str(e))
            return pd.DataFrame()

    # return Table Names
    def list_tables(self) -> list[str]:
        result = self.con.execute("SHOW TABLES").fetchall()
        return [row[0] for row in result]

    # return Full Context for LLM prompt
    def get_full_context(self) -> str:
        tables = self.list_tables()
        if not tables:
            return "No tables found in database."

        context_parts = []
        for table in tables:
            schema = self.get_schema(table)
            samples = self.get_categorical_samples(table)

            col_lines = [f"  - {c['column']} ({c['type']})" for c in schema]
            part = f"TABLE: {table}\nCOLUMNS:\n" + "\n".join(col_lines)

            if samples:
                sample_lines = [f"  - {col}: {vals}" for col, vals in samples.items()]
                part += "\nSAMPLE VALUES:\n" + "\n".join(sample_lines)

            context_parts.append(part)

        return "\n\n".join(context_parts)

    # close connection
    def close(self):
        if self.con:
            self.con.close()