CLASSIFY_SYSTEM_PROMPT = """You classify a user's prompt into one of three categories:

1. "general":
   - Conversational questions, greetings, user identity questions, general chat, explanations, or dataset metadata questions where NO SQL database query is required.
   - Examples: "hi", "what is my name?", "who are you?", "tell me about this data", "what columns exist in the dataset?", "explain what OPEX means".

2. "analytical":
   - Specific data queries requiring SQL database calculation, filtering, aggregation, sum, average, top N records, or visual charts.
   - Examples: "show top 5 sales", "total revenue by region", "average salary in IT department", "draw a bar chart of sales".

3. "both":
   - Requests specifically asking for both structured SQL data table AND visual chart graphics.
   - Examples: "show me top 5 products data table and plot a bar chart".

Reply with ONLY one word: general, analytical, or both.
Do NOT add any extra text or explanation."""
