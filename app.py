import streamlit as st
import pandas as pd
import sqlite3
import re
import json
import plotly.express as px
from groq import Groq

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI SQL Data Analyst",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }
    .main-header {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        padding: 2rem; border-radius: 15px; margin-bottom: 2rem;
        color: white; text-align: center;
    }
    .main-header h1 {
        font-size: 2.5rem; font-weight: 700;
        background: linear-gradient(90deg, #a78bfa, #38bdf8);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .sql-box {
        background: #1e1e2e; color: #cdd6f4; padding: 1rem;
        border-radius: 10px; font-family: 'Courier New', monospace;
        border-left: 4px solid #a78bfa; font-size: 0.9rem; white-space: pre-wrap;
    }
    .answer-box {
        background: linear-gradient(135deg, #1a1a2e, #16213e); color: #e2e8f0;
        padding: 1.5rem; border-radius: 12px; border-left: 4px solid #38bdf8; margin-top: 1rem;
    }
    .stButton>button {
        background: linear-gradient(135deg, #a78bfa, #38bdf8);
        color: white; font-weight: 600; border-radius: 8px; border: none;
        padding: 0.5rem 2rem; width: 100%; font-size: 1rem;
    }
    .stButton>button:hover { opacity: 0.9; transform: translateY(-1px); }
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border-radius: 10px; padding: 1rem; border: 1px solid #334155; text-align: center;
    }
    .warning-box {
        background: linear-gradient(135deg, #2d1b00, #3d2400); color: #fbbf24;
        padding: 1rem; border-radius: 10px; border-left: 4px solid #f59e0b; margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ─── Header ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🧠 AI SQL Data Analyst Agent</h1>
    <p style="color:#94a3b8; font-size:1.1rem;">Upload a CSV → Ask questions in plain English → Get SQL + Answers + Charts</p>
</div>
""", unsafe_allow_html=True)

# ─── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    groq_api_key = st.secrets["GROQ_API_KEY"]
    model_choice = st.selectbox(
        "🤖 LLM Model",
        ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "gemma2-9b-it"],
    )
    st.markdown("---")
    st.markdown("### 📋 How to Use")
    st.markdown("""
1. Upload a **CSV file**
2. Ask a **natural language question**
3. Get **SQL + Answer + Chart**!
    """)
    st.markdown("---")


# ─── Helper: Load CSV into SQLite ───────────────────────────────────────────
@st.cache_resource
def load_csv_to_sqlite(_df: pd.DataFrame, table_name: str = "data"):
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    _df.to_sql(table_name, conn, if_exists="replace", index=False)
    return conn

def get_schema(conn, table_name="data"):
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info({table_name})")
    cols = cursor.fetchall()
    schema_lines = [f"  {c[1]} ({c[2]})" for c in cols]
    return f"Table: {table_name}\nColumns:\n" + "\n".join(schema_lines)

def get_column_names(conn, table_name="data"):
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info({table_name})")
    return [row[1].lower() for row in cursor.fetchall()]

def run_sql(conn, query):
    try:
        df_result = pd.read_sql_query(query, conn)
        return df_result, None
    except Exception as e:
        return None, str(e)


# ─── FIX 1: Validate SQL columns against actual schema ──────────────────────
def validate_sql_columns(sql, actual_columns):
    """Check if SQL uses columns that don't exist in the table."""
    # Extract all quoted and unquoted identifiers from SQL
    sql_lower = sql.lower()
    # Remove string literals to avoid false positives
    sql_clean = re.sub(r"'[^']*'", "", sql_lower)
    sql_clean = re.sub(r'"[^"]*"', "", sql_clean)

    # Keywords to ignore
    keywords = {
        "select", "from", "where", "group", "by", "order", "limit", "having",
        "count", "sum", "avg", "min", "max", "as", "and", "or", "not", "in",
        "is", "null", "like", "between", "case", "when", "then", "else", "end",
        "distinct", "inner", "left", "right", "join", "on", "data", "asc", "desc",
        "true", "false", "1", "100", "*"
    }

    # Find potential column references (words not in keywords)
    words = re.findall(r'\b([a-z_][a-z0-9_]*)\b', sql_clean)
    suspicious = [w for w in words if w not in keywords and not w.isdigit()]

    invalid = [w for w in suspicious if w not in actual_columns]
    return invalid


# ─── FIX 2: Ask Groq with retry on failure ──────────────────────────────────
def ask_groq(client, model, schema, question, sample_rows, error_feedback=None):
    system_prompt = f"""You are an expert SQLite SQL analyst. Your job is to write ONLY valid SQLite SQL queries.

IMPORTANT: The database table is named "data" and has ONLY these exact columns:
{schema}

Sample data (first 3 rows):
{sample_rows}

STRICT RULES - YOU MUST FOLLOW ALL OF THESE:
1. ONLY use column names listed in the schema above. Copy them EXACTLY including case. NEVER invent or guess column names.
2. Always use table name "data".
3. Wrap column names that conflict with SQL keywords in double quotes e.g. "name", "order", "group".
4. NEVER use LIMIT 1 unless user explicitly asks for 1 result. Always use LIMIT 100 by default.
5. For "which X is most/highest/top" questions, return ALL groups ordered by value — not just 1 row.
6. Always include a COUNT or numeric column when grouping so charts can be rendered.
7. Return ONLY valid JSON with keys: "sql", "explanation", "chart_type".
8. chart_type must be one of: bar, line, pie, scatter, none.
   - Use "bar" for comparisons and counts by category.
   - Use "pie" for percentage or share questions.
   - Use "line" for time-based trends.
   - Use "scatter" for correlations between two numbers.
   - Use "none" only for raw row lookups (SELECT * queries).
9. Do NOT include markdown, backticks, or any text outside the JSON.
"""
    messages = [{"role": "system", "content": system_prompt}]

    if error_feedback:
        # Retry with error context so AI can self-correct
        messages.append({"role": "user", "content": f"Question: {question}"})
        messages.append({"role": "assistant", "content": error_feedback["previous_response"]})
        messages.append({"role": "user", "content": f"Your previous SQL failed with error: '{error_feedback['error']}'. Please fix it using ONLY the columns in the schema and try again."})
    else:
        messages.append({"role": "user", "content": f"Question: {question}"})

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.1,
    )
    raw = response.choices[0].message.content.strip()
    raw = re.sub(r"```json|```", "", raw).strip()
    return json.loads(raw), raw


# ─── FIX 3: Detect ambiguous questions ──────────────────────────────────────
def is_ambiguous(question, actual_columns):
    """Detect if question is too vague to answer accurately."""
    question_lower = question.lower().strip()

    # Too short or generic
    if len(question_lower.split()) <= 2:
        return True, "Your question is too short. Please be more specific, e.g. 'Which category has the most orders?'"

    # Vague words with no column reference
    vague_phrases = ["show me", "tell me", "what about", "how about", "compare them", "analyze it", "give me info"]
    for phrase in vague_phrases:
        if phrase in question_lower and not any(col in question_lower for col in actual_columns):
            return True, f"Your question is too vague. Try mentioning a specific column like: {', '.join(actual_columns[:4])}..."

    return False, ""


# ─── Helper: Auto Chart ─────────────────────────────────────────────────────
def render_chart(df_result, chart_type):
    if df_result is None or df_result.empty:
        return

    num_cols = df_result.select_dtypes(include="number").columns.tolist()
    cat_cols = df_result.select_dtypes(exclude="number").columns.tolist()

    # Auto-detect chart type if AI returns "none" but data is chartable
    if chart_type == "none":
        if len(cat_cols) >= 1 and len(num_cols) >= 1:
            chart_type = "bar"
        elif len(num_cols) >= 2:
            chart_type = "scatter"

    if chart_type == "none":
        return

    try:
        if chart_type == "bar" and len(cat_cols) >= 1 and len(num_cols) >= 1:
            fig = px.bar(df_result, x=cat_cols[0], y=num_cols[0],
                         color=cat_cols[0], color_discrete_sequence=px.colors.qualitative.Pastel)
        elif chart_type == "line" and len(num_cols) >= 1:
            x_col = cat_cols[0] if cat_cols else num_cols[0]
            fig = px.line(df_result, x=x_col, y=num_cols[0], markers=True,
                          color_discrete_sequence=["#38bdf8"])
        elif chart_type == "pie" and len(cat_cols) >= 1 and len(num_cols) >= 1:
            fig = px.pie(df_result, names=cat_cols[0], values=num_cols[0])
        elif chart_type == "scatter" and len(num_cols) >= 2:
            fig = px.scatter(df_result, x=num_cols[0], y=num_cols[1],
                             color_discrete_sequence=["#f472b6"])
        else:
            if len(num_cols) == 1 and len(cat_cols) == 0:
                st.info(f"📊 Result: **{df_result[num_cols[0]].iloc[0]:,}**")
                return
            elif len(num_cols) >= 1 and len(cat_cols) >= 1:
                fig = px.bar(df_result, x=cat_cols[0], y=num_cols[0],
                             color_discrete_sequence=["#a78bfa"])
            else:
                st.info("No suitable columns for visualization.")
                return

        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0",
            margin=dict(t=30, b=30),
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"Chart could not be rendered: {e}")


# ─── Main App ────────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader("📂 Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    load_csv_to_sqlite.clear()  # Clear cache on new upload

    st.success(f"✅ Loaded **{len(df):,} rows × {len(df.columns)} columns**")

    col1, col2, col3 = st.columns(3)
    col1.metric("📊 Rows", f"{len(df):,}")
    col2.metric("🔢 Columns", len(df.columns))
    col3.metric("💾 Size", f"{uploaded_file.size / 1024:.1f} KB")

    with st.expander("🔍 Preview Data (first 10 rows)"):
        st.dataframe(df.head(10), use_container_width=True)

    conn = load_csv_to_sqlite(df)
    schema = get_schema(conn)
    actual_columns = get_column_names(conn)
    sample_rows = df.head(3).to_string(index=False)

    with st.expander("🗂️ Database Schema"):
        st.code(schema, language="sql")

    st.markdown("---")
    st.markdown("### 💬 Ask a Question")

    st.markdown("**Quick examples:**")
    num_example_col = df.select_dtypes(include='number').columns[0] if df.select_dtypes(include='number').shape[1] > 0 else df.columns[0]
    cat_example_col = df.select_dtypes(exclude='number').columns[0] if df.select_dtypes(exclude='number').shape[1] > 0 else df.columns[0]
    examples = [
        "Show me the top 5 rows",
        f"Count records by {cat_example_col}",
        f"What is the average {num_example_col}?",
    ]
    col_a, col_b, col_c = st.columns(3)
    q_select = ""
    if col_a.button(examples[0]): q_select = examples[0]
    if col_b.button(examples[1]): q_select = examples[1]
    if col_c.button(examples[2]): q_select = examples[2]

    user_question = st.text_input(
        "Ask anything about your data...",
        value=q_select,
        placeholder="e.g. Which category has the highest total sales?"
    )

    if st.button("🚀 Analyze"):
        if not groq_api_key:
            st.error("⚠️ Please enter your Groq API Key in the sidebar.")
        elif not user_question.strip():
            st.warning("Please enter a question.")
        else:
            # FIX 3: Check ambiguity before calling AI
            ambiguous, ambiguity_msg = is_ambiguous(user_question, actual_columns)
            if ambiguous:
                st.warning(f"⚠️ {ambiguity_msg}")
            else:
                with st.spinner("🤖 AI is thinking..."):
                    try:
                        client = Groq(api_key=groq_api_key)

                        # First attempt
                        result, raw_response = ask_groq(client, model_choice, schema, user_question, sample_rows)
                        sql_query = result.get("sql", "")
                        explanation = result.get("explanation", "")
                        chart_type = result.get("chart_type", "none")

                        # FIX 1: Validate columns before running SQL
                        invalid_cols = validate_sql_columns(sql_query, actual_columns)
                        if invalid_cols:
                            st.warning(f"⚠️ AI used invalid columns: `{', '.join(invalid_cols)}` — auto-retrying...")
                            result, raw_response = ask_groq(
                                client, model_choice, schema, user_question, sample_rows,
                                error_feedback={"error": f"Invalid columns used: {invalid_cols}", "previous_response": raw_response}
                            )
                            sql_query = result.get("sql", "")
                            explanation = result.get("explanation", "")
                            chart_type = result.get("chart_type", "none")

                        # Run SQL
                        df_result, error = run_sql(conn, sql_query)

                        # FIX 2: Auto-retry if SQL execution fails
                        if error:
                            st.warning("⚠️ SQL failed — auto-retrying with error context...")
                            result, _ = ask_groq(
                                client, model_choice, schema, user_question, sample_rows,
                                error_feedback={"error": error, "previous_response": raw_response}
                            )
                            sql_query = result.get("sql", "")
                            explanation = result.get("explanation", "")
                            chart_type = result.get("chart_type", "none")
                            df_result, error = run_sql(conn, sql_query)

                        # Display results
                        col_left, col_right = st.columns([1, 1])

                        with col_left:
                            st.markdown("### 🔎 Generated SQL")
                            st.markdown(f'<div class="sql-box">{sql_query}</div>', unsafe_allow_html=True)
                            st.markdown("### 💡 Explanation")
                            st.markdown(f'<div class="answer-box">{explanation}</div>', unsafe_allow_html=True)

                        with col_right:
                            st.markdown("### 📋 Query Results")
                            if error:
                                st.error(f"SQL Error after retry: {error}")
                            else:
                                st.dataframe(df_result, use_container_width=True)

                        # Auto Visualization
                        if df_result is not None and not df_result.empty:
                            st.markdown("### 📊 Visualization")
                            render_chart(df_result, chart_type)

                    except json.JSONDecodeError:
                        st.error("⚠️ AI returned an unexpected format. Try rephrasing your question.")
                    except Exception as e:
                        st.error(f"❌ Error: {e}")

    st.markdown("---")
    st.markdown("### 🛠️ Direct SQL Query")
    direct_sql = st.text_area("Write your own SQL query:", placeholder='SELECT * FROM data LIMIT 10')
    if st.button("▶️ Run SQL"):
        if direct_sql.strip():
            df_r, err = run_sql(conn, direct_sql)
            if err:
                st.error(err)
            else:
                st.dataframe(df_r, use_container_width=True)
else:
    st.info("👆 Please upload a CSV file to get started.")
    st.markdown("""
    ### 🎯 What this agent can do:
    - **Understand** your CSV data structure automatically
    - **Convert** natural language questions to SQL queries
    - **Execute** queries on an in-memory SQLite database
    - **Visualize** results with the best-fit chart type
    - **Explain** results in plain English
    """)
