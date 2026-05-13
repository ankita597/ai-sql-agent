import streamlit as st
import pandas as pd
import sqlite3
import os
import re
import json
import plotly.express as px
from groq import Groq

st.set_page_config(
    page_title="AI SQL Data Analyst",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

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
        background: linear-gradient(135deg, #a78bfa, #38bdf8); color: white;
        font-weight: 600; border-radius: 8px; border: none;
        padding: 0.5rem 2rem; width: 100%; font-size: 1rem;
    }
    .stButton>button:hover { opacity: 0.9; transform: translateY(-1px); }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1>🧠 AI SQL Data Analyst Agent</h1>
    <p style="color:#94a3b8; font-size:1.1rem;">Upload a CSV → Ask questions in plain English → Get SQL + Answers + Charts</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    groq_api_key = st.secrets.get("GROQ_API_KEY", "")
    if not groq_api_key:
        groq_api_key = st.text_input("🔑 Groq API Key", type="password", placeholder="gsk_...")
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


@st.cache_resource
def load_csv_to_sqlite(csv_bytes: bytes, table_name: str = "data"):
    import io
    df = pd.read_csv(io.BytesIO(csv_bytes))
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    df.to_sql(table_name, conn, if_exists="replace", index=False)
    return conn

def get_schema(conn, table_name="data"):
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info({table_name})")
    cols = cursor.fetchall()
    schema_lines = [f"  {c[1]} ({c[2]})" for c in cols]
    return f"Table: {table_name}\nColumns:\n" + "\n".join(schema_lines)

def run_sql(conn, query):
    try:
        df_result = pd.read_sql_query(query, conn)
        return df_result, None
    except Exception as e:
        return None, str(e)

def ask_groq(client, model, schema, question, sample_rows):
    system_prompt = f"""You are an expert SQL analyst. Given a SQLite database schema and sample data, you:
1. Generate a correct SQLite SQL query to answer the user's question.
2. Provide a short plain-English explanation of the result.
3. Suggest the best chart type (bar, line, pie, scatter, or none).

Schema:
{schema}

Sample rows (first 3):
{sample_rows}

Rules:
- Always use the table name "data"
- Only use columns that exist in the schema
- If a column name has spaces, wrap it in double quotes e.g. "Customer ID", "Purchase Amount (USD)"
- Always use aliases for aggregations e.g. COUNT(*) as count, AVG(col) as avg_col
- Add LIMIT 100 at the end ONLY if the query does not already contain a LIMIT clause
- Never use two LIMIT clauses in the same query
- Return your response as valid JSON only (no markdown), with keys: "sql", "explanation", "chart_type"
- chart_type must be one of: bar, line, pie, scatter, none
- Use "none" ONLY for single-value results (e.g. COUNT(*) or SUM with no grouping)
- If result has 1 text column + 1 number column → always use "bar"
- If result has 2 number columns → use "line"
- If result is grouped by category with counts or totals → use "bar" or "pie"
- Always prefer showing a chart over "none"
"""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Question: {question}"}
        ],
        temperature=0.1,
    )
    raw = response.choices[0].message.content.strip()
    raw = re.sub(r"```json|```", "", raw).strip()
    return json.loads(raw)


def clean_df_for_chart(df):
    """Clean dataframe columns and types for charting."""
    # Clean column names: remove special chars, quotes, spaces
    df = df.copy()
    df.columns = [
        re.sub(r"[\(\)\*\s\"\']+", "_", col).strip("_").lower()
        for col in df.columns
    ]
    df = df.reset_index(drop=True)

    # Convert 0/1 bool-like numeric columns to Yes/No (categorical)
    for col in df.select_dtypes(include="number").columns:
        if df[col].nunique() <= 2 and set(df[col].dropna().unique()).issubset({0, 1, 0.0, 1.0}):
            df[col] = df[col].map({0: "No", 1: "Yes", 0.0: "No", 1.0: "Yes"})

    return df


def smart_chart_type(df):
    """Pick best chart type based on actual data shape."""
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(exclude="number").columns.tolist()
    total_cols = len(df.columns)
    total_rows = len(df)

    # Single value → just show number
    if total_rows == 1 and len(num_cols) == 1 and len(cat_cols) == 0:
        return "single", num_cols, cat_cols

    # Too many columns (SELECT *) → no chart
    if total_cols > 5:
        return "none", num_cols, cat_cols

    if len(cat_cols) >= 1 and len(num_cols) >= 1:
        unique_cats = df[cat_cols[0]].nunique()
        if unique_cats <= 5:
            return "pie", num_cols, cat_cols
        else:
            return "bar", num_cols, cat_cols

    # 2 numeric columns → line chart
    if len(num_cols) >= 2:
        return "line", num_cols, cat_cols

    return "none", num_cols, cat_cols


def render_chart(df_result, ai_chart_type):
    if df_result is None or df_result.empty:
        return

    df = clean_df_for_chart(df_result)
    chart_type, num_cols, cat_cols = smart_chart_type(df)

    if chart_type == "single":
        st.info(f"📊 Result: **{df[num_cols[0]].iloc[0]:,}**")
        return

    if chart_type == "none":
        st.info("📊 No visualization available for this query.")
        return

    try:
        if chart_type == "pie" and len(cat_cols) >= 1 and len(num_cols) >= 1:
            fig = px.pie(df, names=cat_cols[0], values=num_cols[0])
        elif chart_type == "bar" and len(cat_cols) >= 1 and len(num_cols) >= 1:
            fig = px.bar(df, x=cat_cols[0], y=num_cols[0], color_discrete_sequence=["#a78bfa"])
        elif chart_type == "line" and len(num_cols) >= 2:
            fig = px.line(df, x=num_cols[0], y=num_cols[1], markers=True, color_discrete_sequence=["#38bdf8"])
        else:
            if len(cat_cols) >= 1 and len(num_cols) >= 1:
                fig = px.bar(df, x=cat_cols[0], y=num_cols[0], color_discrete_sequence=["#a78bfa"])
            elif len(num_cols) >= 2:
                fig = px.line(df, x=num_cols[0], y=num_cols[1], markers=True, color_discrete_sequence=["#38bdf8"])
            else:
                st.info("📊 No suitable columns for visualization.")
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


uploaded_file = st.file_uploader("📂 Upload your CSV file", type=["csv"])

if uploaded_file:
    csv_bytes = uploaded_file.read()
    df = pd.read_csv(__import__("io").BytesIO(csv_bytes))

    st.success(f"✅ Loaded **{len(df):,} rows × {len(df.columns)} columns**")

    col1, col2, col3 = st.columns(3)
    col1.metric("📊 Rows", f"{len(df):,}")
    col2.metric("🔢 Columns", len(df.columns))
    col3.metric("💾 Size", f"{len(csv_bytes) / 1024:.1f} KB")

    with st.expander("🔍 Preview Data (first 10 rows)"):
        st.dataframe(df.head(10), use_container_width=True)

    conn = load_csv_to_sqlite(csv_bytes)
    schema = get_schema(conn)
    sample_rows = df.head(3).to_string(index=False)

    with st.expander("🗂️ Database Schema"):
        st.code(schema, language="sql")

    st.markdown("---")
    st.markdown("### 💬 Ask a Question")

    if "q_select" not in st.session_state:
        st.session_state["q_select"] = ""

    example_cols = df.columns.tolist()
    examples = [
        "Show me the top 5 rows",
        "Count total number of records",
        (
            f"What is the average of {example_cols[1] if len(example_cols) > 1 else example_cols[0]}?"
            if df.select_dtypes(include="number").shape[1] > 0
            else "Show distinct values"
        ),
    ]

    st.markdown("**Quick examples:**")
    col_a, col_b, col_c = st.columns(3)
    if col_a.button(examples[0]):
        st.session_state["q_select"] = examples[0]
    if col_b.button(examples[1]):
        st.session_state["q_select"] = examples[1]
    if col_c.button(examples[2]):
        st.session_state["q_select"] = examples[2]

    user_question = st.text_input(
        "Ask anything about your data...",
        value=st.session_state["q_select"],
        placeholder="e.g. What are the top 5 products by total sales?",
        key="user_question_input",
    )

    if st.button("🚀 Analyze"):
        if not groq_api_key:
            st.error("⚠️ Please enter your Groq API Key in the sidebar.")
        elif not user_question.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("🤖 AI is thinking..."):
                try:
                    client = Groq(api_key=groq_api_key)
                    result = ask_groq(client, model_choice, schema, user_question, sample_rows)

                    sql_query = result.get("sql", "")
                    explanation = result.get("explanation", "")
                    chart_type = result.get("chart_type", "none")

                    col_left, col_right = st.columns([1, 1])

                    with col_left:
                        st.markdown("### 🔎 Generated SQL")
                        st.markdown(f'<div class="sql-box">{sql_query}</div>', unsafe_allow_html=True)
                        st.markdown("### 💡 Explanation")
                        st.markdown(f'<div class="answer-box">{explanation}</div>', unsafe_allow_html=True)

                    with col_right:
                        st.markdown("### 📋 Query Results")
                        df_result, error = run_sql(conn, sql_query)
                        if error:
                            st.error(f"SQL Error: {error}")
                        else:
                            st.dataframe(df_result, use_container_width=True)

                    st.markdown("### 📊 Visualization")
                    if df_result is not None and not df_result.empty:
                        render_chart(df_result, chart_type)
                    else:
                        st.info("📊 No data to visualize.")

                except json.JSONDecodeError:
                    st.error("⚠️ AI returned an unexpected format. Try rephrasing your question.")
                except Exception as e:
                    st.error(f"❌ Error: {e}")

    st.markdown("---")
    st.markdown("### 🛠️ Direct SQL Query")
    direct_sql = st.text_area("Write your own SQL query:", placeholder="SELECT * FROM data LIMIT 10")
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
