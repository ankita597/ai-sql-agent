import streamlit as st
import pandas as pd
import sqlite3
import os
import re
import json
import matplotlib.pyplot as plt
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
    
    html, body, [class*="css"] {
        font-family: 'Space Grotesk', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    
    .main-header h1 {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #a78bfa, #38bdf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .sql-box {
        background: #1e1e2e;
        color: #cdd6f4;
        padding: 1rem;
        border-radius: 10px;
        font-family: 'Courier New', monospace;
        border-left: 4px solid #a78bfa;
        font-size: 0.9rem;
        white-space: pre-wrap;
    }
    
    .answer-box {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        color: #e2e8f0;
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #38bdf8;
        margin-top: 1rem;
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #a78bfa, #38bdf8);
        color: white;
        font-weight: 600;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 2rem;
        width: 100%;
        font-size: 1rem;
    }
    
    .stButton>button:hover {
        opacity: 0.9;
        transform: translateY(-1px);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border-radius: 10px;
        padding: 1rem;
        border: 1px solid #334155;
        text-align: center;
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

# ─── Sidebar: API Key + Model ────────────────────────────────────────────────
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
def write_csv_to_sqlite(df, path):
    engine = create_engine(...)
    df.to_sql(...)
    engine.dispose()       

def get_engine(path):
    return create_engine(...)  
    cols = cursor.fetchall()
    schema_lines = [f"  {c[1]} ({c[2]})" for c in cols]
    return f"Table: {table_name}\nColumns:\n" + "\n".join(schema_lines)

def run_sql(conn, query):
    try:
        df_result = pd.read_sql_query(query, conn)
        return df_result, None
    except Exception as e:
        return None, str(e)

# ─── Helper: Ask Groq ────────────────────────────────────────────────────────
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
- Always add LIMIT 100 at the end of the query unless the user explicitly asks for all rows
- Return your response as valid JSON only (no markdown), with keys: "sql", "explanation", "chart_type"
- chart_type must be one of: bar, line, pie, scatter, none
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
    # Strip markdown code fences if present
    raw = re.sub(r"```json|```", "", raw).strip()
    return json.loads(raw)

# ─── Helper: Auto Chart ─────────────────────────────────────────────────────
def render_chart(df_result, chart_type):
    if df_result is None or df_result.empty or chart_type == "none":
        return
    
    cols = df_result.columns.tolist()
    num_cols = df_result.select_dtypes(include="number").columns.tolist()
    cat_cols = df_result.select_dtypes(exclude="number").columns.tolist()

    try:
        if chart_type == "bar" and len(cat_cols) >= 1 and len(num_cols) >= 1:
            fig = px.bar(df_result, x=cat_cols[0], y=num_cols[0], color_discrete_sequence=["#a78bfa"])
        elif chart_type == "line" and len(num_cols) >= 1:
            x_col = cat_cols[0] if cat_cols else num_cols[0]
            fig = px.line(df_result, x=x_col, y=num_cols[0], markers=True, color_discrete_sequence=["#38bdf8"])
        elif chart_type == "pie" and len(cat_cols) >= 1 and len(num_cols) >= 1:
            fig = px.pie(df_result, names=cat_cols[0], values=num_cols[0])
        elif chart_type == "scatter" and len(num_cols) >= 2:
            fig = px.scatter(df_result, x=num_cols[0], y=num_cols[1], color_discrete_sequence=["#f472b6"])
        else:
            # Fallback bar
            if len(num_cols) >= 2:
                fig = px.scatter(df_result, x=num_cols[0], y=num_cols[1], color_discrete_sequence=["#f472b6"])
            elif len(num_cols) == 1 and len(cat_cols) >= 1:
                fig = px.bar(df_result, x=cat_cols[0], y=num_cols[0], color_discrete_sequence=["#a78bfa"])
            elif len(num_cols) == 1:
                st.info(f"📊 Result: **{df_result[num_cols[0]].iloc[0]}**")
                return
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
    
    st.success(f"✅ Loaded **{len(df):,} rows × {len(df.columns)} columns**")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("📊 Rows", f"{len(df):,}")
    col2.metric("🔢 Columns", len(df.columns))
    col3.metric("💾 Size", f"{uploaded_file.size / 1024:.1f} KB")
    
    with st.expander("🔍 Preview Data (first 10 rows)"):
        st.dataframe(df.head(10), use_container_width=True)
    
    conn = load_csv_to_sqlite(df)
    schema = get_schema(conn)
    sample_rows = df.head(3).to_string(index=False)

    with st.expander("🗂️ Database Schema"):
        st.code(schema, language="sql")

    st.markdown("---")
    st.markdown("### 💬 Ask a Question")
    
    # Suggested questions
    st.markdown("**Quick examples:**")
    example_cols = df.columns.tolist()
    examples = [
        f"Show me the top 5 rows",
        f"Count total number of records",
        f"What is the average of {example_cols[1] if len(example_cols) > 1 else example_cols[0]}?" if df.select_dtypes(include='number').shape[1] > 0 else "Show distinct values",
    ]
    col_a, col_b, col_c = st.columns(3)
    q_select = ""
    if col_a.button(examples[0]): q_select = examples[0]
    if col_b.button(examples[1]): q_select = examples[1]
    if col_c.button(examples[2]): q_select = examples[2]

    user_question = st.text_input(
        "Ask anything about your data...",
        value=q_select,
        placeholder="e.g. What are the top 5 products by total sales?"
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

                    if df_result is not None and not df_result.empty and chart_type != "none":
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
