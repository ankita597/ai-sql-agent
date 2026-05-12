import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px

from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_groq import ChatGroq
from sqlalchemy import create_engine
import tempfile
import os

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
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Syne:wght@400;600;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Syne', sans-serif;
    }

    .main-header {
        background: #0a0a0f;
        border: 1px solid #1e1e2e;
        padding: 2.5rem 2rem;
        border-radius: 4px;
        margin-bottom: 2rem;
        text-align: center;
        position: relative;
        overflow: hidden;
    }
    .main-header::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; bottom: 0;
        background: repeating-linear-gradient(
            90deg,
            transparent,
            transparent 60px,
            rgba(99,102,241,0.03) 60px,
            rgba(99,102,241,0.03) 61px
        );
    }
    .main-header h1 {
        font-size: 2.8rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        color: #f0f0f5;
        margin: 0;
    }
    .main-header h1 span {
        color: #6366f1;
    }
    .main-header p {
        color: #4a4a6a;
        font-size: 0.95rem;
        margin-top: 0.5rem;
        font-family: 'IBM Plex Mono', monospace;
    }

    .sql-box {
        background: #0d0d14;
        color: #a5b4fc;
        padding: 1.2rem 1.5rem;
        border-radius: 4px;
        font-family: 'IBM Plex Mono', monospace;
        border: 1px solid #1e1e3a;
        border-left: 3px solid #6366f1;
        font-size: 0.85rem;
        white-space: pre-wrap;
        word-break: break-word;
    }

    .answer-box {
        background: #0d0d14;
        color: #c7d2fe;
        padding: 1.5rem;
        border-radius: 4px;
        border: 1px solid #1e1e3a;
        border-left: 3px solid #22d3ee;
        margin-top: 1rem;
        font-size: 0.95rem;
        line-height: 1.6;
    }

    .warning-box {
        background: #1a1000;
        color: #fbbf24;
        padding: 1rem;
        border-radius: 4px;
        border-left: 3px solid #f59e0b;
    }

    .stButton > button {
        background: #6366f1;
        color: white;
        font-weight: 600;
        font-family: 'Syne', sans-serif;
        border-radius: 4px;
        border: none;
        padding: 0.55rem 1.8rem;
        width: 100%;
        font-size: 0.95rem;
        letter-spacing: 0.02em;
        transition: background 0.2s;
    }
    .stButton > button:hover {
        background: #4f52d9;
    }

    .section-label {
        font-size: 0.75rem;
        font-family: 'IBM Plex Mono', monospace;
        color: #4a4a6a;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        margin-bottom: 0.4rem;
    }
    .section-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #e0e0f0;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ─── Header ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🧠 AI SQL <span>Data Analyst</span></h1>
    <p>csv → sqlite → langchain sql agent → groq llm → insights</p>
</div>
""", unsafe_allow_html=True)

# ─── Sidebar ─────────────────────────────────────────────────────────────────
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
3. The **LangChain SQL Agent** auto-generates + runs SQL
4. Get **Answer + SQL + Chart**
    """)
    st.markdown("---")
    st.markdown("""
    <div style='font-family: IBM Plex Mono, monospace; font-size:0.7rem; color:#4a4a6a;'>
    STACK: LangChain · Groq · SQLite<br>SQLAlchemy · Pandas · Plotly
    </div>
    """, unsafe_allow_html=True)


# ─── Helper: Load CSV into a persistent temp SQLite file ────────────────────
@st.cache_resource
def load_csv_to_sqlite(_df: pd.DataFrame, tmp_path: str, table_name: str = "data"):
    engine = create_engine(f"sqlite:///{tmp_path}")
    _df.to_sql(table_name, engine, if_exists="replace", index=False)
    return engine

def get_schema_display(engine, table_name="data"):
    with engine.connect() as conn:
        result = conn.execute(
            __import__("sqlalchemy").text(f"PRAGMA table_info({table_name})")
        )
        cols = result.fetchall()
    lines = [f"  {c[1]} ({c[2]})" for c in cols]
    return f"Table: {table_name}\nColumns:\n" + "\n".join(lines)

def get_column_names(engine, table_name="data"):
    with engine.connect() as conn:
        result = conn.execute(
            __import__("sqlalchemy").text(f"PRAGMA table_info({table_name})")
        )
        return [row[1].lower() for row in result.fetchall()]

def run_sql_direct(engine, query: str):
    """Run SQL directly for chart rendering."""
    try:
        df_result = pd.read_sql_query(query, engine)
        return df_result, None
    except Exception as e:
        return None, str(e)

def extract_sql_from_answer(answer: str) -> str:
    """Try to extract a SQL query from the agent's final answer string."""
    import re
    # Match SQL blocks in markdown
    match = re.search(r"```sql\s*(.*?)\s*```", answer, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    # Match plain SELECT statements
    match = re.search(r"(SELECT\s.+?;)", answer, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    # Match SELECT without semicolon (greedy until newline sequence)
    match = re.search(r"(SELECT\b.+)", answer, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return ""


# ─── Chart Renderer ──────────────────────────────────────────────────────────
NO_VIZ_HTML = """
<div style="
    background:#0d0d14;
    border:1px dashed #2a2a3a;
    border-radius:4px;
    padding:2rem 1.5rem;
    text-align:center;
    margin-top:0.5rem;
">
    <div style="font-size:2rem;margin-bottom:0.5rem;">📉</div>
    <div style="color:#4a4a6a;font-family:'IBM Plex Mono',monospace;font-size:0.88rem;font-weight:600;letter-spacing:0.05em;">
        NO VISUALIZATION AVAILABLE
    </div>
    <div style="color:#2a2a4a;font-size:0.78rem;margin-top:0.4rem;">{reason}</div>
</div>
"""

def render_chart(df_result: pd.DataFrame):
    if df_result is None or df_result.empty:
        st.markdown(NO_VIZ_HTML.format(reason="Query returned no data to visualize."), unsafe_allow_html=True)
        return

    num_cols = df_result.select_dtypes(include="number").columns.tolist()
    cat_cols = df_result.select_dtypes(exclude="number").columns.tolist()

    fig = None

    if len(cat_cols) >= 1 and len(num_cols) >= 1:
        if df_result[cat_cols[0]].nunique() <= 8:
            fig = px.bar(
                df_result, x=cat_cols[0], y=num_cols[0],
                color=cat_cols[0],
                color_discrete_sequence=px.colors.qualitative.Pastel,
                title=f"{num_cols[0]} by {cat_cols[0]}"
            )
        else:
            fig = px.line(
                df_result, x=cat_cols[0], y=num_cols[0], markers=True,
                color_discrete_sequence=["#6366f1"],
                title=f"{num_cols[0]} over {cat_cols[0]}"
            )
    elif len(num_cols) >= 2:
        fig = px.scatter(
            df_result, x=num_cols[0], y=num_cols[1],
            color_discrete_sequence=["#22d3ee"],
            title=f"{num_cols[0]} vs {num_cols[1]}"
        )
    elif len(num_cols) == 1 and len(cat_cols) == 0:
        val = df_result[num_cols[0]].iloc[0]
        st.markdown(f"""
        <div style="background:#0d0d14;border:1px solid #1e1e3a;border-left:3px solid #6366f1;
            border-radius:4px;padding:1.5rem 2rem;text-align:center;margin-top:0.5rem;">
            <div style="color:#4a4a6a;font-family:'IBM Plex Mono',monospace;font-size:0.75rem;letter-spacing:0.1em;">RESULT</div>
            <div style="color:#a5b4fc;font-family:'IBM Plex Mono',monospace;font-size:2rem;font-weight:600;margin-top:0.3rem;">{val:,}</div>
            <div style="color:#4a4a6a;font-size:0.8rem;margin-top:0.3rem;">{num_cols[0]}</div>
        </div>
        """, unsafe_allow_html=True)
        return
    else:
        st.markdown(NO_VIZ_HTML.format(reason="Result contains no numeric columns suitable for charting."), unsafe_allow_html=True)
        return

    if fig:
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#c7d2fe",
            font_family="IBM Plex Mono",
            margin=dict(t=40, b=30),
        )
        st.plotly_chart(fig, use_container_width=True)


# ─── Main App ─────────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader("📂 Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Identify the current file by name + size so we detect when a new CSV is uploaded
    file_identity = f"{uploaded_file.name}_{uploaded_file.size}"

    if st.session_state.get("current_file_identity") != file_identity:
        # New file uploaded — clean up old temp DB and create a fresh one
        old_path = st.session_state.get("tmp_db_path")
        if old_path and os.path.exists(old_path):
            try:
                os.unlink(old_path)
            except Exception:
                pass

        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        st.session_state["tmp_db_path"] = tmp.name
        tmp.close()

        st.session_state["current_file_identity"] = file_identity
        load_csv_to_sqlite.clear()  # Clear cache only when file actually changes

    tmp_db_path = st.session_state["tmp_db_path"]
    engine = load_csv_to_sqlite(df, tmp_db_path)

    # ── Data Overview ────────────────────────────────────────────────────────
    st.success(f"✅ Loaded **{len(df):,} rows × {len(df.columns)} columns**")
    c1, c2, c3 = st.columns(3)
    c1.metric("📊 Rows", f"{len(df):,}")
    c2.metric("🔢 Columns", len(df.columns))
    c3.metric("💾 Size", f"{uploaded_file.size / 1024:.1f} KB")

    with st.expander("🔍 Preview Data (first 10 rows)"):
        st.dataframe(df.head(10), use_container_width=True)

    schema_str = get_schema_display(engine)
    actual_columns = get_column_names(engine)

    with st.expander("🗂️ Database Schema"):
        st.code(schema_str, language="sql")

    st.markdown("---")

    # ── Quick Examples ───────────────────────────────────────────────────────
    st.markdown('<div class="section-label">Natural Language Query</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">💬 Ask a Question</div>', unsafe_allow_html=True)

    num_col = df.select_dtypes(include="number").columns[0] if not df.select_dtypes(include="number").empty else df.columns[0]
    cat_col = df.select_dtypes(exclude="number").columns[0] if not df.select_dtypes(exclude="number").empty else df.columns[0]

    examples = [
        "Show me the top 5 rows",
        f"Count records by {cat_col}",
        f"What is the average {num_col}?",
    ]

    q_select = ""
    col_a, col_b, col_c = st.columns(3)
    if col_a.button(examples[0]): q_select = examples[0]
    if col_b.button(examples[1]): q_select = examples[1]
    if col_c.button(examples[2]): q_select = examples[2]

    user_question = st.text_input(
        "Ask anything about your data...",
        value=q_select,
        placeholder="e.g. Which category has the highest total sales?"
    )

    if st.button("🚀 Analyze with LangChain Agent"):
        if not groq_api_key:
            st.error("⚠️ Please provide your Groq API Key in the sidebar.")
        elif not user_question.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("🤖 LangChain SQL Agent is thinking..."):
                try:
                    # ── 1. Build LangChain SQLDatabase from the SQLite file ──
                    db = SQLDatabase.from_uri(f"sqlite:///{tmp_db_path}")

                    # ── 2. Initialise Groq LLM via LangChain ─────────────────
                    llm = ChatGroq(
                        api_key=groq_api_key,
                        model_name=model_choice,
                        temperature=0,
                    )

                    # ── 3. Create LangChain SQL Agent ─────────────────────────
                    agent_executor = create_sql_agent(
                        llm=llm,
                        db=db,
                        agent_type="openai-tools",   # works well with Groq
                        verbose=False,
                        handle_parsing_errors=True,
                    )

                    # ── 4. Run the agent ──────────────────────────────────────
                    agent_response = agent_executor.invoke(
                        {"input": user_question}
                    )
                    final_answer = agent_response.get("output", str(agent_response))

                    # ── 5. Extract SQL used (from intermediate steps if available) ──
                    sql_used = ""
                    intermediate = agent_response.get("intermediate_steps", [])
                    for step in intermediate:
                        # Each step is (AgentAction, observation)
                        action = step[0] if isinstance(step, (list, tuple)) else step
                        tool_input = getattr(action, "tool_input", "")
                        if isinstance(tool_input, str) and tool_input.upper().startswith("SELECT"):
                            sql_used = tool_input
                        elif isinstance(tool_input, dict):
                            q = tool_input.get("query", "")
                            if q.upper().startswith("SELECT"):
                                sql_used = q

                    # Fallback: try extracting SQL from the answer text
                    if not sql_used:
                        sql_used = extract_sql_from_answer(final_answer)

                    # ── 6. Display Results ────────────────────────────────────
                    col_left, col_right = st.columns([1, 1])

                    with col_left:
                        st.markdown('<div class="section-label">Generated SQL</div>', unsafe_allow_html=True)
                        if sql_used:
                            st.markdown(f'<div class="sql-box">{sql_used}</div>', unsafe_allow_html=True)
                        else:
                            st.info("SQL query not captured from agent steps.")

                        st.markdown('<div class="section-label" style="margin-top:1.2rem;">Agent Answer</div>', unsafe_allow_html=True)
                        st.markdown(f'<div class="answer-box">{final_answer}</div>', unsafe_allow_html=True)

                    with col_right:
                        st.markdown('<div class="section-label">Query Results</div>', unsafe_allow_html=True)
                        if sql_used:
                            df_result, err = run_sql_direct(engine, sql_used)
                            if err:
                                st.error(f"Could not re-run SQL for table display: {err}")
                            elif df_result is not None:
                                st.dataframe(df_result, use_container_width=True)
                        else:
                            st.info("No SQL to re-execute for table display.")

                    # ── 7. Auto Visualization ─────────────────────────────────
                    st.markdown("---")
                    st.markdown('<div class="section-label">Visualization</div>', unsafe_allow_html=True)
                    st.markdown('<div class="section-title">📊 Auto Chart</div>', unsafe_allow_html=True)
                    if sql_used:
                        df_viz, _ = run_sql_direct(engine, sql_used)
                        render_chart(df_viz)
                    else:
                        st.markdown(NO_VIZ_HTML.format(reason="No SQL query was captured to visualize."), unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"❌ Agent Error: {e}")
                    st.info("💡 Tip: Make sure `langchain-community`, `langchain-groq`, and `sqlalchemy` are installed.")

    # ── Direct SQL Panel ──────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-label">Manual Mode</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🛠️ Direct SQL Query</div>', unsafe_allow_html=True)
    direct_sql = st.text_area("Write your own SQL:", placeholder="SELECT * FROM data LIMIT 10", height=100)
    if st.button("▶️ Run SQL"):
        if direct_sql.strip():
            df_r, err = run_sql_direct(engine, direct_sql)
            if err:
                st.error(err)
            else:
                st.dataframe(df_r, use_container_width=True)
                render_chart(df_r)

else:
    st.info("👆 Please upload a CSV file to get started.")
    st.markdown("""
    ### 🎯 What this agent can do:
    - **Understand** your CSV data structure automatically
    - **Convert** natural language questions to SQL via **LangChain SQL Agent**
    - **Execute** queries on an in-memory SQLite database
    - **Visualize** results with auto-detected chart types
    - **Explain** results in plain English via **Groq (Llama 3)**
    """)
