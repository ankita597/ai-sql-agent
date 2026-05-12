import streamlit as st
import pandas as pd
import plotly.express as px
from groq import Groq
from sqlalchemy import create_engine, text as sa_text
import tempfile
import os
import re
import json

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
    html, body, [class*="css"] { font-family: 'Syne', sans-serif; }
    .main-header {
        background: #0a0a0f; border: 1px solid #1e1e2e;
        padding: 2.5rem 2rem; border-radius: 4px; margin-bottom: 2rem;
        text-align: center; position: relative; overflow: hidden;
    }
    .main-header::before {
        content: ''; position: absolute; top:0; left:0; right:0; bottom:0;
        background: repeating-linear-gradient(90deg, transparent, transparent 60px,
            rgba(99,102,241,0.03) 60px, rgba(99,102,241,0.03) 61px);
    }
    .main-header h1 { font-size:2.8rem; font-weight:800; letter-spacing:-0.03em; color:#f0f0f5; margin:0; }
    .main-header h1 span { color:#6366f1; }
    .main-header p { color:#4a4a6a; font-size:0.95rem; margin-top:0.5rem; font-family:'IBM Plex Mono',monospace; }
    .sql-box {
        background:#0d0d14; color:#a5b4fc; padding:1.2rem 1.5rem; border-radius:4px;
        font-family:'IBM Plex Mono',monospace; border:1px solid #1e1e3a;
        border-left:3px solid #6366f1; font-size:0.85rem; white-space:pre-wrap; word-break:break-word;
    }
    .answer-box {
        background:#0d0d14; color:#c7d2fe; padding:1.5rem; border-radius:4px;
        border:1px solid #1e1e3a; border-left:3px solid #22d3ee;
        margin-top:1rem; font-size:0.95rem; line-height:1.6;
    }
    .stButton > button {
        background:#6366f1; color:white; font-weight:600; font-family:'Syne',sans-serif;
        border-radius:4px; border:none; padding:0.55rem 1.8rem;
        width:100%; font-size:0.95rem; letter-spacing:0.02em;
    }
    .stButton > button:hover { background:#4f52d9; }
    .section-label {
        font-size:0.75rem; font-family:'IBM Plex Mono',monospace; color:#4a4a6a;
        letter-spacing:0.1em; text-transform:uppercase; margin-bottom:0.4rem;
    }
    .section-title { font-size:1.1rem; font-weight:700; color:#e0e0f0; margin-bottom:1rem; }
</style>
""", unsafe_allow_html=True)

# ─── Header ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🧠 AI SQL <span>Data Analyst</span></h1>
    <p>csv → sqlite → groq llm agent → sql → insights</p>
</div>
""", unsafe_allow_html=True)

# ─── Sidebar ──────────────────────────────────────────────────────────────────
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
3. The **Groq LLM Agent** auto-generates + runs SQL
4. Get **Answer + SQL + Chart**
    """)
    st.markdown("---")
    st.markdown("""
    <div style='font-family:IBM Plex Mono,monospace;font-size:0.7rem;color:#4a4a6a;'>
    STACK: Groq · SQLite · SQLAlchemy<br>Pandas · Plotly · Streamlit
    </div>
    """, unsafe_allow_html=True)


# ─── DB Helpers ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_csv_to_sqlite(_df: pd.DataFrame, tmp_path: str, table_name: str = "data"):
    engine = create_engine(f"sqlite:///{tmp_path}")
    _df.to_sql(table_name, engine, if_exists="replace", index=False)
    return engine

def get_schema_display(engine, table_name="data"):
    with engine.connect() as conn:
        cols = conn.execute(sa_text(f"PRAGMA table_info({table_name})")).fetchall()
    lines = [f"  {c[1]} ({c[2]})" for c in cols]
    return f"Table: {table_name}\nColumns:\n" + "\n".join(lines)

def get_column_names(engine, table_name="data"):
    with engine.connect() as conn:
        rows = conn.execute(sa_text(f"PRAGMA table_info({table_name})")).fetchall()
    return [r[1].lower() for r in rows]

def run_sql(engine, query: str):
    try:
        return pd.read_sql_query(query, engine), None
    except Exception as e:
        return None, str(e)


# ─── Groq SQL Agent ───────────────────────────────────────────────────────────
def ask_groq_agent(api_key: str, model: str, schema: str, sample: str,
                   question: str, error_ctx: dict = None):
    """
    Multi-turn Groq agent that:
      1. Generates a SQL query
      2. Self-corrects on failure (passed via error_ctx)
    Returns dict with keys: sql, explanation, chart_type
    """
    system = f"""You are an expert SQLite SQL analyst acting as an AI agent.

The SQLite table is named "data" with ONLY these columns:
{schema}

Sample rows (first 3):
{sample}

RULES:
1. Use ONLY column names listed above — copy them exactly.
2. Always reference the table as "data".
3. Wrap SQL-keyword column names in double quotes e.g. "order", "group".
4. Default LIMIT is 100 unless user asks for fewer.
5. For "top / most / highest" questions return ALL groups ordered by value.
6. Always include a numeric column when grouping so charts render.
7. Respond ONLY with valid JSON — no markdown, no backticks, no preamble.
   Keys: "sql" (string), "explanation" (string), "chart_type" (one of: bar|line|pie|scatter|none).
   - bar   → comparisons / counts by category
   - line  → time-based trends
   - pie   → share / percentage questions
   - scatter → correlation between two numbers
   - none  → raw row lookups (SELECT *)"""

    messages = [{"role": "system", "content": system}]

    if error_ctx:
        messages.append({"role": "user",    "content": f"Question: {question}"})
        messages.append({"role": "assistant","content": error_ctx["prev"]})
        messages.append({"role": "user",    "content":
            f"Your previous SQL caused this error: {error_ctx['error']}. "
            "Please fix it using only the columns in the schema."})
    else:
        messages.append({"role": "user", "content": f"Question: {question}"})

    client = Groq(api_key=api_key)
    resp = client.chat.completions.create(model=model, messages=messages, temperature=0.1)
    raw = resp.choices[0].message.content.strip()
    raw_clean = re.sub(r"```json|```", "", raw).strip()
    parsed = json.loads(raw_clean)
    return parsed, raw


# ─── No-Viz Card ─────────────────────────────────────────────────────────────
NO_VIZ_HTML = """
<div style="background:#0d0d14;border:1px dashed #2a2a3a;border-radius:4px;
    padding:2rem 1.5rem;text-align:center;margin-top:0.5rem;">
    <div style="font-size:2rem;margin-bottom:0.5rem;">📉</div>
    <div style="color:#4a4a6a;font-family:'IBM Plex Mono',monospace;
        font-size:0.88rem;font-weight:600;letter-spacing:0.05em;">
        NO VISUALIZATION AVAILABLE
    </div>
    <div style="color:#2a2a4a;font-size:0.78rem;margin-top:0.4rem;">{reason}</div>
</div>
"""

# ─── Chart Renderer ───────────────────────────────────────────────────────────
def render_chart(df_result: pd.DataFrame, chart_type: str = "auto"):
    if df_result is None or df_result.empty:
        st.markdown(NO_VIZ_HTML.format(reason="Query returned no data to visualize."), unsafe_allow_html=True)
        return

    num_cols = df_result.select_dtypes(include="number").columns.tolist()
    cat_cols = df_result.select_dtypes(exclude="number").columns.tolist()
    fig = None

    # Auto-detect if not specified
    if chart_type == "auto" or chart_type == "none":
        if len(cat_cols) >= 1 and len(num_cols) >= 1:
            chart_type = "bar"
        elif len(num_cols) >= 2:
            chart_type = "scatter"

    if chart_type == "bar" and len(cat_cols) >= 1 and len(num_cols) >= 1:
        fig = px.bar(df_result, x=cat_cols[0], y=num_cols[0], color=cat_cols[0],
                     color_discrete_sequence=px.colors.qualitative.Pastel,
                     title=f"{num_cols[0]} by {cat_cols[0]}")
    elif chart_type == "line" and len(num_cols) >= 1:
        x = cat_cols[0] if cat_cols else num_cols[0]
        fig = px.line(df_result, x=x, y=num_cols[0], markers=True,
                      color_discrete_sequence=["#6366f1"],
                      title=f"{num_cols[0]} over {x}")
    elif chart_type == "pie" and len(cat_cols) >= 1 and len(num_cols) >= 1:
        fig = px.pie(df_result, names=cat_cols[0], values=num_cols[0],
                     title=f"{num_cols[0]} share by {cat_cols[0]}")
    elif chart_type == "scatter" and len(num_cols) >= 2:
        fig = px.scatter(df_result, x=num_cols[0], y=num_cols[1],
                         color_discrete_sequence=["#22d3ee"],
                         title=f"{num_cols[0]} vs {num_cols[1]}")
    elif len(num_cols) == 1 and len(cat_cols) == 0:
        val = df_result[num_cols[0]].iloc[0]
        st.markdown(f"""
        <div style="background:#0d0d14;border:1px solid #1e1e3a;border-left:3px solid #6366f1;
            border-radius:4px;padding:1.5rem 2rem;text-align:center;margin-top:0.5rem;">
            <div style="color:#4a4a6a;font-family:'IBM Plex Mono',monospace;font-size:0.75rem;letter-spacing:0.1em;">RESULT</div>
            <div style="color:#a5b4fc;font-family:'IBM Plex Mono',monospace;font-size:2rem;font-weight:600;margin-top:0.3rem;">{val:,}</div>
            <div style="color:#4a4a6a;font-size:0.8rem;margin-top:0.3rem;">{num_cols[0]}</div>
        </div>""", unsafe_allow_html=True)
        return
    else:
        st.markdown(NO_VIZ_HTML.format(reason="Result has no numeric columns suitable for charting."), unsafe_allow_html=True)
        return

    if fig:
        fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                          font_color="#c7d2fe", font_family="IBM Plex Mono",
                          margin=dict(t=40, b=30))
        st.plotly_chart(fig, use_container_width=True)


# ─── Main App ─────────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader("📂 Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Track file identity to reset DB when a new CSV is uploaded
    file_identity = f"{uploaded_file.name}_{uploaded_file.size}"
    if st.session_state.get("current_file_identity") != file_identity:
        old_path = st.session_state.get("tmp_db_path")
        if old_path and os.path.exists(old_path):
            try: os.unlink(old_path)
            except: pass
        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        st.session_state["tmp_db_path"] = tmp.name
        tmp.close()
        st.session_state["current_file_identity"] = file_identity
        load_csv_to_sqlite.clear()

    tmp_db_path = st.session_state["tmp_db_path"]
    engine = load_csv_to_sqlite(df, tmp_db_path)

    # ── Overview ──────────────────────────────────────────────────────────────
    st.success(f"✅ Loaded **{len(df):,} rows × {len(df.columns)} columns**")
    c1, c2, c3 = st.columns(3)
    c1.metric("📊 Rows", f"{len(df):,}")
    c2.metric("🔢 Columns", len(df.columns))
    c3.metric("💾 Size", f"{uploaded_file.size / 1024:.1f} KB")

    with st.expander("🔍 Preview Data (first 10 rows)"):
        st.dataframe(df.head(10), use_container_width=True)

    schema_str = get_schema_display(engine)
    actual_columns = get_column_names(engine)
    sample_rows = df.head(3).to_string(index=False)

    with st.expander("🗂️ Database Schema"):
        st.code(schema_str, language="sql")

    st.markdown("---")

    # ── Quick Examples ────────────────────────────────────────────────────────
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
    ca, cb, cc = st.columns(3)
    if ca.button(examples[0]): q_select = examples[0]
    if cb.button(examples[1]): q_select = examples[1]
    if cc.button(examples[2]): q_select = examples[2]

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
            with st.spinner("🤖 Agent is thinking..."):
                try:
                    # ── 1. Generate SQL via Groq ──────────────────────────────
                    result, raw = ask_groq_agent(
                        groq_api_key, model_choice, schema_str, sample_rows, user_question
                    )
                    sql_query   = result.get("sql", "").strip()
                    explanation = result.get("explanation", "")
                    chart_type  = result.get("chart_type", "auto")

                    # ── 2. Run SQL; auto-retry on error ───────────────────────
                    df_result, error = run_sql(engine, sql_query)
                    if error:
                        st.warning("⚠️ SQL failed — auto-retrying with error context...")
                        result, raw = ask_groq_agent(
                            groq_api_key, model_choice, schema_str, sample_rows,
                            user_question, error_ctx={"error": error, "prev": raw}
                        )
                        sql_query   = result.get("sql", "").strip()
                        explanation = result.get("explanation", "")
                        chart_type  = result.get("chart_type", "auto")
                        df_result, error = run_sql(engine, sql_query)

                    # ── 3. Display SQL + Answer + Table ───────────────────────
                    col_left, col_right = st.columns([1, 1])

                    with col_left:
                        st.markdown('<div class="section-label">Generated SQL</div>', unsafe_allow_html=True)
                        st.markdown(f'<div class="sql-box">{sql_query}</div>', unsafe_allow_html=True)
                        st.markdown('<div class="section-label" style="margin-top:1.2rem;">Agent Explanation</div>', unsafe_allow_html=True)
                        st.markdown(f'<div class="answer-box">{explanation}</div>', unsafe_allow_html=True)

                    with col_right:
                        st.markdown('<div class="section-label">Query Results</div>', unsafe_allow_html=True)
                        if error:
                            st.error(f"SQL Error after retry: {error}")
                        elif df_result is not None:
                            st.dataframe(df_result, use_container_width=True)

                    # ── 4. Visualization ──────────────────────────────────────
                    st.markdown("---")
                    st.markdown('<div class="section-label">Visualization</div>', unsafe_allow_html=True)
                    st.markdown('<div class="section-title">📊 Auto Chart</div>', unsafe_allow_html=True)
                    if error:
                        st.markdown(NO_VIZ_HTML.format(reason="SQL error — no data to visualize."), unsafe_allow_html=True)
                    else:
                        render_chart(df_result, chart_type)

                except json.JSONDecodeError:
                    st.error("⚠️ AI returned an unexpected format. Try rephrasing your question.")
                except Exception as e:
                    st.error(f"❌ Error: {e}")

    # ── Direct SQL Panel ──────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-label">Manual Mode</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🛠️ Direct SQL Query</div>', unsafe_allow_html=True)
    direct_sql = st.text_area("Write your own SQL:", placeholder="SELECT * FROM data LIMIT 10", height=100)
    if st.button("▶️ Run SQL"):
        if direct_sql.strip():
            df_r, err = run_sql(engine, direct_sql)
            if err:
                st.error(err)
            else:
                st.dataframe(df_r, use_container_width=True)
                render_chart(df_r)

else:
    st.info("👆 Please upload a CSV file to get started.")
    st.markdown("""
    ### 🎯 What this agent can do:
    - **Understand** your CSV structure automatically
    - **Convert** natural language to SQL via **Groq LLM Agent**
    - **Execute** queries on an in-memory SQLite database
    - **Visualize** results with auto-detected chart types
    - **Explain** results in plain English
    - **Self-correct** on SQL errors automatically
    """)
