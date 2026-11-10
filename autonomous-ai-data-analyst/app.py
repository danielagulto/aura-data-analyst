import os, io, json, re, hashlib, textwrap
from datetime import datetime
from urllib.parse import quote_plus

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import SQLAlchemyError

st.set_page_config(page_title="AURA | AI Data Analyst", page_icon="◈", layout="wide", initial_sidebar_state="expanded")

# ---------- Styling ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
html, body, [class*="css"] {font-family:'DM Sans',sans-serif;}
.stApp {background:radial-gradient(ellipse at 15% 0%, #17243a 0%, #0a1020 45%, #080d18 100%); color:#e8edf7;}
h1,h2,h3 {font-family:'Space Grotesk',sans-serif; letter-spacing:-.035em;}
[data-testid="stSidebar"] {background:#0b1322; border-right:1px solid #26344a;}
[data-testid="stMetric"] {background:linear-gradient(145deg,#142238,#101a2b); padding:18px 20px; border:1px solid #263b56; border-radius:16px;}
[data-testid="stMetricValue"] {color:#f3f7ff;}
div.stButton > button {border-radius:10px; border:1px solid #385273; background:#172941; color:#eaf3ff;}
div.stButton > button:hover {border-color:#68c8d8; color:#fff;}
div[data-testid="stFileUploader"] {background:#101c2d; border:1px dashed #385273; border-radius:14px; padding:12px;}
.block-container {padding-top:2rem; max-width:1500px;}
.small-muted {color:#94a8c3;font-size:.88rem;}
.hero {padding:22px 26px; border:1px solid #2b405b; border-radius:18px; background:linear-gradient(115deg,rgba(23,43,67,.95),rgba(12,23,39,.85)); margin-bottom:18px;}
.pill {display:inline-block;padding:5px 10px;border-radius:30px;background:#153a43;color:#8ce4df;font-size:12px;border:1px solid #27636a;}
hr {border-color:#26364b;}
</style>
""", unsafe_allow_html=True)

if "df" not in st.session_state: st.session_state.df = None
if "source_name" not in st.session_state: st.session_state.source_name = None
if "history" not in st.session_state: st.session_state.history = []
if "analysis" not in st.session_state: st.session_state.analysis = None

def safe_name(s):
    return re.sub(r"[^A-Za-z0-9_]", "_", str(s))[:60] or "dataset"

def load_upload(file):
    name = file.name.lower()
    if name.endswith(".csv"): return pd.read_csv(file)
    if name.endswith((".xlsx", ".xls")): return pd.read_excel(file)
    if name.endswith(".json"):
        raw = json.load(file)
        if isinstance(raw, dict):
            for key in ("data", "results", "records", "items"):
                if key in raw and isinstance(raw[key], list): raw = raw[key]; break
        return pd.json_normalize(raw)
    if name.endswith(".parquet"): return pd.read_parquet(file)
    raise ValueError("Supported uploads: CSV, XLSX, XLS, JSON, Parquet.")

def get_profile(df):
    rows, cols = df.shape
    missing = df.isna().sum()
    dupes = int(df.duplicated().sum())
    numeric = df.select_dtypes(include=np.number)
    return {
        "rows": rows, "columns": cols, "duplicates": dupes,
        "missing_cells": int(missing.sum()),
        "missing_pct": round(float(missing.sum()/(rows*cols)*100),2) if rows*cols else 0,
        "numeric_columns": list(numeric.columns),
        "categorical_columns": list(df.select_dtypes(include=["object","category","bool"]).columns),
        "date_columns": [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])],
        "memory_mb": round(df.memory_usage(deep=True).sum()/1024**2,2)
    }

def profile_table(df):
    out = pd.DataFrame({
        "Column": df.columns.astype(str),
        "Data type": [str(x) for x in df.dtypes],
        "Missing": [int(df[c].isna().sum()) for c in df.columns],
        "Missing %": [round(df[c].isna().mean()*100,2) for c in df.columns],
        "Unique": [int(df[c].nunique(dropna=True)) for c in df.columns]
    })
    return out

def run_sql_readonly(url, table_name=None, query=None, limit=100000):
    engine = create_engine(url, pool_pre_ping=True, connect_args={"connect_timeout": 10} if url.startswith("postgresql") else {})
    try:
        with engine.connect() as conn:
            if query:
                q = query.strip().rstrip(";")
                # Restrict to one read-only statement; reject comments and mutating keywords.
                if ";" in q or re.search(r"\b(insert|update|delete|drop|alter|create|truncate|grant|revoke|execute|call|copy)\b", q, re.I):
                    raise ValueError("Only a single read-only SELECT or WITH query is permitted.")
                if not re.match(r"^(select|with)\b", q, re.I):
                    raise ValueError("Only SELECT or WITH queries are permitted.")
                result = conn.execute(text(f"SELECT * FROM ({q}) AS analyst_query LIMIT {int(limit)}"))
                return pd.DataFrame(result.fetchall(), columns=result.keys()), engine
            if not table_name: raise ValueError("Choose a table or provide a SELECT query.")
            # quote table identifier using SQLAlchemy dialect preparer
            preparer = engine.dialect.identifier_preparer
            quoted = ".".join(preparer.quote(x) for x in table_name.split("."))
            result = conn.execute(text(f"SELECT * FROM {quoted} LIMIT {int(limit)}"))
            return pd.DataFrame(result.fetchall(), columns=result.keys()), engine
    except Exception:
        engine.dispose()
        raise

def ai_answer(task, df, api_key, model):
    if not api_key:
        return None, "Add an OpenAI API key in the sidebar to enable natural-language AI analysis. Built-in profiling and analytics work without an API key."
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        sample = df.head(8).replace({np.nan: None}).to_dict(orient="records")
        summary = {
            "shape": list(df.shape),
            "columns": [{"name": str(c), "dtype": str(df[c].dtype), "missing": int(df[c].isna().sum()), "unique": int(df[c].nunique())} for c in df.columns],
            "numeric_summary": df.select_dtypes(include=np.number).describe().round(3).to_dict(),
            "sample_rows": sample
        }
        system = """You are an expert data analyst. Analyze only the supplied dataset metadata and sample. Never invent values or claim you computed results you cannot verify. Clearly label hypotheses and limitations. Give a concise executive answer, analytical steps, evidence needed, and recommended next analyses. If the sample is insufficient, say so. Do not reveal secrets or request database credentials."""
        response = client.chat.completions.create(
            model=model or "gpt-4.1-mini",
            messages=[{"role":"system","content":system},{"role":"user","content":f"Task: {task}\nDataset context:\n{json.dumps(summary, default=str)[:18000]}"}],
            temperature=0.2
        )
        return response.choices[0].message.content, None
    except Exception as e:
        return None, f"AI request failed: {e}"

def make_pdf_report(df, task, answer):
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import inch
    buff = io.BytesIO()
    doc = SimpleDocTemplate(buff, pagesize=letter, rightMargin=42,leftMargin=42,topMargin=42,bottomMargin=42)
    styles = getSampleStyleSheet()
    story = [Paragraph("AURA — Data Analysis Report", styles["Title"]), Spacer(1,10),
             Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles["Normal"]), Spacer(1,10),
             Paragraph("Analysis request", styles["Heading2"]), Paragraph(task or "Dataset overview", styles["BodyText"]), Spacer(1,10),
             Paragraph("Dataset overview", styles["Heading2"])]
    p = get_profile(df)
    data = [["Metric","Value"],["Rows",f"{p['rows']:,}"],["Columns",str(p["columns"])],["Missing cells",f"{p['missing_cells']:,}"],["Duplicate rows",f"{p['duplicates']:,}"],["Memory",f"{p['memory_mb']} MB"]]
    t = Table(data, colWidths=[2.3*inch,3.2*inch])
    t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#18314d")),("TEXTCOLOR",(0,0),(-1,0),colors.white),("GRID",(0,0),(-1,-1),.4,colors.HexColor("#9aaabd")),("PADDING",(0,0),(-1,-1),7)]))
    story += [t, Spacer(1,12), Paragraph("AI findings", styles["Heading2"])]
    for para in (answer or "No AI findings were generated.").split("\n"):
        if para.strip(): story.append(Paragraph(para.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;"), styles["BodyText"])); story.append(Spacer(1,5))
    story += [Spacer(1,10), Paragraph("Data preview", styles["Heading2"])]
    preview = [list(map(str, df.columns[:6]))] + df.head(8).iloc[:,:6].fillna("").astype(str).values.tolist()
    pt = Table(preview, repeatRows=1)
    pt.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#18314d")),("TEXTCOLOR",(0,0),(-1,0),colors.white),("GRID",(0,0),(-1,-1),.3,colors.grey),("FONTSIZE",(0,0),(-1,-1),7),("PADDING",(0,0),(-1,-1),4)]))
    story.append(pt)
    doc.build(story)
    buff.seek(0)
    return buff.getvalue()

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("## ◈ AURA")
    st.caption("AUTONOMOUS DATA ANALYST")
    st.markdown('<span class="pill">ANALYTICS WORKSPACE</span>', unsafe_allow_html=True)
    st.divider()
    page = st.radio("WORKSPACE", ["Overview", "Ask the Analyst", "Explore Data", "Visualizations", "Data Quality", "Reports"], label_visibility="visible")
    st.divider()
    st.markdown("### AI configuration")
    api_key = st.text_input("OpenAI API key", type="password", help="Used only for this app session. Do not share your key.")
    model = st.text_input("Model", value=os.getenv("OPENAI_MODEL","gpt-4.1-mini"))
    st.caption("AI features require your own API key. Local profiling and charts do not.")
    st.divider()
    st.markdown("### Connect data")
    upload = st.file_uploader("Upload a dataset", type=["csv","xlsx","xls","json","parquet"])
    if upload is not None:
        try:
            new_df = load_upload(upload)
            st.session_state.df = new_df
            st.session_state.source_name = upload.name
            st.session_state.analysis = None
            st.success(f"Loaded {len(new_df):,} rows.")
        except Exception as e: st.error(f"Could not load file: {e}")
    with st.expander("Connect SQL database"):
        dialect = st.selectbox("Database", ["PostgreSQL","MySQL","Microsoft SQL Server","SQLite"])
        if dialect == "SQLite":
            db_path = st.text_input("SQLite file path", value="")
            sql_url = f"sqlite:///{db_path}" if db_path else ""
        else:
            host = st.text_input("Host")
            port_default = {"PostgreSQL":"5432","MySQL":"3306","Microsoft SQL Server":"1433"}[dialect]
            port = st.text_input("Port", value=port_default)
            dbname = st.text_input("Database name")
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            if dialect == "PostgreSQL": sql_url = f"postgresql+psycopg2://{quote_plus(username)}:{quote_plus(password)}@{host}:{port}/{dbname}" if host and dbname else ""
            elif dialect == "MySQL": sql_url = f"mysql+pymysql://{quote_plus(username)}:{quote_plus(password)}@{host}:{port}/{dbname}" if host and dbname else ""
            else: sql_url = f"mssql+pyodbc://{quote_plus(username)}:{quote_plus(password)}@{host}:{port}/{dbname}?driver=ODBC+Driver+18+for+SQL+Server&TrustServerCertificate=yes" if host and dbname else ""
        if st.button("Inspect database", use_container_width=True):
            if not sql_url: st.warning("Enter the connection details first.")
            else:
                try:
                    eng = create_engine(sql_url, pool_pre_ping=True)
                    names = inspect(eng).get_table_names()
                    st.session_state.db_tables = names
                    st.session_state.db_url = sql_url
                    st.success(f"Connected. Found {len(names)} tables.")
                    eng.dispose()
                except Exception as e: st.error(f"Connection failed: {e}")
        if st.session_state.get("db_tables"):
            selected_table = st.selectbox("Choose table", st.session_state.db_tables)
            row_limit = st.number_input("Maximum rows to load", 100, 500000, 100000, step=1000)
            if st.button("Load selected table", use_container_width=True):
                try:
                    data, eng = run_sql_readonly(st.session_state.db_url, table_name=selected_table, limit=row_limit)
                    st.session_state.df = data
                    st.session_state.source_name = f"SQL: {selected_table}"
                    st.session_state.analysis = None
                    eng.dispose()
                    st.success(f"Loaded {len(data):,} rows.")
                except Exception as e: st.error(str(e))

# ---------- Header ----------
st.markdown('<div class="hero"><div class="small-muted">AUTONOMOUS ANALYTICS / WORKSPACE</div><h1 style="margin:5px 0 3px;">Turn data into decisions.</h1><div class="small-muted">Connect a dataset. Describe the business question. AURA helps you investigate, visualize, and report.</div></div>', unsafe_allow_html=True)

df = st.session_state.df
if df is None:
    st.markdown("### Your analysis starts here")
    st.write("Upload a CSV, Excel, JSON, or Parquet file from the sidebar, or connect a supported SQL database.")
    c1,c2,c3 = st.columns(3)
    with c1: st.metric("Data sources","4 file formats + SQL")
    with c2: st.metric("Analysis tools","Profiling · Charts · QA")
    with c3: st.metric("AI assistant","Optional API key")
    st.info("This deployable first release includes working file ingestion, read-only SQL loading, data profiling, visual exploration, quality checks, AI-assisted analysis, and PDF/CSV/XLSX exports.")
    st.stop()

profile = get_profile(df)
if page == "Overview":
    st.subheader("Dataset overview")
    st.caption(f"Source: {st.session_state.source_name or 'Dataset'}  ·  Loaded {datetime.now().strftime('%H:%M')}")
    cols = st.columns(5)
    cols[0].metric("Rows", f"{profile['rows']:,}")
    cols[1].metric("Columns", f"{profile['columns']:,}")
    cols[2].metric("Missing cells", f"{profile['missing_cells']:,}")
    cols[3].metric("Duplicate rows", f"{profile['duplicates']:,}")
    cols[4].metric("Memory", f"{profile['memory_mb']} MB")
    st.divider()
    left,right = st.columns([1.2,1])
    with left:
        st.markdown("#### Data preview")
        st.dataframe(df.head(12), use_container_width=True, height=360)
    with right:
        st.markdown("#### Dataset composition")
        types = df.dtypes.astype(str).value_counts().reset_index()
        types.columns = ["Data type","Columns"]
        fig = px.pie(types, names="Data type", values="Columns", hole=.62, template="plotly_dark")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(t=10,b=10,l=10,r=10))
        st.plotly_chart(fig, use_container_width=True)
    st.markdown("#### Column profile")
    st.dataframe(profile_table(df), use_container_width=True, hide_index=True)

elif page == "Ask the Analyst":
    st.subheader("Ask the AI data analyst")
    st.caption("Describe the outcome you need. AI responses use dataset metadata and a small sample; verify findings against full-data calculations.")
    task = st.text_area("What should I investigate?", placeholder="Example: Analyze monthly revenue, identify the best-performing product categories, investigate the reasons for declining profit, and recommend next steps.", height=120)
    if st.button("Run analysis", type="primary", disabled=not task.strip()):
        with st.spinner("Inspecting dataset context and preparing analysis..."):
            answer, err = ai_answer(task, df, api_key, model)
            if answer:
                st.session_state.analysis = {"task":task,"answer":answer,"time":datetime.now().isoformat()}
                st.session_state.history.append({"task":task,"time":datetime.now().strftime("%Y-%m-%d %H:%M")})
            elif err: st.warning(err)
    if st.session_state.analysis:
        st.markdown("#### Analysis result")
        st.markdown(st.session_state.analysis["answer"])
        with st.expander("Dataset context supplied to AI"):
            st.json({"shape":df.shape,"columns":list(map(str,df.columns)),"sample_rows":min(8,len(df))})
    st.markdown("#### Suggested tasks")
    prompts = ["Summarize this dataset and identify important patterns.","Check data quality and recommend cleaning steps.","Which numeric variables have the strongest correlations?","Suggest the most useful KPIs and visualizations for this dataset."]
    for p in prompts:
        if st.button(p, key=hashlib.md5(p.encode()).hexdigest()):
            st.session_state.quick_task = p
            st.rerun()

elif page == "Explore Data":
    st.subheader("Explore data")
    chosen = st.multiselect("Select columns", list(df.columns), default=list(df.columns[:min(8,len(df.columns))]))
    st.dataframe(df[chosen].head(1000), use_container_width=True, height=500)
    st.caption(f"Showing up to 1,000 of {len(df):,} loaded rows.")
    st.download_button("Download current dataset as CSV", df.to_csv(index=False).encode("utf-8"), "aura_dataset.csv", "text/csv")

elif page == "Visualizations":
    st.subheader("Visualization studio")
    numeric = list(df.select_dtypes(include=np.number).columns)
    categorical = list(df.select_dtypes(include=["object","category","bool"]).columns)
    if not numeric and not categorical:
        st.info("No chartable columns were detected.")
    else:
        chart_type = st.selectbox("Chart type", ["Bar","Line","Scatter","Histogram","Box","Correlation heatmap","Pie / donut"])
        if chart_type == "Correlation heatmap":
            if len(numeric) >= 2:
                corr = df[numeric].corr(numeric_only=True)
                fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1,zmax=1, template="plotly_dark")
                st.plotly_chart(fig,use_container_width=True)
            else: st.info("Select a dataset with at least two numeric columns.")
        elif chart_type == "Histogram":
            x = st.selectbox("Numeric column", numeric)
            fig = px.histogram(df, x=x, nbins=35, template="plotly_dark")
            st.plotly_chart(fig,use_container_width=True)
        elif chart_type == "Pie / donut":
            if categorical:
                name = st.selectbox("Category", categorical)
                value = st.selectbox("Values (optional)", ["Count"]+numeric)
                data = df.groupby(name, dropna=False).size().reset_index(name="Count") if value=="Count" else df.groupby(name, dropna=False)[value].sum().reset_index()
                fig = px.pie(data, names=name, values="Count" if value=="Count" else value, hole=.45, template="plotly_dark")
                st.plotly_chart(fig,use_container_width=True)
            else: st.info("A categorical column is required.")
        else:
            x_options = list(df.columns)
            x = st.selectbox("X axis", x_options)
            if chart_type in ("Bar","Line"):
                y = st.selectbox("Y axis", numeric)
                color = st.selectbox("Color / series (optional)", ["None"]+categorical)
                agg = st.selectbox("Aggregation", ["sum","mean","median","count"])
                if st.button("Build chart"):
                    if chart_type=="Bar":
                        if agg=="count": data=df.groupby(x,dropna=False).size().reset_index(name="value")
                        else: data=df.groupby(x,dropna=False)[y].agg(agg).reset_index(name="value")
                        fig=px.bar(data,x=x,y="value",template="plotly_dark")
                    else:
                        data=df.groupby(x,dropna=False)[y].agg(agg).reset_index(name="value")
                        fig=px.line(data,x=x,y="value",markers=True,template="plotly_dark")
                    st.plotly_chart(fig,use_container_width=True)
            elif chart_type=="Scatter":
                if len(numeric)>=2:
                    y=st.selectbox("Y axis",numeric)
                    fig=px.scatter(df,x=x,y=y,template="plotly_dark")
                    st.plotly_chart(fig,use_container_width=True)
                else: st.info("Scatter plots need at least two numeric columns.")
            elif chart_type=="Box":
                y=st.selectbox("Numeric column",numeric)
                fig=px.box(df,x=x,y=y,template="plotly_dark")
                st.plotly_chart(fig,use_container_width=True)

elif page == "Data Quality":
    st.subheader("Data quality assessment")
    missing = df.isna().sum().sort_values(ascending=False)
    q1,q2,q3 = st.columns(3)
    q1.metric("Missing cells", f"{profile['missing_cells']:,}")
    q2.metric("Duplicate rows", f"{profile['duplicates']:,}")
    q3.metric("Columns with missing data", int((missing>0).sum()))
    st.markdown("#### Missing values by column")
    miss = missing[missing>0].rename("Missing").reset_index().rename(columns={"index":"Column"})
    if len(miss):
        fig=px.bar(miss,x="Column",y="Missing",template="plotly_dark")
        st.plotly_chart(fig,use_container_width=True)
    else: st.success("No missing values detected.")
    st.markdown("#### Potential outliers (IQR rule)")
    nums=df.select_dtypes(include=np.number)
    out=[]
    for c in nums:
        s=nums[c].dropna()
        if len(s):
            ql,qh=s.quantile(.25),s.quantile(.75); iqr=qh-ql
            n=int(((s<ql-1.5*iqr)|(s>qh+1.5*iqr)).sum())
            out.append({"Column":c,"Potential outliers":n,"Lower bound":ql-1.5*iqr,"Upper bound":qh+1.5*iqr})
    st.dataframe(pd.DataFrame(out),use_container_width=True,hide_index=True)
    st.warning("Quality checks flag potential issues; they do not automatically modify your original data.")
    with st.expander("Create a cleaned working copy"):
        st.write("Choose safe, reversible transformations. The original uploaded dataset remains unchanged.")
        drop_duplicates=st.checkbox("Remove exact duplicate rows")
        fill_numeric=st.selectbox("Fill missing numeric values",["Leave unchanged","Median","Mean","Zero"])
        if st.button("Create cleaned copy"):
            cleaned=df.copy()
            if drop_duplicates: cleaned=cleaned.drop_duplicates()
            if fill_numeric!="Leave unchanged":
                for c in cleaned.select_dtypes(include=np.number):
                    val={"Median":cleaned[c].median(),"Mean":cleaned[c].mean(),"Zero":0}[fill_numeric]
                    cleaned[c]=cleaned[c].fillna(val)
            st.session_state.cleaned_df=cleaned
            st.success(f"Created working copy: {len(cleaned):,} rows.")
        if st.session_state.get("cleaned_df") is not None:
            st.download_button("Download cleaned CSV",st.session_state.cleaned_df.to_csv(index=False).encode(),"aura_cleaned.csv","text/csv")

elif page == "Reports":
    st.subheader("Report center")
    task = st.session_state.analysis["task"] if st.session_state.analysis else "Dataset overview and data quality assessment"
    answer = st.session_state.analysis["answer"] if st.session_state.analysis else "No AI findings available. Add an API key and run a task in Ask the Analyst to include AI findings."
    st.write("Generate a shareable snapshot of the current dataset and latest AI analysis.")
    st.download_button("Export dataset as Excel", data=(lambda b: b)(io.BytesIO().getvalue()) if False else df.to_csv(index=False).encode(), file_name="aura_dataset.csv", mime="text/csv")
    try:
        excel=io.BytesIO()
        with pd.ExcelWriter(excel,engine="openpyxl") as writer:
            df.to_excel(writer,index=False,sheet_name="Dataset")
            profile_table(df).to_excel(writer,index=False,sheet_name="Column Profile")
        st.download_button("Export workbook as Excel",excel.getvalue(),"aura_analysis.xlsx","application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    except Exception as e: st.caption(f"Excel export unavailable: {e}")
    try:
        pdf=make_pdf_report(df,task,answer)
        st.download_button("Export analysis report as PDF",pdf,"aura_analysis_report.pdf","application/pdf",type="primary")
    except Exception as e: st.error(f"PDF generation failed: {e}")
    st.markdown("#### Analysis history")
    if st.session_state.history: st.dataframe(pd.DataFrame(st.session_state.history),use_container_width=True,hide_index=True)
    else: st.caption("Your completed AI analyses will appear here.")

st.divider()
st.caption("AURA Analytics · First deployable release · Verify AI-generated interpretations before making business decisions.")
