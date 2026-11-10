# AURA — Autonomous AI Data Analyst (deployable MVP)

A working first release of the product concept. It includes:
- CSV, Excel, JSON and Parquet uploads
- Read-only SQL table loading for PostgreSQL, MySQL, SQL Server and SQLite
- Dataset overview, column profiling, missing-value and duplicate checks
- Reversible cleaning into a separate working copy
- Interactive Plotly charts and correlation heatmaps
- Optional natural-language AI analysis using your own OpenAI API key
- CSV, Excel and PDF exports
- Docker deployment

## Run locally

Requires Python 3.11+.

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Open http://localhost:8501

## Run with Docker

```bash
docker compose up --build -d
```

Open http://localhost:8501

## Hosting

This app needs a Python hosting environment (not static-only hosting). It can run on a VPS, a Docker-capable host, or a Python app platform that supports Streamlit and outbound database connections. Upload the ZIP, build the Docker image, expose port 8501, and put it behind HTTPS.

For a public production deployment, add authentication/SSO, HTTPS, persistent storage, secret management, rate limits, per-user isolation, audit logging, and a managed database before allowing real customer data.

## AI setup

In the left sidebar, enter your own OpenAI API key and model. The key is held in the current Streamlit session and is not written to a file by this app. Without a key, profiling, data quality checks, charts, SQL loading and exports still work.

## SQL support

- PostgreSQL: host, port, database, username, password
- MySQL: host, port, database, username, password
- SQL Server: host, port, database, username, password; host must have a compatible ODBC driver
- SQLite: local database file path on the server

Use a dedicated read-only database account. The SQL loader only accepts SELECT/WITH queries and limits returned rows. Do not expose database credentials or this app publicly without authentication and HTTPS.

## Important scope note

This is a functional deployable MVP, not yet the complete enterprise platform described in the product brief. BigQuery, Snowflake, REST API ingestion, multi-user RBAC, persistent projects, multi-agent orchestration, scheduled monitoring, advanced forecasting/model training, and production-grade security are not implemented in this first release. The AI assistant currently receives schema/profile metadata and a small sample, so it should not be treated as having exhaustively computed whole-dataset findings. Use the deterministic profiling and charting features to verify conclusions.
