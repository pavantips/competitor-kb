# CLAUDE.md — Project Conventions & Repeated Instructions

This file captures standing instructions, decisions, and patterns established across the Meazure Competitor KB and API Agent projects. Claude should follow these automatically without being asked.

---

## Python Environment

- Always use the local `venv` — never system Python or global pip
- Commands:
  ```bash
  venv/bin/python        # run scripts
  venv/bin/pip           # install packages
  venv/bin/streamlit     # launch Streamlit apps
  ```

---

## Streamlit App Ports

| App | Port |
|---|---|
| Main chatbot (`app.py`) | 8501 (default) |
| Admin tool (`admin_app.py`) | 8502 |

Always start admin app with `--server.port 8502` to avoid conflicts.

---

## ChromaDB Rules

- Use `DefaultEmbeddingFunction()` — NOT `SentenceTransformerEmbeddingFunction`
  - Reason: avoids PyTorch dependency, works on Streamlit Cloud free tier
- Always commit `chroma_db/` including `chroma.sqlite3` to GitHub — Streamlit Cloud reads it from there
- After any ingest (local), always push `chroma_db/` + markdown files to GitHub, then reboot Streamlit Cloud app
- Delta ingests: add chunks without deleting the collection (`get_or_create_collection`)
- Full re-ingest: delete + recreate collection from all markdown files

---

## Data Files

| File | Purpose |
|---|---|
| `competitive_analysis_updated.md` | Base competitor data — do not modify manually |
| `scraped_platforms.md` | Delta file — new platforms added via admin tool |
| `requirements.txt` | Streamlit Cloud dependencies only |
| `requirements-admin.txt` | Local admin tool dependencies (not deployed) |

`ingest_data.py` loads both `competitive_analysis_updated.md` and `scraped_platforms.md`.

---

## Known Dependency Fixes

- **protobuf conflict** (`Descriptors cannot be created directly`):
  ```
  protobuf>=3.20.0,<4.0.0
  ```
  Add to `requirements.txt` — affects Streamlit Cloud deploys.

- **Admin tool extra packages** (local only, not in `requirements.txt`):
  ```
  pypdf, python-docx, pandas, openpyxl
  ```

---

## GitHub Workflow

- Always use the existing `venv/bin/git` flow — no force pushes
- If push is rejected (non-fast-forward): `git pull --rebase origin main` then push
- Commit `chroma_db/` after every ingest — Streamlit Cloud has no local storage
- After pushing: user must manually reboot app on Streamlit Cloud dashboard

---

## Deployment Rules

- **`app.py`** → Streamlit Cloud (public, users access this)
- **`admin_app.py`** → local only, never deployed — contains ingestion + git push controls
- **API keys** → `.streamlit/secrets.toml` only — this file is in `.gitignore`, never committed
- Streamlit Cloud secret: `ANTHROPIC_API_KEY` set via app dashboard → Advanced Settings → Secrets

---

## Model & Cost Preferences

- Default model: `claude-haiku-4-5` (cheapest, fast enough for Q&A)
- Embedding: local ONNX `DefaultEmbeddingFunction` — free, no API cost
- Scraping: Jina AI (`https://r.jina.ai/{url}`) — free tier, no API key needed
- Ingestion costs nothing — embeddings run locally

---

## Streamlit State Management

- Never use the same string as both a widget `key=` and a `st.session_state` key — causes conflict
  - Pattern: widget uses `key="doc_platform"`, stored state uses `key="doc_platform_name"`
- Shared UI functions called in multiple tabs must receive a unique `key` parameter for buttons

---

## Data Ingestion Workflow (Admin Tool)

1. Scrape URL via Jina AI **or** upload document (PDF, DOCX, XLSX, CSV)
2. Preview and edit extracted content
3. Click **Ingest to ChromaDB** — delta only, fast
4. Click **Push to GitHub** — critical, without this Streamlit Cloud won't see the data
5. Reboot Streamlit Cloud app from dashboard

For tabular files (XLSX/CSV): each row becomes one chunk.
For text files (PDF/DOCX/scraped): split by `###` subheadings into chunks.

---

## Sidebar Update Checklist

When adding new platforms, update `app.py`:
- [ ] Platform count in `SYSTEM_PROMPT` string
- [ ] Platform count in sidebar About section
- [ ] Platform list in sidebar
- [ ] `EXAMPLE_QUESTIONS` list — add 1-2 questions per new platform
