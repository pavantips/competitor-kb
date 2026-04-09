"""
admin_app.py
Local-only admin tool for the Competitor KB.
Two ingestion modes: scrape from URL (Jina AI) or upload a document (.pdf, .docx, .xlsx, .csv).
Run: streamlit run admin_app.py --server.port 8502
"""

import io
import os
import re
import csv
import hashlib
import datetime
import subprocess
import requests
import streamlit as st
import chromadb
from chromadb.utils import embedding_functions

# ── Config ─────────────────────────────────────────────────────────────────────
CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "competitive_intel"
SCRAPED_FILE = "./scraped_platforms.md"
JINA_PREFIX = "https://r.jina.ai/"

# ── ChromaDB ───────────────────────────────────────────────────────────────────
@st.cache_resource
def get_collection():
    embed_fn = embedding_functions.DefaultEmbeddingFunction()
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )


def delta_ingest_text(platform_name: str, content: str) -> tuple[int, int]:
    """Ingest free-text content by splitting on ### subheadings."""
    collection = get_collection()
    subsections = re.split(r"(?=^### )", content, flags=re.MULTILINE)
    chunks = []

    if len(subsections) <= 1:
        chunks.append({"text": content.strip(), "title": platform_name})
    else:
        for sub in subsections:
            if not sub.strip():
                continue
            sub_title = sub.split("\n")[0].strip("# ").strip()
            chunks.append({
                "text": f"{platform_name} — {sub_title}\n\n{sub.strip()}",
                "title": f"{platform_name}: {sub_title}",
            })

    return _add_chunks(platform_name, chunks, prefix="scraped")


def delta_ingest_rows(platform_name: str, rows: list[dict]) -> tuple[int, int]:
    """Ingest pre-formed row chunks (from tabular data)."""
    collection = get_collection()
    chunks = [
        {
            "text": r["text"],
            "title": r["title"],
        }
        for r in rows
    ]
    return _add_chunks(platform_name, chunks, prefix="tabular")


def _add_chunks(platform_name: str, chunks: list[dict], prefix: str) -> tuple[int, int]:
    """Shared logic: deduplicate by ID and add to collection."""
    collection = get_collection()
    name_hash = hashlib.md5(platform_name.lower().encode()).hexdigest()[:8]
    documents = [c["text"] for c in chunks]
    metadatas = [{"source": SCRAPED_FILE, "title": c["title"]} for c in chunks]
    ids = [f"{prefix}_{name_hash}_{i}" for i in range(len(chunks))]

    existing_ids = set(collection.get(ids=ids)["ids"])
    new_docs, new_meta, new_ids = [], [], []
    for doc, meta, id_ in zip(documents, metadatas, ids):
        if id_ not in existing_ids:
            new_docs.append(doc)
            new_meta.append(meta)
            new_ids.append(id_)

    if new_docs:
        collection.add(documents=new_docs, metadatas=new_meta, ids=new_ids)

    return len(new_docs), len(chunks)


def full_reingest():
    """Delete collection and rebuild from all data files."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("ingest_data", "./ingest_data.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    chunks = mod.load_markdown_files(".")
    mod.ingest_to_chromadb(chunks)
    st.cache_resource.clear()


def collection_stats() -> int:
    try:
        return get_collection().count()
    except Exception:
        return 0


# ── Jina AI Scraper ────────────────────────────────────────────────────────────
def scrape_with_jina(url: str) -> str:
    resp = requests.get(f"{JINA_PREFIX}{url}", timeout=30, headers={"Accept": "text/plain"})
    resp.raise_for_status()
    return resp.text


# ── Document Extractors ────────────────────────────────────────────────────────
def extract_pdf(file_bytes: bytes) -> str:
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(file_bytes))
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text.strip())
    return "\n\n".join(pages)


def extract_docx(file_bytes: bytes) -> str:
    from docx import Document
    doc = Document(io.BytesIO(file_bytes))
    paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    return "\n\n".join(paragraphs)


def extract_xlsx(file_bytes: bytes) -> tuple[object, list[dict]]:
    """Returns (dataframe, list of row chunks)."""
    import pandas as pd
    df = pd.read_excel(io.BytesIO(file_bytes))
    df = df.fillna("").astype(str)
    return df, _df_to_chunks(df)


def extract_csv(file_bytes: bytes) -> tuple[object, list[dict]]:
    """Returns (dataframe, list of row chunks)."""
    import pandas as pd
    df = pd.read_csv(io.BytesIO(file_bytes))
    df = df.fillna("").astype(str)
    return df, _df_to_chunks(df)


def _df_to_chunks(df) -> list[dict]:
    """Convert each DataFrame row into a text chunk."""
    chunks = []
    headers = df.columns.tolist()
    for idx, row in df.iterrows():
        parts = [f"{col}: {val}" for col, val in zip(headers, row) if str(val).strip() and str(val) != "nan"]
        if parts:
            chunks.append({
                "text": " | ".join(parts),
                "title": f"Row {idx + 1}: {parts[0] if parts else ''}",
            })
    return chunks


# ── Markdown formatter ─────────────────────────────────────────────────────────
def format_as_markdown(platform_name: str, source_label: str, content: str) -> str:
    date_str = datetime.date.today().strftime("%B %Y")
    return f"""
---

## UPLOADED: {platform_name.upper()}

> **Source:** {source_label}
> **Added:** {date_str}

{content.strip()}
"""


# ── Git helpers ────────────────────────────────────────────────────────────────
def git_push(platform_name: str) -> tuple[bool, str]:
    try:
        subprocess.run(["git", "add", "chroma_db/", SCRAPED_FILE], check=True, capture_output=True)
        result = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
        if not result.stdout.strip():
            return True, "Nothing new to commit — already up to date."
        subprocess.run(
            ["git", "commit", "-m", f"Admin: ingest '{platform_name}'"],
            check=True, capture_output=True,
        )
        subprocess.run(["git", "push", "origin", "main"], check=True, capture_output=True)
        return True, "Pushed successfully."
    except subprocess.CalledProcessError as e:
        return False, e.stderr.decode() if isinstance(e.stderr, bytes) else str(e.stderr)


# ── Shared Push Footer ─────────────────────────────────────────────────────────
def render_push_section(key: str = "push_btn"):
    if st.session_state.get("ingested") and st.session_state.get("last_ingested"):
        st.divider()
        st.subheader("Step 3 — Push to GitHub")
        st.caption(f"Last ingested: **{st.session_state.last_ingested}** · Push to sync Streamlit Cloud.")
        if st.button("🚀 Push to GitHub", type="primary", key=key):
            with st.spinner("Committing and pushing..."):
                ok, msg = git_push(st.session_state.last_ingested)
                if ok:
                    st.success(f"{msg} Reboot the Streamlit Cloud app to load the new data.")
                else:
                    st.error(f"Push failed: {msg}")


# ── UI ─────────────────────────────────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="Competitor KB — Admin",
        page_icon="🔧",
        layout="wide",
    )

    st.title("🔧 Competitor KB — Admin Tool")
    st.caption("Local only · Add new competitors via URL scrape or document upload → Ingest → Push to GitHub")

    # ── Sidebar ────────────────────────────────────────────────────────────────
    with st.sidebar:
        st.metric("Chunks in ChromaDB", collection_stats())
        st.divider()
        st.subheader("⚠️ Full Re-ingest")
        st.caption("Deletes and rebuilds the entire ChromaDB. Use only if needed.")
        if st.button("Re-ingest Everything", use_container_width=True):
            with st.spinner("Full re-ingest in progress..."):
                try:
                    full_reingest()
                    st.success("Done! ChromaDB fully rebuilt.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")

    # ── Tabs ───────────────────────────────────────────────────────────────────
    tab_url, tab_doc = st.tabs(["🌐  Scrape from URL", "📄  Upload Document"])

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 1 — Scrape from URL
    # ══════════════════════════════════════════════════════════════════════════
    with tab_url:
        st.subheader("Step 1 — Enter Platform Details")
        col1, col2 = st.columns([1, 2])
        with col1:
            url_platform = st.text_input("Platform / Provider Name", placeholder="e.g. ExamSoft", key="url_platform")
        with col2:
            url_input = st.text_input("URL to scrape", placeholder="e.g. https://examsoft.com/features", key="url_input")

        if st.button("🔍 Scrape with Jina AI", type="primary",
                     disabled=not (url_platform.strip() and url_input.strip()), key="scrape_btn"):
            with st.spinner(f"Fetching via Jina AI..."):
                try:
                    raw = scrape_with_jina(url_input.strip())
                    st.session_state.scraped_content = raw
                    st.session_state.scraped_platform = url_platform.strip()
                    st.session_state.scraped_url = url_input.strip()
                    st.session_state.ingested = False
                    st.success(f"Scraped {len(raw):,} characters")
                except Exception as e:
                    st.error(f"Scrape failed: {e}")

        if "scraped_content" in st.session_state:
            st.divider()
            st.subheader("Step 2 — Preview & Edit")
            st.caption("Remove nav menus, footers, CTAs, and other noise before ingesting.")

            edited_url = st.text_area(
                "Scraped content (editable)",
                value=st.session_state.scraped_content,
                height=450,
                key="url_edit_area",
                label_visibility="collapsed",
            )

            col_a, col_b = st.columns([1, 5])
            with col_a:
                if st.button("✅ Ingest to ChromaDB", type="primary", key="url_ingest_btn"):
                    with st.spinner("Ingesting..."):
                        try:
                            md_block = format_as_markdown(
                                st.session_state.scraped_platform,
                                st.session_state.scraped_url,
                                edited_url,
                            )
                            with open(SCRAPED_FILE, "a", encoding="utf-8") as f:
                                f.write(md_block)

                            added, total = delta_ingest_text(st.session_state.scraped_platform, edited_url)
                            st.session_state.ingested = True
                            st.session_state.last_ingested = st.session_state.scraped_platform
                            st.success(f"Added {added} new chunks ({total} total)")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Ingest failed: {e}")
            with col_b:
                if st.button("🗑️ Clear", key="url_clear_btn"):
                    for k in ["scraped_content", "scraped_platform", "scraped_url"]:
                        st.session_state.pop(k, None)
                    st.rerun()

        render_push_section(key="push_btn_url")

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 2 — Upload Document
    # ══════════════════════════════════════════════════════════════════════════
    with tab_doc:
        st.subheader("Step 1 — Upload Document")
        st.caption("Supported formats: PDF, DOCX, XLSX, CSV")

        col1, col2 = st.columns([1, 2])
        with col1:
            doc_platform = st.text_input("Platform / Provider Name", placeholder="e.g. ExamSoft", key="doc_platform")
        with col2:
            uploaded_file = st.file_uploader(
                "Choose a file",
                type=["pdf", "docx", "xlsx", "csv"],
                key="file_uploader",
            )

        if uploaded_file and doc_platform.strip():
            if st.button("📂 Extract Content", type="primary", key="extract_btn"):
                with st.spinner(f"Extracting from {uploaded_file.name}..."):
                    try:
                        file_bytes = uploaded_file.read()
                        ext = uploaded_file.name.rsplit(".", 1)[-1].lower()

                        if ext == "pdf":
                            text = extract_pdf(file_bytes)
                            st.session_state.doc_content = text
                            st.session_state.doc_type = "text"
                            st.success(f"Extracted {len(text):,} characters from PDF")

                        elif ext == "docx":
                            text = extract_docx(file_bytes)
                            st.session_state.doc_content = text
                            st.session_state.doc_type = "text"
                            st.success(f"Extracted {len(text):,} characters from DOCX")

                        elif ext == "xlsx":
                            df, row_chunks = extract_xlsx(file_bytes)
                            st.session_state.doc_df = df
                            st.session_state.doc_rows = row_chunks
                            st.session_state.doc_type = "tabular"
                            st.success(f"Extracted {len(df)} rows × {len(df.columns)} columns from XLSX")

                        elif ext == "csv":
                            df, row_chunks = extract_csv(file_bytes)
                            st.session_state.doc_df = df
                            st.session_state.doc_rows = row_chunks
                            st.session_state.doc_type = "tabular"
                            st.success(f"Extracted {len(df)} rows × {len(df.columns)} columns from CSV")

                        st.session_state.doc_platform_name = doc_platform.strip()
                        st.session_state.doc_filename = uploaded_file.name
                        st.session_state.ingested = False

                    except Exception as e:
                        st.error(f"Extraction failed: {e}")
                        st.info("Make sure the required packages are installed: pip install pypdf python-docx pandas openpyxl")

        # ── Preview ────────────────────────────────────────────────────────
        if "doc_type" in st.session_state:
            st.divider()
            st.subheader("Step 2 — Preview & Edit")

            if st.session_state.doc_type == "text":
                st.caption("Review and edit the extracted text before ingesting.")
                edited_doc = st.text_area(
                    "Extracted content (editable)",
                    value=st.session_state.doc_content,
                    height=450,
                    key="doc_edit_area",
                    label_visibility="collapsed",
                )

                col_a, col_b = st.columns([1, 5])
                with col_a:
                    if st.button("✅ Ingest to ChromaDB", type="primary", key="doc_ingest_text_btn"):
                        with st.spinner("Ingesting..."):
                            try:
                                md_block = format_as_markdown(
                                    st.session_state.doc_platform_name,
                                    st.session_state.doc_filename,
                                    edited_doc,
                                )
                                with open(SCRAPED_FILE, "a", encoding="utf-8") as f:
                                    f.write(md_block)

                                added, total = delta_ingest_text(st.session_state.doc_platform_name, edited_doc)
                                st.session_state.ingested = True
                                st.session_state.last_ingested = st.session_state.doc_platform_name
                                st.success(f"Added {added} new chunks ({total} total)")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Ingest failed: {e}")
                with col_b:
                    if st.button("🗑️ Clear", key="doc_text_clear_btn"):
                        for k in ["doc_content", "doc_type", "doc_platform_name", "doc_filename"]:
                            st.session_state.pop(k, None)
                        st.rerun()

            elif st.session_state.doc_type == "tabular":
                st.caption(f"Each row will become one chunk. {len(st.session_state.doc_rows)} rows ready to ingest.")

                # Show the dataframe
                st.dataframe(st.session_state.doc_df, use_container_width=True, height=300)

                # Show how chunks look
                with st.expander("Preview chunk format (first 3 rows)"):
                    for row in st.session_state.doc_rows[:3]:
                        st.markdown(f"**{row['title']}**")
                        st.code(row["text"])

                col_a, col_b = st.columns([1, 5])
                with col_a:
                    if st.button("✅ Ingest to ChromaDB", type="primary", key="doc_ingest_tabular_btn"):
                        with st.spinner("Ingesting rows..."):
                            try:
                                # Save a text summary to scraped_platforms.md
                                summary = "\n".join([r["text"] for r in st.session_state.doc_rows])
                                md_block = format_as_markdown(
                                    st.session_state.doc_platform_name,
                                    st.session_state.doc_filename,
                                    summary,
                                )
                                with open(SCRAPED_FILE, "a", encoding="utf-8") as f:
                                    f.write(md_block)

                                added, total = delta_ingest_rows(
                                    st.session_state.doc_platform_name,
                                    st.session_state.doc_rows,
                                )
                                st.session_state.ingested = True
                                st.session_state.last_ingested = st.session_state.doc_platform_name
                                st.success(f"Added {added} new chunks ({total} rows total)")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Ingest failed: {e}")
                with col_b:
                    if st.button("🗑️ Clear", key="doc_tabular_clear_btn"):
                        for k in ["doc_df", "doc_rows", "doc_type", "doc_platform_name", "doc_filename"]:
                            st.session_state.pop(k, None)
                        st.rerun()

        render_push_section(key="push_btn_doc")


if __name__ == "__main__":
    main()
