"""
admin_app.py
Local-only admin tool — scrape competitor pages via Jina AI, preview, ingest to ChromaDB, push to GitHub.
Run: streamlit run admin_app.py --server.port 8502
"""

import os
import re
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


def delta_ingest(platform_name: str, content: str) -> tuple[int, int]:
    """Add chunks for a new platform without touching existing data."""
    collection = get_collection()

    # Split by ### subheadings to create chunks
    subsections = re.split(r"(?=^### )", content, flags=re.MULTILINE)
    chunks = []

    if len(subsections) <= 1:
        chunks.append({"text": content.strip(), "title": platform_name})
    else:
        for sub in subsections:
            if not sub.strip():
                continue
            sub_title = sub.split("\n")[0].strip("# ").strip()
            full_text = f"{platform_name} — {sub_title}\n\n{sub.strip()}"
            chunks.append({"text": full_text, "title": f"{platform_name}: {sub_title}"})

    name_hash = hashlib.md5(platform_name.lower().encode()).hexdigest()[:8]
    documents = [c["text"] for c in chunks]
    metadatas = [{"source": SCRAPED_FILE, "title": c["title"]} for c in chunks]
    ids = [f"scraped_{name_hash}_{i}" for i in range(len(chunks))]

    # Skip IDs that already exist (idempotent)
    existing = collection.get(ids=ids)
    existing_ids = set(existing["ids"])

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
    import importlib.util, sys
    spec = importlib.util.spec_from_file_location("ingest_data", "./ingest_data.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    chunks = mod.load_markdown_files(".")
    mod.ingest_to_chromadb(chunks)
    # Clear cached resource so next query uses fresh collection
    st.cache_resource.clear()


def collection_stats() -> int:
    try:
        return get_collection().count()
    except Exception:
        return 0


# ── Jina AI Scraper ────────────────────────────────────────────────────────────
def scrape_with_jina(url: str) -> str:
    jina_url = f"{JINA_PREFIX}{url}"
    resp = requests.get(jina_url, timeout=30, headers={"Accept": "text/plain"})
    resp.raise_for_status()
    return resp.text


# ── Markdown formatter ─────────────────────────────────────────────────────────
def format_as_markdown(platform_name: str, url: str, content: str) -> str:
    date_str = datetime.date.today().strftime("%B %Y")
    return f"""
---

## SCRAPED: {platform_name.upper()}

> **Source:** {url}
> **Scraped:** {date_str}

{content.strip()}
"""


# ── Git helpers ────────────────────────────────────────────────────────────────
def git_push(platform_name: str) -> tuple[bool, str]:
    try:
        subprocess.run(["git", "add", "chroma_db/", SCRAPED_FILE], check=True, capture_output=True)
        result = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True
        )
        if not result.stdout.strip():
            return True, "Nothing new to commit — already up to date."
        msg = f"Admin: scrape + ingest '{platform_name}'"
        subprocess.run(["git", "commit", "-m", msg], check=True, capture_output=True)
        subprocess.run(["git", "push", "origin", "main"], check=True, capture_output=True)
        return True, "Pushed successfully."
    except subprocess.CalledProcessError as e:
        err = e.stderr.decode() if isinstance(e.stderr, bytes) else str(e.stderr)
        return False, err


# ── UI ──────────────────────────────────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="Competitor KB — Admin",
        page_icon="🔧",
        layout="wide",
    )

    st.title("🔧 Competitor KB — Admin Tool")
    st.caption("Local only · Scrape → Preview → Ingest → Push to GitHub")

    # ── Sidebar ────────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("How to use")
        st.markdown("""
        1. Enter **platform name** + **URL**
        2. Click **Scrape** to fetch via Jina AI
        3. Review and edit the preview
        4. Click **Ingest** to add to ChromaDB
        5. Click **Push to GitHub** to sync with Streamlit app
        """)
        st.divider()

        chunk_count = collection_stats()
        st.metric("Chunks in ChromaDB", chunk_count)

        st.divider()
        st.subheader("⚠️ Full Re-ingest")
        st.caption("Deletes and rebuilds the entire ChromaDB from all markdown files. Use only if needed.")
        if st.button("Re-ingest Everything", use_container_width=True):
            with st.spinner("Full re-ingest in progress..."):
                try:
                    full_reingest()
                    st.success("Done! ChromaDB fully rebuilt.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")

    # ── Step 1: Input ──────────────────────────────────────────────────────────
    st.subheader("Step 1 — Enter Platform Details")
    col1, col2 = st.columns([1, 2])
    with col1:
        platform_name = st.text_input("Platform / Provider Name", placeholder="e.g. ExamSoft")
    with col2:
        url = st.text_input("URL to scrape", placeholder="e.g. https://examsoft.com/features")

    scrape_ready = bool(platform_name.strip() and url.strip())
    if st.button("🔍 Scrape with Jina AI", type="primary", disabled=not scrape_ready):
        with st.spinner(f"Fetching {url} via Jina AI..."):
            try:
                raw = scrape_with_jina(url.strip())
                st.session_state.scraped_content = raw
                st.session_state.scraped_platform = platform_name.strip()
                st.session_state.scraped_url = url.strip()
                st.session_state.ingested = False
                st.success(f"Scraped {len(raw):,} characters from {url}")
            except Exception as e:
                st.error(f"Scrape failed: {e}")

    # ── Step 2: Preview & Edit ─────────────────────────────────────────────────
    if "scraped_content" in st.session_state:
        st.divider()
        st.subheader("Step 2 — Preview & Edit")
        st.caption("Edit the content below before ingesting. Remove irrelevant sections (nav menus, footers, etc.).")

        edited = st.text_area(
            label="Scraped content",
            value=st.session_state.scraped_content,
            height=450,
            label_visibility="collapsed",
        )

        col_a, col_b = st.columns([1, 5])
        with col_a:
            if st.button("✅ Ingest to ChromaDB", type="primary"):
                with st.spinner("Ingesting..."):
                    try:
                        # Append formatted block to scraped_platforms.md
                        md_block = format_as_markdown(
                            st.session_state.scraped_platform,
                            st.session_state.scraped_url,
                            edited,
                        )
                        with open(SCRAPED_FILE, "a", encoding="utf-8") as f:
                            f.write(md_block)

                        # Delta ingest
                        added, total = delta_ingest(st.session_state.scraped_platform, edited)

                        st.session_state.ingested = True
                        st.session_state.last_ingested = st.session_state.scraped_platform
                        st.success(f"Added {added} new chunks ({total} total for '{st.session_state.scraped_platform}')")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Ingest failed: {e}")
        with col_b:
            if st.button("🗑️ Clear & Start Over"):
                for key in ["scraped_content", "scraped_platform", "scraped_url", "ingested"]:
                    st.session_state.pop(key, None)
                st.rerun()

    # ── Step 3: Push to GitHub ─────────────────────────────────────────────────
    if st.session_state.get("ingested") and st.session_state.get("last_ingested"):
        st.divider()
        st.subheader("Step 3 — Push to GitHub")
        st.caption(f"Last ingested: **{st.session_state.last_ingested}**. Push to sync Streamlit Cloud.")

        if st.button("🚀 Push to GitHub", type="primary"):
            with st.spinner("Committing and pushing..."):
                ok, msg = git_push(st.session_state.last_ingested)
                if ok:
                    st.success(f"{msg} Reboot the Streamlit Cloud app to pick up the new data.")
                else:
                    st.error(f"Push failed: {msg}")


if __name__ == "__main__":
    main()
