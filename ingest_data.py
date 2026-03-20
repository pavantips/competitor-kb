"""
ingest_data.py
Ingests competitive intelligence markdown into ChromaDB for RAG queries.
Run once locally: python ingest_data.py
"""

import os
import re
import chromadb
from chromadb.utils import embedding_functions

DOCS_DIR = "."  # Directory containing .md files
CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "competitive_intel"


def load_markdown_files(directory: str) -> list[dict]:
    """Load all markdown files and split into platform-level chunks."""
    chunks = []
    for filename in os.listdir(directory):
        if filename != "competitive_analysis.md":
            continue
        filepath = os.path.join(directory, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        # Split on ## PLATFORM headers to get one chunk per platform section
        sections = re.split(r"(?=^## PLATFORM \d+)", content, flags=re.MULTILINE)

        # Also grab the intro, matrix, and strengths/gaps sections
        for section in sections:
            if not section.strip():
                continue

            # Further split long platform sections by ### subheadings
            subsections = re.split(r"(?=^### )", section, flags=re.MULTILINE)

            if len(subsections) <= 1:
                # Short section — keep as one chunk
                title = section.split("\n")[0].strip("# ").strip()
                chunks.append({
                    "text": section.strip(),
                    "source": filename,
                    "title": title,
                })
            else:
                # Platform section — first subsection has platform name
                platform_header = subsections[0]
                platform_name = platform_header.split("\n")[0].strip("# ").strip()

                for sub in subsections[1:]:
                    sub_title = sub.split("\n")[0].strip("# ").strip()
                    # Prepend platform name so context is preserved in each chunk
                    full_text = f"{platform_name} — {sub_title}\n\n{sub.strip()}"
                    chunks.append({
                        "text": full_text,
                        "source": filename,
                        "title": f"{platform_name}: {sub_title}",
                    })

    print(f"✅ Loaded {len(chunks)} chunks from {directory}")
    return chunks


def ingest_to_chromadb(chunks: list[dict]):
    """Store chunks in ChromaDB using the built-in ONNX MiniLM embeddings (no PyTorch needed)."""
    embed_fn = embedding_functions.DefaultEmbeddingFunction()

    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # Delete existing collection if rebuilding
    try:
        client.delete_collection(name=COLLECTION_NAME)
        print(f"🗑️  Deleted existing '{COLLECTION_NAME}' collection")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )

    documents = [c["text"] for c in chunks]
    metadatas = [{"source": c["source"], "title": c["title"]} for c in chunks]
    ids = [f"chunk_{i}" for i in range(len(chunks))]

    # Ingest in batches of 50
    batch_size = 50
    for i in range(0, len(chunks), batch_size):
        batch_docs = documents[i:i + batch_size]
        batch_meta = metadatas[i:i + batch_size]
        batch_ids = ids[i:i + batch_size]
        collection.add(documents=batch_docs, metadatas=batch_meta, ids=batch_ids)
        print(f"   Ingested batch {i // batch_size + 1} ({len(batch_docs)} chunks)")

    print(f"\n✅ Ingestion complete! {len(chunks)} chunks stored in '{CHROMA_DIR}'")
    print(f"   Collection: '{COLLECTION_NAME}'")


if __name__ == "__main__":
    print("🚀 Starting competitive intelligence ingestion...\n")
    chunks = load_markdown_files(DOCS_DIR)
    ingest_to_chromadb(chunks)
    print("\n🎉 Ready! Run: streamlit run app.py")
