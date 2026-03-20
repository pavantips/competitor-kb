"""
app.py
Meazure Learning — Competitive Intelligence RAG Chatbot
Run: streamlit run app.py
"""

import os
import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
import anthropic

# ── Config ────────────────────────────────────────────────────────────────────
CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "competitive_intel"
MODEL = "claude-haiku-4-5"   # Cost-optimized; swap to claude-sonnet-4-6 for richer answers
MAX_RESULTS = 5              # Number of ChromaDB chunks to retrieve per query

SYSTEM_PROMPT = """You are a competitive intelligence assistant for Meazure Learning's 
Solutions Consulting and RFP team. You have access to detailed knowledge about 11 
exam/assessment platforms that compete with or complement Meazure Learning.

Your job is to:
1. Answer questions about competitor capabilities, features, and positioning
2. Help craft RFP responses by comparing platforms accurately
3. Provide honest assessments including where Meazure has gaps
4. Suggest positioning strategies based on the competitive landscape

Always base your answers on the provided context chunks. If you don't have enough 
information to answer confidently, say so and suggest what additional research might help.
Be concise, direct, and sales-team-friendly. Use bullet points for comparisons."""

EXAMPLE_QUESTIONS = [
    "Does Caveon support AI item generation?",
    "Which competitors have offline exam delivery?",
    "Compare proctoring options: Meazure vs Prolydian vs Cirrus",
    "What are Meazure's known gaps vs competitors?",
    "Which platforms are partners vs competitors for Meazure?",
    "What's Excelsoft's unique differentiator?",
    "If an RFP requires GDPR compliance, who should I worry about?",
    "Which platforms target the healthcare/nursing market?",
]

# ── ChromaDB setup ─────────────────────────────────────────────────────────────
@st.cache_resource
def get_collection():
    embed_fn = embedding_functions.DefaultEmbeddingFunction()
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client.get_collection(name=COLLECTION_NAME, embedding_function=embed_fn)


def retrieve_context(query: str, n_results: int = MAX_RESULTS) -> tuple[str, list[str]]:
    """Query ChromaDB and return formatted context + source titles."""
    collection = get_collection()
    results = collection.query(query_texts=[query], n_results=n_results)
    
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    
    context_parts = []
    sources = []
    for doc, meta in zip(docs, metas):
        context_parts.append(f"[Source: {meta['title']}]\n{doc}")
        sources.append(meta["title"])
    
    return "\n\n---\n\n".join(context_parts), sources


# ── Claude API ─────────────────────────────────────────────────────────────────
def get_api_key() -> str:
    """Get API key from Streamlit secrets or environment."""
    if "ANTHROPIC_API_KEY" in st.secrets:
        return st.secrets["ANTHROPIC_API_KEY"]
    key = os.getenv("ANTHROPIC_API_KEY", "")
    if not key:
        st.error("⚠️ ANTHROPIC_API_KEY not found. Add it to .env or Streamlit secrets.")
        st.stop()
    return key


def ask_claude(query: str, context: str, history: list) -> str:
    """Send query + retrieved context to Claude and return the answer."""
    client = anthropic.Anthropic(api_key=get_api_key())
    
    # Build messages with conversation history
    messages = []
    for msg in history[-6:]:  # Last 3 turns for context
        messages.append({"role": msg["role"], "content": msg["content"]})
    
    # Add current query with retrieved context
    user_content = f"""Based on the following competitive intelligence context, please answer my question.

CONTEXT:
{context}

QUESTION: {query}"""
    
    messages.append({"role": "user", "content": user_content})
    
    response = client.messages.create(
        model=MODEL,
        max_tokens=1000,
        system=SYSTEM_PROMPT,
        messages=messages,
    )
    return response.content[0].text


# ── UI ─────────────────────────────────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="Meazure Competitive Intel",
        page_icon="🎯",
        layout="wide",
    )

    # Header
    st.title("🎯 Competitive Intelligence Assistant")
    st.caption("Meazure Learning | Solutions Consulting | Powered by Claude + ChromaDB")

    # Sidebar
    with st.sidebar:
        st.header("About")
        st.markdown("""
        This RAG chatbot answers questions about **11 exam/assessment platforms** 
        using a vector database of competitive intelligence research.
        
        **Platforms covered:**
        - Caveon
        - Surpass  
        - Cirrus
        - Excelsoft (Saras)
        - ITS (Internet Testing Systems)
        - Elsevier (HESI/Evolve)
        - Questionmark
        - Risr
        - Prolydian
        - ROC-P
        - CMS / AUTHORWise
        """)

        st.divider()
        st.subheader("💡 Example Questions")
        for q in EXAMPLE_QUESTIONS:
            if st.button(q, key=q, use_container_width=True):
                st.session_state.pending_question = q

        st.divider()
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        st.divider()
        st.caption(f"Model: `{MODEL}`")
        st.caption(f"Retrieving top {MAX_RESULTS} chunks per query")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("📚 Sources used"):
                    for src in msg["sources"]:
                        st.markdown(f"- {src}")

    # Handle sidebar button clicks
    pending = st.session_state.pop("pending_question", None)

    # Chat input
    user_input = st.chat_input("Ask about any competitor platform...") or pending

    if user_input:
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Retrieve + generate
        with st.chat_message("assistant"):
            with st.spinner("Searching competitive intel database..."):
                try:
                    context, sources = retrieve_context(user_input)
                    answer = ask_claude(user_input, context, st.session_state.messages[:-1])
                    
                    st.markdown(answer)
                    with st.expander("📚 Sources used"):
                        for src in sources:
                            st.markdown(f"- {src}")

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                    })

                except Exception as e:
                    err_msg = f"❌ Error: {str(e)}"
                    st.error(err_msg)
                    if "chroma_db" in str(e).lower() or "collection" in str(e).lower():
                        st.info("💡 Have you run `python ingest_data.py` yet?")


if __name__ == "__main__":
    main()
