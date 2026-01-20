import streamlit as st
import tempfile
import os
import sys

# -------------------------------------------------
# TEMPORARY PATH FIX
# -------------------------------------------------
sys.path.insert(0, "src")

# -------------------------------------------------
# PROJECT IMPORTS
# -------------------------------------------------
from ingestion.ingestion_pipeline import run_ingestion
from llm.llm_answer import answer_query
from core.constants import DocumentType
from core.exception import RAGError
from utils.small_talk import is_small_talk, small_talk_response

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="RAG Chatbot",
    layout="centered"
)

st.title("📄 Retrieval-Augmented Generation Chatbot")

# -------------------------------------------------
# SESSION STATE
# -------------------------------------------------
st.session_state.setdefault("ingested", False)
st.session_state.setdefault("upload_dir", None)
st.session_state.setdefault("messages", [])

# -------------------------------------------------
# DOCUMENT UPLOAD
# -------------------------------------------------
st.header("1. Upload Documents")

uploaded_files = st.file_uploader(
    "Upload documents (PDF / TXT / DOCX)",
    type=["pdf", "txt", "docx"],
    accept_multiple_files=True
)

doc_type = st.selectbox(
    "Select document type",
    options=[
        DocumentType.UNSTRUCTURED,
        DocumentType.GENERAL_STRUCTURED,
        DocumentType.ISO_STRUCTURED,
    ],
    format_func=lambda x: x.value.replace("_", " ").title()
)

if uploaded_files and st.button("Run Ingestion"):
    with st.spinner("Ingesting documents..."):
        try:
            tmp_dir = tempfile.mkdtemp()

            for file in uploaded_files:
                with open(os.path.join(tmp_dir, file.name), "wb") as f:
                    f.write(file.getbuffer())

            run_ingestion(upload_dir=tmp_dir, document_type=doc_type)

            st.session_state.ingested = True
            st.session_state.upload_dir = tmp_dir
            st.session_state.messages = []  # reset chat on new ingestion

            st.success("Ingestion completed successfully.")

        except Exception as e:
            st.error(str(e))

# -------------------------------------------------
# CHAT INTERFACE
# -------------------------------------------------
if st.session_state.ingested:
    st.divider()
    st.header("2. Chat with Documents")

    # ---- Render chat history ----
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            if msg["role"] == "assistant" and msg.get("chunks"):
                with st.expander("📎 Sources"):
                    for i, doc in enumerate(msg["chunks"], start=1):
                        meta = doc.metadata or {}

                        st.markdown(f"**Chunk {i}**")
                        st.markdown(doc.page_content)

                        st.caption(
                            f"📄 {meta.get('file_name', 'N/A')} | "
                            f"📑 Pages {meta.get('pages', 'N/A')} | "
                            f"🔁 Rerank {round(meta.get('rerank_score', 0), 2)}"
                        )
                        st.markdown("---")

    # ---- Chat input ----
    user_query = st.chat_input("Ask a question")

    if user_query:
        with st.chat_message("user"):
            st.markdown(user_query)

    # 2️⃣ Save for history
        st.session_state.messages.append({
        "role": "user",
        "content": user_query
        })


        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # 🔹 INTENT ROUTING
                    if is_small_talk(user_query):
                        answer = small_talk_response(user_query)
                        chunks = None
                    else:
                        result = answer_query(user_query)
                        answer = result.get("answer", "")
                        chunks = result.get("chunks", [])

                    st.markdown(answer)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "chunks": chunks
                    })

                except RAGError as e:
                    st.error(f"Query failed: {e}")
                except Exception as e:
                    st.error(f"Unexpected error: {e}")



else:
    st.info("Upload and ingest documents to enable chatting.")
