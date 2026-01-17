import streamlit as st
import tempfile
import os
import sys

# -------------------------------------------------
# TEMPORARY PATH FIX (acceptable for Streamlit)
# -------------------------------------------------
sys.path.insert(0, "src")

# -------------------------------------------------
# PROJECT IMPORTS (single source of truth)
# -------------------------------------------------
from ingestion.ingestion_pipeline import run_ingestion
from llm.llm_answer import answer_query
from core.constants import DocumentType
from core.exception import RAGError

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="RAG System",
    layout="centered"
)

st.title("📄 Retrieval-Augmented Generation")

# -------------------------------------------------
# SESSION STATE
# -------------------------------------------------
if "ingested" not in st.session_state:
    st.session_state.ingested = False

if "upload_dir" not in st.session_state:
    st.session_state.upload_dir = None

# -------------------------------------------------
# DOCUMENT UPLOAD
# -------------------------------------------------
st.header("1. Upload Documents")

uploaded_files = st.file_uploader(
    label="Upload documents (PDF / TXT / DOCX)",
    accept_multiple_files=True,
    type=["pdf", "txt", "docx"]
)

# -------------------------------------------------
# DOCUMENT TYPE SELECTION (EXPLICIT USER INTENT)
# -------------------------------------------------
doc_type = st.selectbox(
    "Select document type",
    options=[
        DocumentType.UNSTRUCTURED,
        DocumentType.GENERAL_STRUCTURED,
        DocumentType.ISO_STRUCTURED,
    ],
    format_func=lambda x: x.value.replace("_", " ").title()
)

# -------------------------------------------------
# INGESTION ACTION (ONE AND ONLY ONE)
# -------------------------------------------------
if uploaded_files and st.button("Run Ingestion"):
    with st.spinner("Ingesting documents..."):
        try:
            tmp_dir = tempfile.mkdtemp()

            for file in uploaded_files:
                file_path = os.path.join(tmp_dir, file.name)
                with open(file_path, "wb") as f:
                    f.write(file.getbuffer())

            run_ingestion(
                upload_dir=tmp_dir,
                document_type=doc_type
            )

            st.session_state.ingested = True
            st.session_state.upload_dir = tmp_dir

            st.success("Ingestion completed successfully.")

        except RAGError as e:
            st.error(f"Ingestion failed: {e}")
        except Exception as e:
            st.error(f"Unexpected error: {e}")

# -------------------------------------------------
# QUERY INTERFACE
# -------------------------------------------------
if st.session_state.ingested:
    st.divider()
    st.header("2. Ask a Question")

    query = st.text_input("Enter your question")

    if query and st.button("Get Answer"):
        with st.spinner("Generating answer..."):
            try:
                result = answer_query(query)

                # assuming answer_query returns structured output
                answer = (
                    result["answer"]
                    if isinstance(result, dict) and "answer" in result
                    else result
                )

                st.subheader("Answer")
                st.write(answer)

            except RAGError as e:
                st.error(f"Query failed: {e}")
            except Exception as e:
                st.error(f"Unexpected error: {e}")
else:
    st.info("Upload and ingest documents to enable querying.")
