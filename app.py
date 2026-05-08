
import streamlit as st

from config import AppConfig
from document_service import DocumentService
from vector_store_service import VectorStoreService
from memory_service import MemoryService
from rag_service import RAGService


st.set_page_config(
    page_title="RAG PDF Q&A",
    page_icon="📚",
    layout="wide",
)


def init_app():
    """Initialize services and session variables."""

    if "config" not in st.session_state:
        st.session_state.config = AppConfig()

    if "document_service" not in st.session_state:
        st.session_state.document_service = DocumentService(
            st.session_state.config
        )

    if "vector_service" not in st.session_state:
        st.session_state.vector_service = VectorStoreService(
            st.session_state.config
        )

    if "memory_service" not in st.session_state:
        st.session_state.memory_service = MemoryService(
            max_history=st.session_state.config.MAX_CHAT_HISTORY
        )

    if "rag_service" not in st.session_state:
        st.session_state.rag_service = RAGService(
            st.session_state.config,
            st.session_state.memory_service,
        )

    if "api_key" not in st.session_state:
        st.session_state.api_key = ""

    if "session_id" not in st.session_state:
        st.session_state.session_id = ""

  
    if "session_files" not in st.session_state:
        st.session_state.session_files = {}

    # Stores uploaded file signatures per session to avoid repeated ingestion
    if "file_signatures" not in st.session_state:
        st.session_state.file_signatures = {}

    # Used to reset the visible uploader after clearing session PDFs
    if "uploader_versions" not in st.session_state:
        st.session_state.uploader_versions = {}

    if "chain_session" not in st.session_state:
        st.session_state.chain_session = None


def get_file_signature(file):
    """Return a simple unique signature for a file."""

    if file is None:
        return None

    return f"{file.name}_{file.size}"


def get_current_pdf_names():
    """
    Get current PDF names for the session.
    First check Streamlit session_state.
    If missing after app restart, recover names from Chroma metadata.
    """

    session_id = st.session_state.session_id

    if not session_id:
        return []

    files = st.session_state.session_files.get(session_id)

    if files:
        # Support old format if it was stored as a string before
        if isinstance(files, str):
            files = [files]
            st.session_state.session_files[session_id] = files

        return files

    # Recover file names from persistent Chroma metadata
    source_files = st.session_state.vector_service.get_source_files(session_id)

    if source_files:
        source_files = sorted(source_files)
        st.session_state.session_files[session_id] = source_files
        return source_files

    return []


def get_uploader_key():
    """Create a unique uploader key per session."""

    session_id = st.session_state.session_id or "no_session"

    if session_id not in st.session_state.uploader_versions:
        st.session_state.uploader_versions[session_id] = 0

    version = st.session_state.uploader_versions[session_id]

    return f"pdf_uploader_{session_id}_{version}"


def reset_uploader():
    """Reset file uploader for the current session."""

    session_id = st.session_state.session_id or "no_session"

    if session_id not in st.session_state.uploader_versions:
        st.session_state.uploader_versions[session_id] = 0

    st.session_state.uploader_versions[session_id] += 1


def auto_ingest_pdfs(uploaded_files):
    """
    Automatically ingest new PDFs for the current session.
    Existing PDFs in the same session are kept.
    New PDFs are added to the same Chroma collection.
    """

    session_id = st.session_state.session_id

    if not uploaded_files or not session_id:
        return

    current_files = set(get_current_pdf_names())

    stored_signatures = st.session_state.file_signatures.get(session_id, [])

    
    if isinstance(stored_signatures, str):
        stored_signatures = [stored_signatures]

    stored_signatures = set(stored_signatures)

    new_files = []

    for file in uploaded_files:
        signature = get_file_signature(file)

        # Avoid repeated ingestion on Streamlit reruns
        if signature in stored_signatures:
            continue

        # Avoid duplicate ingestion after app restart if file name already exists in Chroma
        if file.name in current_files:
            continue

        new_files.append(file)

    if not new_files:
        return

    with st.spinner("Processing uploaded PDFs automatically..."):
        chunks, errors = st.session_state.document_service.process_pdfs(
            new_files
        )

        for error in errors:
            st.warning(error)

        if not chunks:
            st.error("No valid text chunks were created from the uploaded PDFs.")
            return

        count = st.session_state.vector_service.add_documents(
            chunks,
            session_id,
        )

        updated_files = set(get_current_pdf_names())
        updated_signatures = set(st.session_state.file_signatures.get(session_id, []))

        for file in new_files:
            updated_files.add(file.name)
            updated_signatures.add(get_file_signature(file))

        st.session_state.session_files[session_id] = sorted(updated_files)
        st.session_state.file_signatures[session_id] = list(updated_signatures)

        # Rebuild chain because new documents were added
        st.session_state.chain_session = None

        st.success(
            f"Ingested {len(new_files)} new PDF(s) with {count} chunks."
        )


def build_rag_chain():
    """Create RAG chain for current session."""

    session_id = st.session_state.session_id

    if st.session_state.chain_session == session_id:
        return True

    try:
        st.session_state.rag_service.initialize_llm(
            st.session_state.api_key
        )

        retriever = st.session_state.vector_service.get_retriever(
            session_id
        )

        st.session_state.rag_service.create_chain(retriever)

        st.session_state.chain_session = session_id

        return True

    except Exception as e:
        st.error(f"Error building RAG chain: {e}")
        return False


# -----------------------------
# Start App
# -----------------------------
init_app()

st.title("📚 Conversational RAG PDF Q&A")
st.caption("Upload multiple PDFs per session and ask questions about them.")


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("Settings")

    api_key = st.text_input(
        "Groq API Key",
        type="password",
    )

    if api_key:
        st.session_state.api_key = api_key

    session_id = st.text_input(
        "Session ID",
        value=st.session_state.session_id,
        placeholder="Example: 1",
    ).strip()

    if session_id != st.session_state.session_id:
        st.session_state.chain_session = None

    st.session_state.session_id = session_id

    st.divider()

    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        key=get_uploader_key(),
    )

    auto_ingest_pdfs(uploaded_files)

    current_files = get_current_pdf_names()

    if current_files:
        st.success(f"{len(current_files)} PDF(s) in this session:")
        for file_name in current_files:
            st.write(f"• {file_name}")
    else:
        st.info("No PDFs uploaded for this session.")

    st.divider()

    if st.button("Clear Chat History", use_container_width=True):
        if session_id:
            st.session_state.memory_service.clear_history(session_id)
            st.rerun()

    if st.button("Clear Session PDFs", use_container_width=True):
        if session_id:
            st.session_state.vector_service.clear_session(session_id)
            st.session_state.session_files.pop(session_id, None)
            st.session_state.file_signatures.pop(session_id, None)
            st.session_state.chain_session = None

            reset_uploader()

            st.rerun()


# -----------------------------
# Validation
# -----------------------------
if not st.session_state.api_key:
    st.warning("Enter your Groq API key in the sidebar.")
    st.stop()

if not st.session_state.session_id:
    st.info("Enter a Session ID in the sidebar.")
    st.stop()


# -----------------------------
# Status
# -----------------------------
chunk_count = st.session_state.vector_service.count_documents(
    st.session_state.session_id
)

current_files = get_current_pdf_names()

col1, col2, col3 = st.columns(3)

col1.metric(
    "Session ID",
    st.session_state.session_id,
)

col2.metric(
    "Stored Chunks",
    chunk_count,
)

col3.metric(
    "PDFs",
    len(current_files),
)

st.divider()

# -----------------------------
# Chat History
# -----------------------------
st.subheader("💬 Chat with Your PDFs")

if current_files:
    with st.expander("PDFs in this session"):
        for file_name in current_files:
            st.write(f"• {file_name}")

messages = st.session_state.memory_service.get_messages(
    st.session_state.session_id
)

if not messages:
    st.caption("No chat history yet.")

for message in messages:
    role = "user" if message.type == "human" else "assistant"

    with st.chat_message(role):
        st.markdown(message.content)


# -----------------------------
# Question Input
# -----------------------------
question = st.chat_input("Ask a question about your PDFs...")

if question:
    if chunk_count == 0:
        st.warning("Upload at least one PDF before asking questions.")
        st.stop()

    if not build_rag_chain():
        st.stop()

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = st.session_state.rag_service.ask(
                question,
                st.session_state.session_id,
            )

        st.markdown(result["answer"])

        with st.expander("Retrieved Sources"):
            for index, doc in enumerate(
                result.get("context", []),
                start=1,
            ):
                source = doc.metadata.get("source_file", "Unknown file")
                page = doc.metadata.get("page_number", "N/A")

                st.markdown(
                    f"**Source {index}: {source} — Page {page}**"
                )
                st.write(doc.page_content[:500])
                st.divider()