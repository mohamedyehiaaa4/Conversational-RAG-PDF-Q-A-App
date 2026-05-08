
import os
import tempfile
from pathlib import Path
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from config import AppConfig

class DocumentService:
    """
    Handles PDF document processing including validation, loading, and chunking.
    """
    ALLOWED_EXTENSIONS = {".pdf"}

    def __init__(self, config: AppConfig):
        self.config = config
        self.temp_dir = config.TEMP_PDF_DIR

        # Initialize text splitter
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", " ", ""],
        )

    def validate_pdf(self, file_path: Path) -> tuple[bool, str]:

        # Check file extension
        if file_path.suffix.lower() not in self.ALLOWED_EXTENSIONS:
            return False, "Invalid file type. Only PDF files are allowed."

        # Check file size
        file_size = file_path.stat().st_size
        max_size = self.config.get_max_pdf_size_bytes()
        if file_size > max_size:
            size_mb = self.config.MAX_PDF_SIZE_MB
            return False, f"File size exceeds {size_mb}MB limit."

        return True, ""

    def save_upload_file(self, uploaded_file) -> Path:

        temp_file_path = self.temp_dir / uploaded_file.name

        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        return temp_file_path

    def load_pdf(self, file_path: Path) -> List[Document]:

        loader = PyPDFLoader(str(file_path))
        documents = loader.load()
        return documents

    def add_metadata(
        self, documents: List[Document], source_name: str
    ) -> List[Document]:

        for doc in documents:
            doc.metadata["source_file"] = source_name
            # Page number is already in metadata from PyPDFLoader
        return documents

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
  
        chunks = self.splitter.split_documents(documents)
        return chunks

    def process_pdfs(self, uploaded_files: List) -> tuple[List[Document], List[str]]:
     
        all_chunks = []
        error_messages = []
        file_names = []

        for uploaded_file in uploaded_files:
            try:
                # Save the uploaded file
                file_path = self.save_upload_file(uploaded_file)
                # Validate the PDF
                is_valid, error_msg = self.validate_pdf(file_path)
                if not is_valid:
                    error_messages.append(f"{uploaded_file.name}: {error_msg}")
                    file_path.unlink()  # Delete invalid file
                    continue
                # Load the PDF
                documents = self.load_pdf(file_path)
                # Add metadata
                documents = self.add_metadata(documents, uploaded_file.name)
                # Chunk the documents
                chunks = self.chunk_documents(documents)

                all_chunks.extend(chunks)
                file_names.append(uploaded_file.name)

            except Exception as e:
                error_messages.append(f"{uploaded_file.name}: {str(e)}")
                continue

        return all_chunks, error_messages

    def cleanup_temp_files(self):
        """Remove temporary PDF files after processing."""
        try:
            for file in self.temp_dir.glob("*.pdf"):
                file.unlink()
        except Exception as e:
            print(f"Error cleaning up temp files: {e}")
