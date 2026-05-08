from typing import List, Optional

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

from config import AppConfig


class VectorStoreService:
    def __init__(self, config: AppConfig):
        self.config = config
        self._vector_stores = {}

        self.embeddings = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL,
            encode_kwargs={"normalize_embeddings": True},
        )

    def _get_or_create_vector_store(self, session_id: str) -> Chroma:
        if session_id not in self._vector_stores:
            self._vector_stores[session_id] = Chroma(
                collection_name=self.config.get_collection_name(session_id),
                embedding_function=self.embeddings,
                persist_directory=str(self.config.CHROMA_PERSIST_DIR),
            )

        return self._vector_stores[session_id]

    def add_documents(self, chunks: List[Document], session_id: str) -> int:
        if not chunks:
            return 0

        vector_store = self._get_or_create_vector_store(session_id)
        return len(vector_store.add_documents(chunks))

    def get_source_files(self, session_id: str) -> list:
        try:
            vector_store = self._get_or_create_vector_store(session_id)
            metadatas = vector_store._collection.get(
                include=["metadatas"]
            ).get("metadatas", [])

            return sorted({
                metadata["source_file"]
                for metadata in metadatas
                if metadata and metadata.get("source_file")
            })

        except Exception:
            return []

    def _doc_key(self, doc: Document) -> tuple:
        return (
            doc.metadata.get("source_file", ""),
            doc.metadata.get("page_number", ""),
            doc.metadata.get("chunk_id", ""),
            doc.page_content[:100],
        )

    def _add_unique_docs(
        self,
        docs: list,
        seen: set,
        new_docs: List[Document],
        limit: Optional[int] = None,
    ):
        for doc in new_docs:
            key = self._doc_key(doc)

            if key not in seen:
                docs.append(doc)
                seen.add(key)

            if limit and len(docs) >= limit:
                break

    def retrieve_balanced(
        self,
        query: str,
        session_id: str,
        top_k: Optional[int] = None,
    ) -> List[Document]:

        if top_k is None:
            top_k = self.config.TOP_K_RESULTS

        vector_store = self._get_or_create_vector_store(session_id)
        source_files = self.get_source_files(session_id)

        if not source_files:
            return vector_store.max_marginal_relevance_search(
                query,
                k=top_k,
                fetch_k=25,
                lambda_mult=0.5,
            )

        docs = []
        seen = set()
        per_file_k = max(1, top_k // len(source_files))

        for source_file in source_files:
            try:
                file_docs = vector_store.similarity_search(
                    query,
                    k=per_file_k,
                    filter={"source_file": source_file},
                )
                self._add_unique_docs(docs, seen, file_docs)

            except Exception:
                continue

        if len(docs) < top_k:
            try:
                extra_docs = vector_store.max_marginal_relevance_search(
                    query,
                    k=top_k,
                    fetch_k=25,
                    lambda_mult=0.5,
                )
                self._add_unique_docs(docs, seen, extra_docs, top_k)

            except Exception:
                pass

        return docs[:top_k]

    def get_retriever(
        self,
        session_id: str,
        top_k: Optional[int] = None,
    ):
        if top_k is None:
            top_k = self.config.TOP_K_RESULTS

        return RunnableLambda(
            lambda query: self.retrieve_balanced(
                query=str(query),
                session_id=session_id,
                top_k=top_k,
            )
        )

    def get_vector_store(self, session_id: str) -> Chroma:
        return self._get_or_create_vector_store(session_id)

    def count_documents(self, session_id: str) -> int:
        try:
            return self._get_or_create_vector_store(
                session_id
            )._collection.count()

        except Exception:
            return 0

    def collection_exists(self, session_id: str) -> bool:
        return self.count_documents(session_id) > 0

    def clear_session(self, session_id: str):
        try:
            self._get_or_create_vector_store(session_id).delete_collection()
            self._vector_stores.pop(session_id, None)

        except Exception as e:
            print(f"Error clearing session {session_id}: {e}")