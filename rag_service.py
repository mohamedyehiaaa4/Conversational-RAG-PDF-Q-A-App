
from typing import Dict, Any
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.retrievers import BaseRetriever
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from config import AppConfig
from memory_service import MemoryService


class RAGService:
    """
    Builds and runs the conversational RAG pipeline.
    """

    def __init__(self, config: AppConfig, memory_service: MemoryService):
        self.config = config
        self.memory_service = memory_service
        self.llm = None
        self.chain = None

    def initialize_llm(self, api_key: str) -> None:
        """
        Initialize ChatGroq using runtime API key.
        """
        if not api_key:
            raise ValueError("Groq API key is required.")

        self.llm = ChatGroq(
            model=self.config.GROQ_MODEL,
            groq_api_key=api_key,
            temperature=0,
            max_tokens=1024,
        )

    def _create_contextualize_prompt(self) -> ChatPromptTemplate:
        """
        Prompt for rewriting follow-up questions.
        """
        contextualize_q_system_prompt = (
            "Given the chat history and the latest user question, "
            "which might reference previous context, rewrite the question "
            "as a standalone question. Do NOT answer the question. "
            "If the question is already standalone, return it unchanged."
        )

        return ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

    def _create_qa_prompt(self) -> ChatPromptTemplate:
        """
        Prompt for answering from retrieved PDF context only.
        """
        qa_system_prompt = (
        "You are a helpful assistant for question-answering over uploaded PDF documents. "
        "The retrieved context may come from one or multiple PDF files. "
        "Use only the retrieved context below to answer the user's question. "
        "If information from multiple PDFs is relevant, combine it clearly. "
        "When possible, mention which PDF file or page the information came from. "
        "Do not use outside knowledge. "
        "If the answer is not found in the context, respond exactly with: "
        "'I could not find this information in the uploaded documents.' "
        "Give a clear and student-friendly answer.\n\n"
        "Retrieved context:\n{context}"
            )
        return ChatPromptTemplate.from_messages(
            [
                ("system", qa_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

    def create_chain(self, retriever: BaseRetriever):
        """
        Create history-aware RAG chain.
        """
        if self.llm is None:
            raise ValueError("LLM not initialized. Call initialize_llm() first.")

        history_aware_retriever = create_history_aware_retriever(
            self.llm,
            retriever,
            self._create_contextualize_prompt(),
        )

        qa_chain = create_stuff_documents_chain(
            self.llm,
            self._create_qa_prompt(),
        )

        rag_chain = create_retrieval_chain(
            history_aware_retriever,
            qa_chain,
        )

        self.chain = RunnableWithMessageHistory(
            rag_chain,
            self.memory_service.get_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )

    def ask(self, question: str, session_id: str) -> Dict[str, Any]:
        """
        Ask a question and return answer plus retrieved context.
        """
        if self.chain is None:
            raise ValueError("Chain not initialized. Call create_chain() first.")

        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")

        try:
            result = self.chain.invoke(
                {"input": question},
                config={"configurable": {"session_id": session_id}},
            )

            return {
                "answer": result.get(
                    "answer",
                    "I could not find this information in the uploaded documents.",
                ),
                "context": result.get("context", []),
            }

        except Exception as e:
            raise RuntimeError(f"Error during RAG query: {str(e)}")