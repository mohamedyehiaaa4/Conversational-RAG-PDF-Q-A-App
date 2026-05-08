"""
Memory Service module for managing chat history per session.
"""

from langchain_community.chat_message_histories import ChatMessageHistory


class MemoryService:
    """
    Manages chat history for multiple Session IDs.
    """

    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.histories = {}

    def get_history(self, session_id: str) -> ChatMessageHistory:
        """
        Get or create chat history for a session.
        """
        if session_id not in self.histories:
            self.histories[session_id] = ChatMessageHistory()

        return self.histories[session_id]

    def get_messages(self, session_id: str):
        """
        Return messages for a session.
        """
        return self.get_history(session_id).messages

    def clear_history(self, session_id: str):
        """
        Clear chat history for a session.
        """
        self.get_history(session_id).clear()