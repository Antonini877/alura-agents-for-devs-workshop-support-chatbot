from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory

try:
    from config import MEMORY_DB_URL
except Exception:
    MEMORY_DB_URL = "sqlite:///agent_memory.db"


def get_sql_history(session_id: str) -> SQLChatMessageHistory:
    """Retorna um histórico de mensagens baseado em SQLite para a sessão fornecida."""
    return SQLChatMessageHistory(
        session_id=session_id, connection_string=MEMORY_DB_URL
    )


def wrap_with_sql_history(runnable):
    """Envelopa um runnable com memória por sessão usando SQLChatMessageHistory."""
    return RunnableWithMessageHistory(
        runnable,
        get_sql_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )