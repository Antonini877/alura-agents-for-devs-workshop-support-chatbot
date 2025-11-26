import os
from dotenv import load_dotenv


def load_env() -> None:
    """Carrega variáveis de ambiente do .env (se existir)."""
    load_dotenv()


# Configurações padrão, podem ser sobrescritas via .env
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "gemini-2.0-flash")
MEMORY_DB_URL = os.getenv("MEMORY_DB_URL", "sqlite:///agent_memory.db")


__all__ = [
    "load_env",
    "GOOGLE_API_KEY",
    "MODEL_NAME",
    "MEMORY_DB_URL",
]