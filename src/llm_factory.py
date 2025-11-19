import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI


def build_llm() -> ChatGoogleGenerativeAI:
    """Fábrica de LLM centralizada (SRP), com streaming desativado."""
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    model_name = os.getenv("MODEL_NAME", "gemini-2.0-flash")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY não encontrado. Defina no .env ou ambiente.")
    return ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key, streaming=False)