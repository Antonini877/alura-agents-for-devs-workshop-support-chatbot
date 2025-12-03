import os
import json
from typing import Optional

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain.tools import tool
from pydantic import BaseModel, Field


def build_rag(doc_path: str, collection_name: str = "commercial-catalog"):
    # Carrega variáveis de ambiente e obtém a API key do Gemini
    load_dotenv()
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY/GEMINI_API_KEY não configurado. Defina a variável de ambiente ou um .env."
        )

    with open(doc_path, "r", encoding="utf-8") as f:
        raw_text = f.read()
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    docs = splitter.create_documents([raw_text])
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=api_key)
    vs = Chroma.from_documents(docs, embedding=embeddings, collection_name=collection_name)
    k = int(os.environ.get("RETRIEVER_K", "4"))
    retriever = vs.as_retriever(k=k)

    def format_docs(docs):
        return "\n\n".join(d.page_content for d in docs)

    # Tool: envio de e-mail via .env
    class SendEmailInput(BaseModel):
        to: str = Field(..., description="E-mail do destinatário")
        subject: Optional[str] = Field(None, description="Assunto do e-mail")
        message_text: str = Field(..., description="Corpo do e-mail em texto")

    # Import robusto para funcionar tanto como pacote quanto script local
    try:
        from .email_tool import send_email as _send_email
    except Exception:
        from email_tool import send_email as _send_email  # type: ignore

    @tool("send_email", args_schema=SendEmailInput)
    def send_email_tool(to: str, subject: Optional[str], message_text: str) -> str:
        """Envia um e-mail. Use quando o usuário pedir explicitamente envio por e-mail e fornecer um destinatário."""
        res = _send_email(to=to, subject=subject or "Resposta do catálogo", message_text=message_text)
        return json.dumps(res)

    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0, google_api_key=api_key)
    llm_with_tools = llm.bind_tools([send_email_tool])

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
Você é um assistente de catálogo. Regras:
- Responda exclusivamente com base nos Contextos.
- Não invente informações: se faltar, diga "Não encontrado no catálogo".
- Para perguntas sim/não, responda apenas "Sim" ou "Não".
- Quando o usuário pedir para enviar por e-mail e fornecer um destinatário (endereço de e-mail presente), chame a ferramenta send_email.
- O campo message_text da ferramenta deve ser exatamente a resposta que você daria ao usuário.
"""
        ),
        (
            "human",
            "Contextos:\n{context}\n\nPergunta:\n{question}"
        ),
    ])

    def route_tools(ai_msg):
        # Se o modelo chamou a ferramenta, execute-a e retorne dict com resposta e resultado
        tool_calls = getattr(ai_msg, "tool_calls", []) or []
        if tool_calls:
            # Assume um único tool_call relevante
            tc = tool_calls[0]
            args = tc.get("args", {})
            to = args.get("to")
            subject = args.get("subject")
            msg_text = args.get("message_text") or ""
            try:
                res = _send_email(to=to, subject=subject, message_text=msg_text)
            except Exception as e:
                res = {"status": "error", "error": str(e)}
            return {"answer": msg_text, "email": res}
        # Sem ferramenta: retorna conteúdo textual
        return {"answer": getattr(ai_msg, "content", str(ai_msg)), "email": None}

    chain = (
        {"context": retriever | RunnableLambda(format_docs), "question": RunnableLambda(lambda x: x)}
        | prompt
        | llm_with_tools
        | RunnableLambda(route_tools)
    )
    return retriever, chain, embeddings
