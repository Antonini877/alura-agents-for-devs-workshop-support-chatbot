import os
from typing import TypedDict, Optional

from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

# Integração com RAG existente para informações de produto
try:
    from .rag import build_rag  # quando importado como pacote
except Exception:
    from rag import build_rag  # fallback para execução direta


class SalesState(TypedDict):
    cliente: str
    contexto: str
    plano: str
    diagnostico: str
    produto_info: str
    texto_prospeccao: str
    saida_final: str


def _build_llm():
    load_dotenv()
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY/GEMINI_API_KEY ausente. Configure no ambiente/.env.")
    model_name = os.environ.get("MODEL_NAME", "gemini-2.0-flash")
    return ChatGoogleGenerativeAI(model=model_name, temperature=0, google_api_key=api_key)


# Prompts dos agentes
prompt_coord = ChatPromptTemplate.from_messages([
    (
        "system",
        "Você é coordenador comercial. Crie um plano de abordagem consultiva conciso para o cliente." \
        " Considere o contexto do negócio e defina passos: Diagnóstico → Informações do Produto → Texto de Prospecção."
    ),
    (
        "human",
        "Cliente: {cliente}\nContexto: {contexto}\n\nRetorne apenas o plano (bullet points)."
    ),
])

prompt_consultivo = ChatPromptTemplate.from_messages([
    (
        "system",
        "Você realiza atendimento consultivo. Produza um diagnóstico objetivo (necessidades, dores, critérios)."
    ),
    (
        "human",
        "Plano: {plano}\nCliente: {cliente}\nContexto: {contexto}\n\nRetorne um diagnóstico com 4-6 pontos."
    ),
])

prompt_prospeccao = ChatPromptTemplate.from_messages([
    (
        "system",
        "Você escreve textos de prospecção comerciais claros e persuasivos, mantendo tom consultivo."
    ),
    (
        "human",
        "Cliente: {cliente}\nDiagnóstico:\n{diagnostico}\n\nInformações de Produto:\n{produto_info}\n\nEscreva um e-mail curto (assunto + corpo em 2-3 parágrafos)."
    ),
])

prompt_agregador = ChatPromptTemplate.from_messages([
    (
        "system",
        "Você é agregador. Monte a saída final contendo: Resumo do Plano, Diagnóstico e Texto de Prospecção."
    ),
    (
        "human",
        "Plano:\n{plano}\n\nDiagnóstico:\n{diagnostico}\n\nTexto de Prospecção:\n{texto_prospeccao}"
    ),
])


def build_sales_graph(doc_path: Optional[str] = None):
    """Compila o grafo de colaboração dos agentes comerciais.

    doc_path: caminho do catálogo .md para RAG (default: env CATALOG_MD_PATH ou ingestion/products_catalog.md)
    """
    llm = _build_llm()
    parser = StrOutputParser()

    # RAG para o especialista de produto
    doc_md = doc_path or os.environ.get("CATALOG_MD_PATH", "ingestion/products_catalog.md")
    retriever, rag_chain, _emb = build_rag(doc_md)

    def coordenador(state: SalesState):
        plano = (prompt_coord | llm | parser).invoke({
            "cliente": state["cliente"],
            "contexto": state["contexto"],
        })
        return {"plano": plano}

    def consultivo(state: SalesState):
        diag = (prompt_consultivo | llm | parser).invoke({
            "plano": state["plano"],
            "cliente": state["cliente"],
            "contexto": state["contexto"],
        })
        return {"diagnostico": diag}

    def especialista_produto(state: SalesState):
        # Usa RAG existente: pergunta baseada no diagnóstico para buscar contexto e resposta
        pergunta = f"Quais pontos do produto são mais relevantes? Contexto do cliente: {state['diagnostico']}"
        # Obtém documentos de contexto para robustez
        docs_ctx = retriever.invoke(pergunta)
        ctx_text = "\n\n".join(d.page_content for d in docs_ctx)
        ans = rag_chain.invoke(pergunta)
        # Compatibilidade com o formato de retorno do chain (str ou dict)
        produto_info = ans.get("answer") if isinstance(ans, dict) else str(ans)
        # Inclui fatias de contexto para enriquecer o escritor
        produto_info = produto_info + (f"\n\nContextos:\n{ctx_text}" if ctx_text else "")
        return {"produto_info": produto_info}

    def escritor_prospeccao(state: SalesState):
        texto = (prompt_prospeccao | llm | parser).invoke({
            "cliente": state["cliente"],
            "diagnostico": state["diagnostico"],
            "produto_info": state["produto_info"],
        })
        return {"texto_prospeccao": texto}

    def agregador(state: SalesState):
        saida = (prompt_agregador | llm | parser).invoke({
            "plano": state["plano"],
            "diagnostico": state["diagnostico"],
            "texto_prospeccao": state["texto_prospeccao"],
        })
        return {"saida_final": saida}

    g = StateGraph(SalesState)
    g.add_node("Coordenador", coordenador)
    g.add_node("Consultivo", consultivo)
    g.add_node("EspecialistaProduto", especialista_produto)
    g.add_node("Prospeccao", escritor_prospeccao)
    g.add_node("Agregador", agregador)

    g.add_edge("Coordenador", "Consultivo")
    g.add_edge("Consultivo", "EspecialistaProduto")
    g.add_edge("EspecialistaProduto", "Prospeccao")
    g.add_edge("Prospeccao", "Agregador")
    g.add_edge("Agregador", END)
    g.set_entry_point("Coordenador")
    app = g.compile()
    return app


def run_collab(cliente: str, contexto: str, doc_path: Optional[str] = None):
    """Executa a colaboração dos agentes e retorna o estado final."""
    app = build_sales_graph(doc_path)
    result = app.invoke({"cliente": cliente, "contexto": contexto})
    return result

