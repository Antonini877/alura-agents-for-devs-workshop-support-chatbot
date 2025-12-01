import os
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, Query, Body
from fastapi.responses import JSONResponse, StreamingResponse

from .rag import build_rag


load_dotenv()

app = FastAPI(title="Commercial Catalog Bot API", version="1.0.0")

# Inicializa RAG no startup com catálogo padrão
CATALOG_MD_PATH = os.environ.get("CATALOG_MD_PATH", "ingestion/products_catalog.md")
retriever = None
chain = None
embeddings = None


def bootstrap(doc_path: str = CATALOG_MD_PATH):
    global retriever, chain, embeddings
    retriever, chain, embeddings = build_rag(doc_path)


@app.on_event("startup")
def on_startup():
    bootstrap(CATALOG_MD_PATH)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat")
def chat(
    question: str = Body(..., embed=True, description="Pergunta sobre o catálogo"),
    doc: Optional[str] = Body(None, embed=True, description="Caminho opcional para catálogo .md"),
):
    if doc:
        # Recria RAG para o catálogo informado
        bootstrap(doc)
    # Resposta direta e contextos usados
    docs_ctx = retriever.invoke(question)
    ctxs = [d.page_content for d in docs_ctx]
    answer = chain.invoke(question)
    return JSONResponse({"answer": answer, "contexts": ctxs})


@app.get("/chat/stream")
def chat_stream(q: str = Query(..., description="Pergunta"), doc: Optional[str] = Query(None)):
    if doc:
        bootstrap(doc)

    def event_generator():
        for chunk in chain.stream(q):
            # Server-Sent Events
            yield f"data: {chunk}\n\n"
        yield "event: done\ndata: [DONE]\n\n"

    headers = {
        "Cache-Control": "no-cache",
        "Content-Type": "text/event-stream",
        "Connection": "keep-alive",
    }
    return StreamingResponse(event_generator(), headers=headers)

