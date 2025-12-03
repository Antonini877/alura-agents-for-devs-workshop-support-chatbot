import os
import time
import logging
from uuid import uuid4
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, Query, Body, Request
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.responses import Response

from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

from .rag import build_rag
from .email_tool import send_email


load_dotenv()

# Logger estruturado simples
class SimpleFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        base = {
            "level": record.levelname,
            "ts": int(time.time() * 1000),
            "msg": record.getMessage(),
            "module": record.module,
        }
        if hasattr(record, "extra") and isinstance(record.extra, dict):
            base.update(record.extra)
        return str(base)


logger = logging.getLogger("commercial_bot_api")
handler = logging.StreamHandler()
handler.setFormatter(SimpleFormatter())
logger.setLevel(logging.INFO)
logger.addHandler(handler)


app = FastAPI(title="Commercial Catalog Bot API", version="1.0.0")

# Prometheus metrics
REQUESTS_TOTAL = Counter(
    "commercial_requests_total", "Total de requisições", ["path", "method"]
)
RESPONSES_TOTAL = Counter(
    "commercial_responses_total", "Total de respostas", ["path", "method", "status_code"]
)
ERRORS_TOTAL = Counter(
    "commercial_errors_total", "Total de erros", ["path", "method"]
)
LATENCY = Histogram(
    "commercial_request_latency_seconds", "Latência da requisição (s)", ["path", "method"]
)
EMAILS_SENT_TOTAL = Counter("commercial_emails_sent_total", "E-mails enviados")
EMAILS_ERRORS_TOTAL = Counter("commercial_emails_errors_total", "Erros ao enviar e-mails")
EMAIL_SEND_LATENCY = Histogram("commercial_email_send_latency_seconds", "Latência de envio de e-mail")


# Inicializa RAG no startup com catálogo padrão
CATALOG_MD_PATH = os.environ.get("CATALOG_MD_PATH", "ingestion/products_catalog.md")
retriever = None
chain = None
embeddings = None


def bootstrap(doc_path: str = CATALOG_MD_PATH):
    global retriever, chain, embeddings
    retriever, chain, embeddings = build_rag(doc_path)
    logger.info(
        "rag_bootstrap",
        extra={"extra": {"doc_path": doc_path, "status": "initialized"}},
    )


@app.middleware("http")
async def observability_middleware(request: Request, call_next):
    start = time.perf_counter()
    path = request.url.path
    method = request.method
    request_id = request.headers.get("X-Request-ID") or str(uuid4())
    REQUESTS_TOTAL.labels(path=path, method=method).inc()
    try:
        response = await call_next(request)
        status = response.status_code
        duration = time.perf_counter() - start
        LATENCY.labels(path=path, method=method).observe(duration)
        RESPONSES_TOTAL.labels(path=path, method=method, status_code=str(status)).inc()
        response.headers["X-Request-ID"] = request_id
        logger.info(
            "request_done",
            extra={
                "extra": {
                    "path": path,
                    "method": method,
                    "status": status,
                    "latency_ms": round(duration * 1000, 2),
                    "request_id": request_id,
                }
            },
        )
        return response
    except Exception as e:
        duration = time.perf_counter() - start
        ERRORS_TOTAL.labels(path=path, method=method).inc()
        LATENCY.labels(path=path, method=method).observe(duration)
        logger.error(
            "request_error",
            extra={
                "extra": {
                    "path": path,
                    "method": method,
                    "error": str(e),
                    "latency_ms": round(duration * 1000, 2),
                    "request_id": request_id,
                }
            },
        )
        raise


@app.on_event("startup")
def on_startup():
    bootstrap(CATALOG_MD_PATH)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/chat")
def chat(
    request: Request,
    question: str = Body(..., embed=True, description="Pergunta sobre o catálogo"),
    doc: Optional[str] = Body(None, embed=True, description="Caminho opcional para catálogo .md"),
):
    if doc:
        # Recria RAG para o catálogo informado
        bootstrap(doc)
    docs_ctx = retriever.invoke(question)
    ctxs = [d.page_content for d in docs_ctx]
    res = chain.invoke(question)
    # Compatibilidade: chain pode retornar str ou dict
    if isinstance(res, dict):
        answer = res.get("answer")
        email_result = res.get("email")
    else:
        answer = res
        email_result = None
    # Atualiza métricas baseado no resultado retornado pela tool do agente
    if isinstance(email_result, dict):
        status = email_result.get("status")
        if status == "sent":
            EMAILS_SENT_TOTAL.inc()
        elif status == "error":
            EMAILS_ERRORS_TOTAL.inc()

    logger.info(
        "chat_answer",
        extra={
            "extra": {
                "path": str(request.url.path),
                "question_len": len(question),
                "contexts": len(ctxs),
                "answer_preview": str(answer)[:200] if answer is not None else None,
                "email": (email_result or {}).get("status") if email_result else None,
            }
        },
    )
    return JSONResponse({"answer": answer, "contexts": ctxs, "email": email_result})


@app.get("/chat/stream")
def chat_stream(request: Request, q: str = Query(..., description="Pergunta"), doc: Optional[str] = Query(None)):
    if doc:
        bootstrap(doc)
    request_id = str(uuid4())

    def event_generator():
        chunk_count = 0
        for chunk in chain.stream(q):
            chunk_count += 1
            yield f"data: {chunk}\n\n"
        logger.info(
            "chat_stream_done",
            extra={
                "extra": {
                    "path": str(request.url.path),
                    "question_len": len(q),
                    "chunks": chunk_count,
                    "request_id": request_id,
                }
            },
        )
        yield "event: done\ndata: [DONE]\n\n"

    headers = {
        "Cache-Control": "no-cache",
        "Content-Type": "text/event-stream",
        "Connection": "keep-alive",
        "X-Request-ID": request_id,
    }
    return StreamingResponse(event_generator(), headers=headers)


# Removido: endpoint dedicado de envio de e-mail; a lógica foi integrada ao /chat
