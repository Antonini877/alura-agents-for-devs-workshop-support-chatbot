# syntax=docker/dockerfile:1
FROM python:3.11-slim

WORKDIR /app

# Instala dependências de sistema mínimas
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copia projeto
COPY requirements.txt ./
COPY ingestion ./ingestion
COPY projetos ./projetos

# Instala dependências
RUN pip install --no-cache-dir -r requirements.txt -r projetos/commercial_bot/requirements.txt

# Porta da API
EXPOSE 8000

# Variáveis esperadas (definidas em runtime)
ENV CATALOG_MD_PATH=ingestion/products_catalog.md \
    RETRIEVER_K=5

# Comando para iniciar a API FastAPI
CMD ["python", "-m", "uvicorn", "projetos.commercial_bot.api:app", "--host", "0.0.0.0", "--port", "8000"]

