# Commercial Bot — Arquitetura de Agentes

Este documento apresenta o desenho de arquitetura dos agentes que compõem o fluxo de colaboração do `commercial_bot`.

## Visão Geral
- Coordenador: define o plano de abordagem consultiva (passos e prioridades).
- Consultivo: elabora diagnóstico objetivo das necessidades do cliente.
- Especialista de Produto: consulta o catálogo via RAG para trazer informações relevantes.
- Prospeccão (Escritor): cria o texto de prospecção (assunto + corpo).
- Agregador: compõe a saída final para entrega ao usuário.

## Diagrama de Arquitetura

```
                         +----------------------------+
                         |   CLI collab / API HTTP    |
                         +-------------+--------------+
                                       |
                                       v
  +------------------------+       usa LLM       +------+
  |      Coordenador       |-------------------->| LLM  |
  +-------------+----------+                     +------+
                |
                v
  +------------------------+       usa LLM       +------+
  |       Consultivo       |-------------------->| LLM  |
  +-------------+----------+                     +------+
                |
                v
  +------------------------+   consulta           +-------------+
  | Especialista de Produto|--------------------->|   RAG       |
  +-------------+----------+                      +------+------+
                |                                        |
                |                                        v
                |                               +---------------------+
                |                               | products_catalog.md |
                |                               +---------------------+
                v
  +------------------------+       usa LLM       +------+
  |  Prospeccão (Escritor) |-------------------->| LLM  |
  +-------------+----------+                     +------+
                |
                v
  +------------------------+       usa LLM       +------+
  |       Agregador        |-------------------->| LLM  |
  +-------------+----------+                     +------+
                |
                v
  +---------------------------------------------------------------+
  | Saída Final: Plano + Diagnóstico + E-mail (assunto + corpo)   |
  +---------------------------------------------------------------+
```

## Componentes
- `multi_agent.py`: implementação do grafo com os agentes e agregação de resultados.
- `rag.py`: construção do RAG (`build_rag`) e retriever sobre o catálogo.
- `commercial_bot_cli.py`: CLI com dois subcomandos — `qa` (RAG simples) e `collab` (multi-agentes).
- `ingestion/products_catalog.md`: fonte de conhecimento do catálogo de produtos.

## Fluxo de Execução
- Entrada: `cliente` (identificação/perfil) e `contexto` (dores, metas, cenário).
- Coordenador cria o plano em tópicos.
- Consultivo produz diagnóstico (4–6 pontos).
- Especialista de Produto consulta o catálogo via RAG e retorna pontos relevantes.
- Prospeccão escreve o e-mail curto (assunto + 2–3 parágrafos) com base no diagnóstico e produto.
- Agregador monta a saída final contendo plano, diagnóstico e e-mail.

## Configuração
- Variáveis de ambiente: defina a chave de API (`GOOGLE_API_KEY`) no ambiente ou `.env`.
- Opcional: caminho do catálogo via `--doc` na CLI ou variável de ambiente (`CATALOG_MD_PATH`).

## Execução Rápida (CLI)
- Colaboração multi-agentes:
  - `python projetos/commercial_bot/commercial_bot_cli.py collab --cliente "Loja XYZ" --contexto "Quer aumentar conversões no e-commerce de eletrônicos" --doc ingestion/products_catalog.md`
- Pergunta direta ao catálogo (RAG):
  - `python projetos/commercial_bot/commercial_bot_cli.py qa -q "Quais diferenciais do produto A para e-commerce?" --doc ingestion/products_catalog.md`
