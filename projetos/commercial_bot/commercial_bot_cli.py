import argparse
from rag import build_rag

try:
    from .multi_agent import run_collab  # pacote
except Exception:
    from multi_agent import run_collab  # fallback


def main():
    parser = argparse.ArgumentParser(description="Bot comercial para catálogo de produtos")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subcomando: Q&A RAG simples
    rag_parser = subparsers.add_parser("qa", help="Pergunta-resposta baseada no catálogo (RAG)")
    rag_parser.add_argument("--question", "-q", required=True, help="Pergunta sobre o catálogo")
    rag_parser.add_argument("--doc", default="ingestion/products_catalog.md", help="Caminho do catálogo .md")

    # Subcomando: colaboração multi-agentes (consultivo + produto + prospecção)
    collab_parser = subparsers.add_parser("collab", help="Fluxo de colaboração entre agentes comerciais")
    collab_parser.add_argument("--cliente", required=True, help="Identificação ou perfil do cliente")
    collab_parser.add_argument(
        "--contexto",
        required=True,
        help="Contexto do cliente (dores, metas, cenário de uso)",
    )
    collab_parser.add_argument("--doc", default="ingestion/products_catalog.md", help="Caminho do catálogo .md")

    args = parser.parse_args()

    if args.command == "qa":
        retriever, chain, _ = build_rag(args.doc)
        answer = chain.invoke(args.question)
        print(answer)
    elif args.command == "collab":
        final = run_collab(cliente=args.cliente, contexto=args.contexto, doc_path=args.doc)
        # Imprime saída agregada
        saida = final.get("saida_final") or final
        print(saida)


if __name__ == "__main__":
    main()
