import argparse
from rag import build_rag


def main():
    parser = argparse.ArgumentParser(description="Bot comercial para catálogo de produtos")
    parser.add_argument("--question", "-q", required=True, help="Pergunta sobre o catálogo")
    parser.add_argument("--doc", default="ingestion/products_catalog.md", help="Caminho do catálogo .md")
    args = parser.parse_args()

    retriever, chain, _ = build_rag(args.doc)
    answer = chain.invoke(args.question)
    print(answer)


if __name__ == "__main__":
    main()