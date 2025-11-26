import os
import sys
import numpy as np
from rouge_score import rouge_scorer
from rag import build_rag


def cosine(a, b):
    a = np.array(a)
    b = np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def eval_dataset():
    return [
        {"question": "Qual é o preço do Notebook Atlas X15?", "ground_truth": "R$ 6.499"},
        {"question": "O Fone Orpheus Pro possui cancelamento ativo de ruído?", "ground_truth": "Sim"},
        {"question": "Qual material do Cabo USB-C Titan Shield?", "ground_truth": "Nylon trançado"},
    ]


def run_evaluation(doc_path: str):
    retriever, chain, embeddings = build_rag(doc_path, collection_name="commercial-catalog-eval")
    data = eval_dataset()
    predictions = []
    contexts = []
    for item in data:
        q = item["question"]
        docs_ctx = retriever.invoke(q)
        contexts.append([d.page_content for d in docs_ctx])
        ans = chain.invoke(q)
        predictions.append(ans)

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    metrics = []
    for i, item in enumerate(data):
        gt = item["ground_truth"]
        pred = predictions[i]
        q = item["question"]
        ctxs = contexts[i]
        rouge = scorer.score(gt, pred)
        correctness = rouge["rougeL"].fmeasure
        emb_q = embeddings.embed_query(q)
        emb_a = embeddings.embed_query(pred)
        relevancy = cosine(emb_q, emb_a)
        sims = [cosine(emb_a, embeddings.embed_query(c)) for c in ctxs] if ctxs else []
        faithfulness = float(np.mean(sims)) if sims else 0.0
        metrics.append({
            "correctness": correctness,
            "relevancy": relevancy,
            "faithfulness": faithfulness,
        })

    avg = {
        "correctness": float(np.mean([m["correctness"] for m in metrics])) if metrics else 0.0,
        "relevancy": float(np.mean([m["relevancy"] for m in metrics])) if metrics else 0.0,
        "faithfulness": float(np.mean([m["faithfulness"] for m in metrics])) if metrics else 0.0,
    }
    return metrics, avg, predictions


def main():
    doc_path = os.environ.get("CATALOG_MD_PATH", "ingestion/products_catalog.md")
    metrics, avg, predictions = run_evaluation(doc_path)
    print("Métricas por item:", metrics)
    print("Médias:", {k: round(v, 6) for k, v in avg.items()})
    thresholds = {
        "correctness": float(os.environ.get("THRESHOLD_CORRECTNESS", "0.45")),
        "relevancy": float(os.environ.get("THRESHOLD_RELEVANCY", "0.6")),
        "faithfulness": float(os.environ.get("THRESHOLD_FAITHFULNESS", "0.5")),
    }
    epsilon = float(os.environ.get("THRESHOLD_EPSILON", "1e-9"))
    checks = {k: {"avg": avg[k], "threshold": thresholds[k], "pass": (avg[k] + epsilon) >= thresholds[k]} for k in thresholds}
    ok = all(info["pass"] for info in checks.values())
    print("Verificação por métrica:", {k: {"avg": round(v["avg"], 6), "threshold": v["threshold"], "pass": v["pass"]} for k, v in checks.items()})
    if not ok:
        print("Falha: métricas abaixo dos thresholds:", thresholds)
        sys.exit(1)
    print("Sucesso: métricas atingiram os thresholds:", thresholds)


if __name__ == "__main__":
    main()