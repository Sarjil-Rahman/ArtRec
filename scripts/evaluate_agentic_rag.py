from __future__ import annotations

import argparse
import json
from pathlib import Path

from artrec.agentic.agent import AgenticRAGPipeline
from artrec.agentic.evaluation import evaluate_pipeline, load_gold
from artrec.agentic.ingestion import load_sample_documents


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the local agentic RAG pipeline.")
    parser.add_argument("--documents", default="data/documents/sample")
    parser.add_argument("--gold", default="data/eval/rag_gold.jsonl")
    parser.add_argument("--output", default="artifacts/agentic_eval")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    pipeline = AgenticRAGPipeline(load_sample_documents(args.documents))
    metrics, predictions, failures = evaluate_pipeline(
        pipeline=pipeline,
        gold_records=load_gold(args.gold),
        top_k=args.top_k,
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8"
    )
    write_jsonl(output_dir / "predictions.jsonl", predictions)
    write_jsonl(output_dir / "failure_cases.jsonl", failures)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    print(f"Wrote evaluation artifacts to {output_dir}")


if __name__ == "__main__":
    main()
