from __future__ import annotations

import json
from pathlib import Path
from typing import List

from .tag_extractor import TagExtractor


def save_tags_json(output_path: Path, tags) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(tags, f, ensure_ascii=False, indent=2)


def main(argv: List[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Extract tag candidates from CSV summaries (embeddings-only)")
    parser.add_argument(
        "--files", "-f",
        nargs="+",
        default=[
            "data/DevOps-All-QAs (Jira).csv",
            "data/Infra-All-QAs (Jira).csv",
        ],
        help="CSV file paths to process",
    )
    # Align option name/format with tag_assigner
    parser.add_argument(
        "--column", "-col",
        default="要約",
        help="Target column name to extract from CSV (same as tag_assigner)",
    )
    parser.add_argument("--min-support", "-m", type=int, default=2, help="Minimum frequency to include a token")
    parser.add_argument("--max-axes", "-x", type=int, default=10, help="Maximum number of semantic axes (clusters)")
    parser.add_argument("--top-k", "-k", type=int, default=100, help="Max tags per axis")
    parser.add_argument(
        "--output", "-o",
        default="data/tag_candidates.embeddings.json",
        help="Output JSON path",
    )
    parser.add_argument(
        "--embed-model", "-mname",
        default="paraphrase-multilingual-MiniLM-L12-v2",
        help="Enable embedding mode with given SentenceTransformer model (e.g., paraphrase-multilingual-MiniLM-L12-v2)",
    )
    parser.add_argument(
        "--embed-clusters", "-ec",
        type=int,
        default=None,
        help="Number of semantic clusters (axes). If omitted, a heuristic is used",
    )
    parser.add_argument(
        "--embed-limit", "-el",
        type=int,
        default=2000,
        help="Max number of tokens to embed for clustering",
    )

    args = parser.parse_args(argv)

    files = [Path(p) for p in args.files]
    extractor = TagExtractor(
        summary_column=args.column,
        min_support=args.min_support,
        max_axes=args.max_axes,
        top_k_per_axis=args.top_k,
        embed_model=args.embed_model,
        embed_clusters=args.embed_clusters,
        embed_limit=args.embed_limit,
    )
    tags = extractor.extract_from_files(files)
    save_tags_json(Path(args.output), tags)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
