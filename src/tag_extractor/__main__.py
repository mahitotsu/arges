from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Pattern, Set, Optional
import unicodedata
import math


@dataclass
class ExtractionConfig:
    files: List[Path]
    summary_column: str = "要約"
    # Generic noise removal patterns only (no domain words)
    noise_patterns: List[Pattern[str]] = field(
        default_factory=lambda: [
            re.compile(r"#?\d{2,}")
        ]
    )
    # Tuning
    min_support: int = 2  # minimum frequency for a token to be considered
    max_axes: int = 10
    top_k_per_axis: int = 100
    # Embeddings mode (tokens -> embeddings -> clustering)
    embed_model: Optional[str] = "paraphrase-multilingual-MiniLM-L12-v2"  # default model
    embed_clusters: Optional[int] = None  # number of semantic clusters (axes)
    embed_limit: int = 2000  # max number of tokens to embed for clustering


def normalize_text(text: str, cfg: ExtractionConfig) -> str:
    s = text
    for rx in cfg.noise_patterns:
        s = rx.sub("", s)
    return s


def read_summaries(files: Iterable[Path], summary_col: str) -> Iterable[str]:
    for file in files:
        with file.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                s = row.get(summary_col) or ""
                if s:
                    yield s


def extract_japanese_tokens(s: str) -> List[str]:
    # Extract contiguous Japanese sequences length 2..20 (purely from data)
    return [m.group(0) for m in re.finditer(r"[\u3040-\u30FF\u4E00-\u9FFF]{2,20}", s)]


def find_ascii_tokens(s: str) -> Set[str]:
    tokens: Set[str] = set()
    for m in re.finditer(r"\b[A-Za-z][A-Za-z0-9.+_-]{1,}\b", s):
        tok = m.group(0)
        if not tok.isdigit():
            tokens.add(tok)
    return tokens


def canon_ascii(token: str) -> str:
    # Unicode normalize, lower-case, collapse separators to '-'
    t = unicodedata.normalize("NFKC", token)
    t = t.strip().lower()
    t = re.sub(r"[\s_\.]+", "-", t)
    t = re.sub(r"-+", "-", t)
    return t


def canon_jp(token: str) -> str:
    # Unicode normalize and collapse duplicate long sound marks
    t = unicodedata.normalize("NFKC", token).strip()
    t = re.sub("ー{2,}", "ー", t)
    return t


def dedup_tokens(tokens: Iterable[str], canon_fn, freq_map: Dict[str, int]) -> List[str]:
    # Map canonical -> best original (highest freq then shortest then lexicographically)
    best: Dict[str, str] = {}
    for tok in tokens:
        c = canon_fn(tok)
        cur = best.get(c)
        if cur is None:
            best[c] = tok
        else:
            # Choose the more frequent; tie-breaker: shorter; then lexicographic
            def key(x: str) -> tuple:
                return (freq_map.get(x, 0), -len(x), x)
            if key(tok) > key(cur):
                best[c] = tok
    # Return tokens sorted by their frequency (desc) then lexicographically
    items = sorted(best.values(), key=lambda x: (-freq_map.get(x, 0), x))
    return items


def extract_tags(cfg: ExtractionConfig) -> Dict[str, List[str]]:
    summaries = [normalize_text(s, cfg) for s in read_summaries(cfg.files, cfg.summary_column)]

    ascii_counter: Dict[str, int] = {}
    ja_counter: Dict[str, int] = {}
    for s in summaries:
        for t in find_ascii_tokens(s):
            ascii_counter[t] = ascii_counter.get(t, 0) + 1
        for t in extract_japanese_tokens(s):
            ja_counter[t] = ja_counter.get(t, 0) + 1

    # Filter by min_support
    ascii_counter = {k: v for k, v in ascii_counter.items() if v >= cfg.min_support}
    ja_counter = {k: v for k, v in ja_counter.items() if v >= cfg.min_support}
    # Embedding-based clustering path only (tokens -> embeddings -> semantic clusters)
    return extract_tags_with_embeddings(cfg, ascii_counter, ja_counter)


def extract_tags_with_embeddings(
    cfg: ExtractionConfig,
    ascii_counter: Dict[str, int],
    ja_counter: Dict[str, int],
) -> Dict[str, List[str]]:
    # Lazy imports to keep heavy deps optional
    import importlib
    try:
        st_mod = importlib.import_module("sentence_transformers")
        SentenceTransformer = getattr(st_mod, "SentenceTransformer")
    except Exception as e:
        raise RuntimeError(
            "Embeddings mode requires 'sentence-transformers'. Install extras or run: uv add sentence-transformers scikit-learn"
        ) from e
    try:
        cluster_mod = importlib.import_module("sklearn.cluster")
        KMeans = getattr(cluster_mod, "KMeans")
    except Exception as e:
        raise RuntimeError(
            "Embeddings mode requires 'scikit-learn'. Install with: uv add scikit-learn"
        ) from e

    # Combine frequency maps and prepare candidate tokens
    # Keep tokens already filtered by min_support in caller
    # Create canonicalization-aware dedup across JP and ASCII
    freq: Dict[str, int] = {}
    for k, v in ascii_counter.items():
        freq[k] = freq.get(k, 0) + v
    for k, v in ja_counter.items():
        freq[k] = freq.get(k, 0) + v

    if not freq:
        return {}

    # Sort tokens by frequency desc then lexicographically
    tokens_all = sorted(freq.keys(), key=lambda k: (-freq[k], k))

    # Deduplicate within language groups first, then recombine, preferring higher frequency
    ascii_dedup = dedup_tokens([t for t in tokens_all if t in ascii_counter], canon_ascii, ascii_counter)
    ja_dedup = dedup_tokens([t for t in tokens_all if t in ja_counter], canon_jp, ja_counter)
    # Merge keeping overall frequency order
    merged = ascii_dedup + ja_dedup
    merged = sorted(set(merged), key=lambda k: (-freq.get(k, 0), k))

    # Limit to a reasonable number for embedding
    limit = max(1, min(cfg.embed_limit, cfg.top_k_per_axis * max(1, cfg.max_axes)))
    candidates = merged[:limit]
    if len(candidates) == 0:
        return {}

    # Build embeddings
    model_name = cfg.embed_model or "paraphrase-multilingual-MiniLM-L12-v2"
    model = SentenceTransformer(model_name)
    embs = model.encode(candidates, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)

    # Determine number of clusters (semantic axes)
    if cfg.embed_clusters is None:
        # heuristic: sqrt of N/2, capped by max_axes and at least 2 (unless N==1)
        if len(candidates) == 1:
            k = 1
        else:
            k = int(max(2, min(cfg.max_axes, math.sqrt(len(candidates) / 2))))
    else:
        k = int(cfg.embed_clusters)
    k = max(1, min(k, len(candidates)))

    # Cluster
    if k == 1:
        labels = [0] * len(candidates)
    else:
        km = KMeans(n_clusters=k, n_init="auto", random_state=42)
        labels = km.fit_predict(embs)

    # Group tokens by cluster and sort within by frequency
    clusters: Dict[int, List[str]] = {}
    for tok, lab in zip(candidates, labels):
        clusters.setdefault(int(lab), []).append(tok)

    # Build axes with representative token as axis label
    result: Dict[str, List[str]] = {}
    for lab, toks in clusters.items():
        toks_sorted = sorted(toks, key=lambda k2: (-freq.get(k2, 0), k2))
        if not toks_sorted:
            continue
        rep = toks_sorted[0]
        axis_name = f"意味クラスタ: 代表『{rep}』"
        result[axis_name] = toks_sorted[: cfg.top_k_per_axis]
    return result


def save_tags_json(output_path: Path, tags: Dict[str, List[str]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(tags, f, ensure_ascii=False, indent=2)


def main(argv: List[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Extract tag candidates from Jira CSV summaries (embeddings-only)")
    parser.add_argument(
        "--files",
        nargs="+",
        default=[
            "data/DevOps-All-QAs (Jira).csv",
            "data/Infra-All-QAs (Jira).csv",
        ],
        help="CSV file paths to process",
    )
    parser.add_argument(
        "--summary-col",
        default="要約",
        help="Summary column name in CSV",
    )
    parser.add_argument("--min-support", type=int, default=2, help="Minimum frequency to include a token")
    parser.add_argument("--max-axes", type=int, default=10, help="Maximum number of semantic axes (clusters)")
    parser.add_argument("--top-k", type=int, default=100, help="Max tags per axis")
    parser.add_argument(
        "--output",
        default="data/tag_candidates.embeddings.json",
        help="Output JSON path",
    )
    parser.add_argument(
        "--embed-model",
        default="paraphrase-multilingual-MiniLM-L12-v2",
        help="Enable embedding mode with given SentenceTransformer model (e.g., paraphrase-multilingual-MiniLM-L12-v2)",
    )
    parser.add_argument(
        "--embed-clusters",
        type=int,
        default=None,
        help="Number of semantic clusters (axes). If omitted, a heuristic is used",
    )
    parser.add_argument(
        "--embed-limit",
        type=int,
        default=2000,
        help="Max number of tokens to embed for clustering",
    )

    args = parser.parse_args(argv)

    files = [Path(p) for p in args.files]
    cfg = ExtractionConfig(
        files=files,
        summary_column=args.summary_col,
        min_support=args.min_support,
        max_axes=args.max_axes,
        top_k_per_axis=args.top_k,
        embed_model=args.embed_model,
        embed_clusters=args.embed_clusters,
        embed_limit=args.embed_limit,
    )
    tags = extract_tags(cfg)
    save_tags_json(Path(args.output), tags)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
