from __future__ import annotations

import csv
import math
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Pattern, Set


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


class TagExtractor:
    """Extract tag candidates from CSV summaries using tokenization + embeddings clustering."""

    def __init__(
        self,
        *,
        summary_column: str = "要約",
        noise_patterns: Optional[List[Pattern[str]]] = None,
        min_support: int = 2,
        max_axes: int = 10,
        top_k_per_axis: int = 100,
        embed_model: Optional[str] = "paraphrase-multilingual-MiniLM-L12-v2",
        embed_clusters: Optional[int] = None,
        embed_limit: int = 2000,
    ) -> None:
        self.summary_column = summary_column
        self.noise_patterns = noise_patterns or [re.compile(r"#?\d{2,}")]
        self.min_support = int(min_support)
        self.max_axes = int(max_axes)
        self.top_k_per_axis = int(top_k_per_axis)
        self.embed_model = embed_model
        self.embed_clusters = embed_clusters
        self.embed_limit = int(embed_limit)

    # -------------------- Public API --------------------
    def extract_from_files(self, files: List[Path]) -> Dict[str, List[str]]:
        """Run extraction on the given CSV files and return tag axes -> values."""
        cfg = ExtractionConfig(
            files=files,
            summary_column=self.summary_column,
            noise_patterns=self.noise_patterns,
            min_support=self.min_support,
            max_axes=self.max_axes,
            top_k_per_axis=self.top_k_per_axis,
            embed_model=self.embed_model,
            embed_clusters=self.embed_clusters,
            embed_limit=self.embed_limit,
        )
        return self._extract_tags(cfg)

    # -------------------- Internal helpers --------------------
    def _normalize_text(self, text: str, cfg: ExtractionConfig) -> str:
        s = text
        for rx in cfg.noise_patterns:
            s = rx.sub("", s)
        return s

    def _read_summaries(self, files: Iterable[Path], summary_col: str) -> Iterable[str]:
        for file in files:
            with file.open(newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    s = row.get(summary_col) or ""
                    if s:
                        yield s

    @staticmethod
    def _extract_japanese_tokens(s: str) -> List[str]:
        # Extract contiguous Japanese sequences length 2..20
        return [m.group(0) for m in re.finditer(r"[\u3040-\u30FF\u4E00-\u9FFF]{2,20}", s)]

    @staticmethod
    def _find_ascii_tokens(s: str) -> Set[str]:
        tokens: Set[str] = set()
        for m in re.finditer(r"\b[A-Za-z][A-Za-z0-9.+_-]{1,}\b", s):
            tok = m.group(0)
            if not tok.isdigit():
                tokens.add(tok)
        return tokens

    @staticmethod
    def _canon_ascii(token: str) -> str:
        # Unicode normalize, lower-case, collapse separators to '-'
        t = unicodedata.normalize("NFKC", token)
        t = t.strip().lower()
        t = re.sub(r"[\s_\.]+", "-", t)
        t = re.sub(r"-+", "-", t)
        return t

    @staticmethod
    def _canon_jp(token: str) -> str:
        # Unicode normalize and collapse duplicate long sound marks
        t = unicodedata.normalize("NFKC", token).strip()
        t = re.sub("ー{2,}", "ー", t)
        return t

    @staticmethod
    def _dedup_tokens(tokens: Iterable[str], canon_fn, freq_map: Dict[str, int]) -> List[str]:
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

    def _extract_tags(self, cfg: ExtractionConfig) -> Dict[str, List[str]]:
        summaries = [self._normalize_text(s, cfg) for s in self._read_summaries(cfg.files, cfg.summary_column)]

        ascii_counter: Dict[str, int] = {}
        ja_counter: Dict[str, int] = {}
        for s in summaries:
            for t in self._find_ascii_tokens(s):
                ascii_counter[t] = ascii_counter.get(t, 0) + 1
            for t in self._extract_japanese_tokens(s):
                ja_counter[t] = ja_counter.get(t, 0) + 1

        # Filter by min_support
        ascii_counter = {k: v for k, v in ascii_counter.items() if v >= cfg.min_support}
        ja_counter = {k: v for k, v in ja_counter.items() if v >= cfg.min_support}
        # Embedding-based clustering path only (tokens -> embeddings -> semantic clusters)
        return self._extract_tags_with_embeddings(cfg, ascii_counter, ja_counter)

    def _extract_tags_with_embeddings(
        self,
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
        ascii_dedup = self._dedup_tokens([t for t in tokens_all if t in ascii_counter], self._canon_ascii, ascii_counter)
        ja_dedup = self._dedup_tokens([t for t in tokens_all if t in ja_counter], self._canon_jp, ja_counter)
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
            axis_name = f"Semantic cluster: representative '{rep}'"
            result[axis_name] = toks_sorted[: cfg.top_k_per_axis]
        return result
