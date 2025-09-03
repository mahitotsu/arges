"""
Tag assignment module

Simple, readable tag assignment utilities. The TagAssigner focuses on:
- loading tag definitions,
- assigning tags to text (regex or semantic), and
- processing CSVs with a small, composable flow.

Keep data keys in the tag definition (Japanese) as-is to avoid breaking inputs/outputs.
"""

import csv
import json
import re
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict


# Library code should not configure logging; users/CLI can configure handlers.
logger = logging.getLogger(__name__)


class TagAssigner:
    """Assign tags to CSV records"""

    def __init__(
        self,
        tags_file: str,
        *,
        semantic: bool | None = None,
        model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
        threshold: float = 0.35,
        top_k: int = 3,
        batch_size: int = 256,
    ):
        """
        Initialize tag assigner

        Args:
            tags_file: Path to tag definition JSON file
            semantic: Whether to use semantic (meaning-based) matching; None falls back to regex
            model_name: SentenceTransformer model name to use in semantic mode
            threshold: Threshold for semantic similarity (0–1)
            top_k: Maximum number of tags per category
        """
        self.tags = self._load_tags(tags_file)
        self._prepare_matchers()

        # Semantic settings
        self.semantic_requested = bool(semantic) if semantic is not None else False
        self.model_name = model_name
        self.threshold = threshold
        self.top_k = max(1, int(top_k))
        self.batch_size = max(1, int(batch_size))

        self._semantic_enabled = False
        self._st_model = None
        self._st_util = None
        self._tag_embeddings: dict[str, Tuple[List[str], Any]] = {}

        if self.semantic_requested:
            self._init_semantic()
    
    def _load_tags(self, tags_file: str) -> List[Dict[str, Any]]:
        """Load tag definition file"""
        with open(tags_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _prepare_matchers(self):
        """Prepare regex patterns for tag matching"""
        self.tag_patterns = {}
        
        for tag_category in self.tags:
            tag_name = tag_category["タグ名"]
            patterns = []
            
            for tag_value in tag_category["タグの値"]:
                # Convert tag value to regex pattern
                # Alphanumeric: direct, Japanese: allow partial match
                pattern = self._create_pattern(tag_value)
                patterns.append((tag_value, pattern))
            
            self.tag_patterns[tag_name] = patterns
    
    def _create_pattern(self, tag_value: str) -> re.Pattern:
        """Create regex pattern from a tag value"""
        # Escape special characters
        escaped = re.escape(tag_value)
        
        # Allow common variations
        pattern_variations = [
            escaped,  # exact
            escaped.replace(r'\ ', r'[\s_-]*'),  # whitespace/underscore/hyphen variations
            escaped.lower(),  # lowercase variant
        ]
        
        # For English, ignore case; for Japanese allow partial match too
        if re.search(r'[a-zA-Z]', tag_value):
            pattern = '|'.join(pattern_variations)
            return re.compile(f'({pattern})', re.IGNORECASE)
        else:
            # Partial match for Japanese as well
            pattern = '|'.join(pattern_variations)
            return re.compile(f'({pattern})', re.IGNORECASE)

    # =============================
    # Semantic (meaning-based) matching
    # =============================
    def _init_semantic(self) -> None:
        """Initialize SentenceTransformer and precompute tag embeddings"""
        try:
            from sentence_transformers import SentenceTransformer, util  # type: ignore
        except Exception as e:
            # If deps are missing, disable and print a message
            self._semantic_enabled = False
            self._st_model = None
            self._st_util = None
            self._tag_embeddings = {}
            msg = (
                "Failed to enable semantic matching. Required dependencies may be missing. "
                "Install example: uv add 'sentence-transformers>=2.2.2' "
                f"Detail: {e}"
            )
            logger.warning(msg)
            return

        # Load model
        try:
            self._st_model = SentenceTransformer(self.model_name)
            self._st_util = util
        except Exception as e:  # network/model download failures
            self._semantic_enabled = False
            logger.warning(f"Failed to load model: {e}")
            return

        # Precompute embeddings of tag values per category
        try:
            for tag_category in self.tags:
                tag_name = tag_category["タグ名"]
                values: List[str] = list(tag_category["タグの値"])  # コピー
                # encode is internally batched
                embeddings = self._st_model.encode(values, convert_to_tensor=True, normalize_embeddings=True)
                self._tag_embeddings[tag_name] = (values, embeddings)
            self._semantic_enabled = True
        except Exception as e:
            self._semantic_enabled = False
            logger.warning(f"Failed to precompute tag embeddings: {e}")
    
    def _extract_key_tokens(self, text: str) -> List[str]:
        """Extract key tokens from text"""
        if not text:
            return []
        
        # Japanese tokens (hiragana, katakana, kanji sequences)
        japanese_tokens = re.findall(r'[ひ-んア-ヶー一-龠]+', text)
        
        # Alphanumeric tokens (English, digits, some symbols)
        english_tokens = re.findall(r'[a-zA-Z0-9_.-]+', text)
        
        # Exclude tokens that are too long (>= 30 chars)
        japanese_tokens = [t for t in japanese_tokens if len(t) <= 30]
        english_tokens = [t for t in english_tokens if len(t) <= 30]
        
        return japanese_tokens + english_tokens
    
    def assign_tags(self, text: str) -> Dict[str, List[str]]:
        """
        Assign tags to a piece of text
        
        Args:
            text: Input text to assign tags to
            
        Returns:
            Map of tag name to list of tag values
        """
        if not text:
            return {tag_category["タグ名"]: [] for tag_category in self.tags}

        # Prefer semantic mode if enabled
        if self._semantic_enabled:
            return self._assign_tags_semantic(text)
        
        result = defaultdict(list)
        tokens = self._extract_key_tokens(text)
        
        for tag_category in self.tags:
            tag_name = tag_category["タグ名"]
            matched_values = set()  # avoid duplicates
            
            # Try matching against each tag value
            for tag_value, pattern in self.tag_patterns[tag_name]:
                # Direct match in original text
                if pattern.search(text):
                    matched_values.add(tag_value)
                    continue
                
                # Token-level matching
                for token in tokens:
                    if pattern.search(token):
                        matched_values.add(tag_value)
                        break
                
            result[tag_name] = list(matched_values)
        
        return dict(result)

    def _assign_tags_semantic(self, text: str) -> Dict[str, List[str]]:
        """Assign tags using SentenceTransformer-based semantic similarity"""
        if not self._semantic_enabled or not self._st_model:
            # Fallback: no tags
            return {tag_category["タグ名"]: [] for tag_category in self.tags}

        try:
            q = self._st_model.encode(text, convert_to_tensor=True, normalize_embeddings=True)
        except Exception as e:
            logger.debug(f"Failed to embed text (fallback to empty): {e}")
            return {tag_category["タグ名"]: [] for tag_category in self.tags}

        assigned: Dict[str, List[str]] = {}
        for tag_category in self.tags:
            tag_name = tag_category["タグ名"]
            values, emb = self._tag_embeddings.get(tag_name, ([], None))
            if emb is None or not values:
                assigned[tag_name] = []
                continue

            # Cosine similarity
            sims = self._st_util.cos_sim(q, emb).cpu().numpy().flatten()

            # Threshold filter + limit by top_k
            indexed = [(i, float(sims[i])) for i in range(len(values))]
            indexed.sort(key=lambda x: x[1], reverse=True)

            selected: List[str] = []
            for i, score in indexed[: self.top_k]:
                if score >= self.threshold:
                    selected.append(values[i])

            assigned[tag_name] = selected

        return assigned

    def _assign_tags_semantic_batch(self, texts: List[str]) -> List[Dict[str, List[str]]]:
        """Assign semantic tags for a list of texts (batched)"""
        if not self._semantic_enabled or not self._st_model:
            return [{tag_category["タグ名"]: [] for tag_category in self.tags} for _ in texts]

        # Handle empty strings: return empty arrays
        nonempty_mask = [bool(t) for t in texts]
        nonempty_indices = [i for i, m in enumerate(nonempty_mask) if m]
        assigned: List[Dict[str, List[str]]] = [
            {tag_category["タグ名"]: [] for tag_category in self.tags} for _ in texts
        ]

        if not nonempty_indices:
            return assigned

        # Batch encode
        try:
            # SentenceTransformer batches internally, but pass batch_size explicitly
            q_all = self._st_model.encode(
                [texts[i] for i in nonempty_indices],
                convert_to_tensor=True,
                normalize_embeddings=True,
                batch_size=self.batch_size,
            )
        except Exception as e:
            logger.debug(f"Failed to embed texts (batch): {e}")
            return assigned

        # Compute cosine similarities per category, then apply threshold and top_k
        # q_all: (B, D)
        for tag_category in self.tags:
            tag_name = tag_category["タグ名"]
            values, emb = self._tag_embeddings.get(tag_name, ([], None))
            if emb is None or not values:
                continue

            # sims: (B, V)
            sims = self._st_util.cos_sim(q_all, emb).cpu().numpy()

            # For each row, apply top_k and threshold
            for bi, row_idx in enumerate(nonempty_indices):
                scores = sims[bi]
                indexed = [(i, float(scores[i])) for i in range(len(values))]
                indexed.sort(key=lambda x: x[1], reverse=True)
                selected: List[str] = []
                for i, score in indexed[: self.top_k]:
                    if score >= self.threshold:
                        selected.append(values[i])
                assigned[row_idx][tag_name] = selected

        return assigned
    
    
    # -------------------- CSV processing --------------------
    def process_csv(
        self,
        csv_file: str,
        target_column: str,
        output_file: Optional[str] = None,
        *,
        show_progress: bool = False,
        progress_interval: int = 1000,
    ) -> List[Dict[str, Any]]:
        """
        Process CSV and assign tags

        Args:
            csv_file: Path to input CSV file
            target_column: Target column name for tag assignment
            output_file: Path to output CSV file (if None, return results only)

        Returns:
            List of records with assigned tags
        """
        results: List[Dict[str, Any]] = []

        # If semantic is enabled, process CSV in batches; otherwise sequential.
        if self._semantic_enabled:
            return self._process_csv_semantic(csv_file, target_column, output_file, show_progress, progress_interval)
        else:
            return self._process_csv_regex(csv_file, target_column, output_file, show_progress, progress_interval)

        if output_file:
            self._write_results_to_csv(results, output_file)

        return results
    
    def _write_results_to_csv(self, results: List[Dict[str, Any]], output_file: str):
        """Write results to CSV file"""
        if not results:
            return
        
        fieldnames = list(results[0].keys())
        
        with open(output_file, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
    
    def generate_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate a summary of tag assignment results"""
        if results is None:
            results = []
        
        summary = {
            "total_records": len(results),
            "tag_statistics": {}
        }
        
        # Compute statistics per tag category
        for tag_category in self.tags:
            tag_name = tag_category["タグ名"]
            column_name = f"タグ_{tag_name}"

            tag_counts = defaultdict(int)
            records_with_tags = 0

            for result in results:
                tag_values = result.get(column_name, '')
                if tag_values:
                    records_with_tags += 1
                    for value in tag_values.split('; '):
                        if value.strip():
                            tag_counts[value.strip()] += 1

            coverage = 0.0
            if len(results) > 0:
                coverage = round((records_with_tags / len(results)) * 100, 1)

            summary["tag_statistics"][tag_name] = {
                "records_with_tags": records_with_tags,
                "coverage_percentage": coverage,
                "tag_value_counts": dict(tag_counts)
            }
        
        return summary

    # -------------------- Internal CSV methods --------------------
    def _process_csv_semantic(
        self,
        csv_file: str,
        target_column: str,
        output_file: Optional[str],
        show_progress: bool,
        progress_interval: int,
    ) -> List[Dict[str, Any]]:
        import time

        results: List[Dict[str, Any]] = []
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)

            writer = None
            out_f = None
            if output_file:
                out_f = open(output_file, 'w', encoding='utf-8', newline='')

            buffer_rows: List[Dict[str, Any]] = []
            buffer_texts: List[str] = []
            first_row_fieldnames: Optional[List[str]] = None

            processed = 0
            start_time = time.time() if show_progress else None

            def flush_buffer() -> None:
                nonlocal writer, out_f, buffer_rows, buffer_texts, first_row_fieldnames, processed
                if not buffer_rows:
                    return
                assignments = self._assign_tags_semantic_batch(buffer_texts)
                for r, a in zip(buffer_rows, assignments):
                    enhanced_row = r.copy()
                    for tag_name, tag_values in a.items():
                        enhanced_row[f"タグ_{tag_name}"] = '; '.join(tag_values) if tag_values else ''
                    results.append(enhanced_row)
                    if out_f is not None:
                        if writer is None:
                            if first_row_fieldnames is None:
                                first_row_fieldnames = list(enhanced_row.keys())
                            writer = csv.DictWriter(out_f, fieldnames=first_row_fieldnames)
                            writer.writeheader()
                        writer.writerow(enhanced_row)
                processed += len(buffer_rows)
                buffer_rows = []
                buffer_texts = []
                if show_progress and progress_interval > 0 and processed % max(1, progress_interval) == 0:
                    if start_time:
                        elapsed = time.time() - start_time
                        speed = processed / elapsed if elapsed > 0 else 0
                        print(f"[semantic] processed {processed} rows (speed ~{speed:.1f}/s)")

            for row in reader:
                if target_column not in row:
                    if out_f is not None:
                        out_f.close()
                    raise ValueError(f"Column '{target_column}' not found. Available columns: {list(row.keys())}")
                text = row.get(target_column) or ''
                buffer_rows.append(row)
                buffer_texts.append(text)
                if len(buffer_rows) >= self.batch_size:
                    flush_buffer()

            # flush tail
            flush_buffer()
            if out_f is not None:
                out_f.close()

        return results

    def _process_csv_regex(
        self,
        csv_file: str,
        target_column: str,
        output_file: Optional[str],
        show_progress: bool,
        progress_interval: int,
    ) -> List[Dict[str, Any]]:
        import time

        results: List[Dict[str, Any]] = []
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)

            processed = 0
            start_time = time.time() if show_progress else None

            for row in reader:
                if target_column not in row:
                    raise ValueError(f"Column '{target_column}' not found. Available columns: {list(row.keys())}")
                text = row.get(target_column, '')
                assigned_tags = self.assign_tags(text)
                enhanced_row = row.copy()
                for tag_name, tag_values in assigned_tags.items():
                    enhanced_row[f"タグ_{tag_name}"] = '; '.join(tag_values) if tag_values else ''
                results.append(enhanced_row)
                processed += 1
                if show_progress and processed % max(1, progress_interval) == 0:
                    if start_time:
                        elapsed = time.time() - start_time
                        speed = processed / elapsed if elapsed > 0 else 0
                        print(f"[regex] processed {processed} rows (speed ~{speed:.1f}/s)")

        if output_file:
            self._write_results_to_csv(results, output_file)

        return results
