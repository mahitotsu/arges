"""
タグ割り当てモジュール

CSVファイルの各レコードに対して、タグ名とタグ値のリストを基に
最適なタグを割り当てるモジュールです。
"""

import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict


class TagAssigner:
    """CSVレコードにタグを割り当てるクラス"""

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
        タグ割り当て器を初期化

        Args:
            tags_file: タグ定義JSONファイルのパス
            semantic: セマンティック（意味ベース）マッチングを使うかどうか。None の場合は正規表現にフォールバック
            model_name: セマンティックモードで使用するSentenceTransformerモデル名
            threshold: セマンティック類似度の閾値（0〜1）
            top_k: 各カテゴリで最大いくつまでタグを付与するか
        """
        self.tags = self._load_tags(tags_file)
        self._prepare_matchers()

        # セマンティック設定
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
        """タグ定義ファイルを読み込み"""
        with open(tags_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _prepare_matchers(self):
        """タグマッチング用の正規表現パターンを準備"""
        self.tag_patterns = {}
        
        for tag_category in self.tags:
            tag_name = tag_category["タグ名"]
            patterns = []
            
            for tag_value in tag_category["タグの値"]:
                # タグ値を正規表現パターンに変換
                # 英数字はそのまま、日本語は部分マッチで対応
                pattern = self._create_pattern(tag_value)
                patterns.append((tag_value, pattern))
            
            self.tag_patterns[tag_name] = patterns
    
    def _create_pattern(self, tag_value: str) -> re.Pattern:
        """タグ値から正規表現パターンを生成"""
        # 特殊文字をエスケープ
        escaped = re.escape(tag_value)
        
        # よくある変形パターンに対応
        pattern_variations = [
            escaped,  # 完全一致
            escaped.replace(r'\ ', r'[\s_-]*'),  # スペース、アンダースコア、ハイフンの変形
            escaped.lower(),  # 小文字版
        ]
        
        # 英語の場合は大文字小文字を区別しない
        if re.search(r'[a-zA-Z]', tag_value):
            pattern = '|'.join(pattern_variations)
            return re.compile(f'({pattern})', re.IGNORECASE)
        else:
            # 日本語の場合は部分マッチも許可
            pattern = '|'.join(pattern_variations)
            return re.compile(f'({pattern})', re.IGNORECASE)

    # =============================
    # セマンティック（意味ベース）関連
    # =============================
    def _init_semantic(self) -> None:
        """SentenceTransformerの初期化とタグ埋め込みの事前計算"""
        try:
            from sentence_transformers import SentenceTransformer, util  # type: ignore
        except Exception as e:
            # 依存がない場合は明示的に無効化してメッセージを添える
            self._semantic_enabled = False
            self._st_model = None
            self._st_util = None
            self._tag_embeddings = {}
            msg = (
                "セマンティックマッチングを有効化できませんでした。必要な依存関係が未インストールの可能性があります。\n"
                "インストール例: uv add 'sentence-transformers>=2.2.2'\n"
                f"詳細: {e}"
            )
            print(msg)
            return

        # モデル読み込み
        try:
            self._st_model = SentenceTransformer(self.model_name)
            self._st_util = util
        except Exception as e:  # ネットワークやモデルDLの失敗時
            self._semantic_enabled = False
            print(f"モデルの読み込みに失敗しました: {e}")
            return

        # タグ値の埋め込みをカテゴリごとに事前計算
        try:
            for tag_category in self.tags:
                tag_name = tag_category["タグ名"]
                values: List[str] = list(tag_category["タグの値"])  # コピー
                # encodeは内部でバッチ化される
                embeddings = self._st_model.encode(values, convert_to_tensor=True, normalize_embeddings=True)
                self._tag_embeddings[tag_name] = (values, embeddings)
            self._semantic_enabled = True
        except Exception as e:
            self._semantic_enabled = False
            print(f"タグ埋め込みの事前計算に失敗しました: {e}")
    
    def _extract_key_tokens(self, text: str) -> List[str]:
        """テキストから重要なトークンを抽出"""
        if not text:
            return []
        
        # 日本語トークン（ひらがな、カタカナ、漢字の連続）
        japanese_tokens = re.findall(r'[ひ-んア-ヶー一-龠]+', text)
        
        # 英数字トークン（英語、数字、一部記号の連続）
        english_tokens = re.findall(r'[a-zA-Z0-9_.-]+', text)
        
        # 長すぎるトークンは除外（30文字以上）
        japanese_tokens = [t for t in japanese_tokens if len(t) <= 30]
        english_tokens = [t for t in english_tokens if len(t) <= 30]
        
        return japanese_tokens + english_tokens
    
    def assign_tags(self, text: str) -> Dict[str, List[str]]:
        """
        テキストに対してタグを割り当て
        
        Args:
            text: 割り当て対象のテキスト
            
        Returns:
            各タグ名に対するタグ値のリスト
        """
        if not text:
            return {tag_category["タグ名"]: [] for tag_category in self.tags}

        # セマンティックモードが有効ならそちらを優先
        if self._semantic_enabled:
            return self._assign_tags_semantic(text)
        
        result = defaultdict(list)
        tokens = self._extract_key_tokens(text)
        text_lower = text.lower()
        
        for tag_category in self.tags:
            tag_name = tag_category["タグ名"]
            matched_values = set()  # 重複を避けるためにsetを使用
            
            # 各タグ値とのマッチングを試行
            for tag_value, pattern in self.tag_patterns[tag_name]:
                # 元テキストでの直接マッチング
                if pattern.search(text):
                    matched_values.add(tag_value)
                    continue
                
                # トークン単位でのマッチング
                for token in tokens:
                    if pattern.search(token):
                        matched_values.add(tag_value)
                        break
                
                # キーワードベースの特別ルール
                if self._special_keyword_match(tag_name, tag_value, text_lower, tokens):
                    matched_values.add(tag_value)
            
            result[tag_name] = list(matched_values)
        
        return dict(result)

    def _assign_tags_semantic(self, text: str) -> Dict[str, List[str]]:
        """SentenceTransformerを用いた意味ベースのタグ割り当て"""
        if not self._semantic_enabled or not self._st_model:
            # フォールバック
            return {tag_category["タグ名"]: [] for tag_category in self.tags}

        try:
            q = self._st_model.encode(text, convert_to_tensor=True, normalize_embeddings=True)
        except Exception as e:
            print(f"テキストの埋め込みに失敗しました（フォールバックして空配列）: {e}")
            return {tag_category["タグ名"]: [] for tag_category in self.tags}

        assigned: Dict[str, List[str]] = {}
        for tag_category in self.tags:
            tag_name = tag_category["タグ名"]
            values, emb = self._tag_embeddings.get(tag_name, ([], None))
            if emb is None or not values:
                assigned[tag_name] = []
                continue

            # 類似度計算（cosine）
            sims = self._st_util.cos_sim(q, emb).cpu().numpy().flatten()

            # 閾値フィルタ＋top_kで制限
            indexed = [(i, float(sims[i])) for i in range(len(values))]
            indexed.sort(key=lambda x: x[1], reverse=True)

            selected: List[str] = []
            for i, score in indexed[: self.top_k]:
                if score >= self.threshold:
                    selected.append(values[i])

            assigned[tag_name] = selected

        return assigned

    def _assign_tags_semantic_batch(self, texts: List[str]) -> List[Dict[str, List[str]]]:
        """テキストのリストに対して意味ベースのタグ割り当て（バッチ）を実行"""
        if not self._semantic_enabled or not self._st_model:
            return [{tag_category["タグ名"]: [] for tag_category in self.tags} for _ in texts]

        # 空文字対策：None/空は空配列返却
        nonempty_mask = [bool(t) for t in texts]
        nonempty_indices = [i for i, m in enumerate(nonempty_mask) if m]
        assigned: List[Dict[str, List[str]]] = [
            {tag_category["タグ名"]: [] for tag_category in self.tags} for _ in texts
        ]

        if not nonempty_indices:
            return assigned

        # バッチエンコード
        try:
            # SentenceTransformerは内部でもバッチ化されるが、明示的にbatch_sizeを渡す
            q_all = self._st_model.encode(
                [texts[i] for i in nonempty_indices],
                convert_to_tensor=True,
                normalize_embeddings=True,
                batch_size=self.batch_size,
            )
        except Exception as e:
            print(f"テキストの埋め込み（バッチ）に失敗しました: {e}")
            return assigned

        # 各カテゴリごとにcosine類似度を一括計算し、しきい値・top_kでフィルタ
        # q_all: (B, D)
        for tag_category in self.tags:
            tag_name = tag_category["タグ名"]
            values, emb = self._tag_embeddings.get(tag_name, ([], None))
            if emb is None or not values:
                continue

            # sims: (B, V)
            sims = self._st_util.cos_sim(q_all, emb).cpu().numpy()

            # 各行についてtop_k・thresholdで抽出
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
    
    def _special_keyword_match(self, tag_name: str, tag_value: str, text_lower: str, tokens: List[str]) -> bool:
        """特別なキーワードマッチングルール"""
        
        # 依頼種別の特別ルール
        if tag_name == "依頼種別":
            if tag_value == "新規利用依頼" and any(word in text_lower for word in ["新規", "利用", "申請"]):
                return True
            elif tag_value == "権限申請" and any(word in text_lower for word in ["権限", "申請", "付与"]):
                return True
            elif tag_value == "作成依頼" and any(word in text_lower for word in ["作成", "新規作成"]):
                return True
            elif tag_value == "設定依頼" and any(word in text_lower for word in ["設定", "config"]):
                return True
            elif tag_value == "削除依頼" and any(word in text_lower for word in ["削除", "delete"]):
                return True
            elif tag_value == "確認依頼" and any(word in text_lower for word in ["確認", "チェック"]):
                return True
            elif tag_value == "構築依頼" and any(word in text_lower for word in ["構築", "build"]):
                return True
            elif tag_value == "デプロイ依頼" and any(word in text_lower for word in ["デプロイ", "deploy"]):
                return True
        
        # エラー・障害の特別ルール
        elif tag_name == "エラー・障害":
            if tag_value == "ログインできない" and any(word in text_lower for word in ["ログイン", "login"]) and any(word in text_lower for word in ["できない", "不可", "failed"]):
                return True
            elif tag_value == "接続できない" and any(word in text_lower for word in ["接続", "connect"]) and any(word in text_lower for word in ["できない", "不可", "failed"]):
                return True
            elif tag_value == "エラー発生" and any(word in text_lower for word in ["エラー", "error"]):
                return True
            elif tag_value == "デプロイ失敗" and any(word in text_lower for word in ["デプロイ", "deploy"]) and any(word in text_lower for word in ["失敗", "failed"]):
                return True
        
        # クラウド・技術の特別ルール
        elif tag_name == "クラウド・技術":
            if tag_value == "Google Cloud" and any(word in text_lower for word in ["gcp", "google", "cloud"]):
                return True
            elif tag_value == "GitHub" and any(word in text_lower for word in ["github", "git"]):
                return True
            elif tag_value == "Kubernetes" and any(word in text_lower for word in ["k8s", "kubernetes"]):
                return True
        
        return False
    
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
        CSVファイルを処理してタグを割り当て

        Args:
            csv_file: 入力CSVファイルのパス
            target_column: タグ割り当て対象の列名
            output_file: 出力CSVファイルのパス（Noneの場合は結果を返すのみ）

        Returns:
            タグが割り当てられたレコードのリスト
        """
        results: List[Dict[str, Any]] = []

        # セマンティックが有効なら、CSVはバッチ処理（高速化）
        if self._semantic_enabled:
            import time
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

                def flush_buffer():
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
                        raise ValueError(f"列 '{target_column}' が見つかりません。利用可能な列: {list(row.keys())}")
                    text = row.get(target_column) or ''
                    buffer_rows.append(row)
                    buffer_texts.append(text)
                    if len(buffer_rows) >= self.batch_size:
                        flush_buffer()

                # 端数をflush
                flush_buffer()
                if out_f is not None:
                    out_f.close()

            return results

        # それ以外（正規表現など）は従来通り逐次処理
        import time
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)

            processed = 0
            start_time = time.time() if show_progress else None

            for row in reader:
                if target_column not in row:
                    raise ValueError(f"列 '{target_column}' が見つかりません。利用可能な列: {list(row.keys())}")
                text = row[target_column]
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
    
    def _write_results_to_csv(self, results: List[Dict[str, Any]], output_file: str):
        """結果をCSVファイルに書き出し"""
        if not results:
            return
        
        fieldnames = list(results[0].keys())
        
        with open(output_file, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
    
    def generate_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """タグ割り当て結果のサマリーを生成"""
        if results is None:
            results = []
        
        summary = {
            "total_records": len(results),
            "tag_statistics": {}
        }
        
        # 各タグカテゴリの統計を計算
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
