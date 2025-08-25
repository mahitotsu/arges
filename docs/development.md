# 開発ガイド

このドキュメントは Arges プロジェクトの開発者向けの情報を提供します。

## 開発環境

このプロジェクトは、モダンな Python 開発のベストプラクティスに従っています：

- **pyproject.toml**: モダンな Python プロジェクト設定
- **Makefile**: 便利な開発コマンド
- **Pre-commit hooks**: 自動化されたコード品質チェック
- **型ヒント**: 完全な型注釈サポート
- **テスト**: カバレッジレポート付き Pytest
- **ドキュメント**: Material テーマを使用した MkDocs

### コード品質ツール

#### Python

- **Black**: コードフォーマッタ
- **Ruff**: 高速なリンター・フォーマッタ（flake8、isort、pyupgradeの代替）
- **MyPy**: 静的型チェッカー
- **Bandit**: セキュリティチェッカー
- **Safety**: 依存関係の脆弱性チェック

#### Markdown

- **markdownlint-cli2**: Markdownリンター
- **Prettier**: Markdownフォーマッタ

すべてのツールは統一された設定（行長88文字など）を使用しています。

### 技術スタック

#### 核となる技術

- **AWS Bedrock**: LLMサービス（Titan、Claude、Cohereなど）
- **boto3**: AWS SDK for Python
- **scikit-learn**: 機械学習・クラスタリング
- **pandas**: データ処理・分析
- **plotly, matplotlib, seaborn**: データ可視化
- **numpy, scipy**: 数値計算

#### ファイル形式サポート

- **openpyxl**: Excel ファイル処理
- **pandas**: CSV、JSON、TXT対応

## 開発環境セットアップ

### 前提条件

- Python 3.9 以上
- Node.js 16 以上（Markdownツール用）
- Git
- Make
- **AWS アカウント**（AWS Bedrock利用のため）
- **AWS CLI**（認証情報設定用、オプション）

### セットアップ手順

```bash
# リポジトリをクローン
git clone https://github.com/mahitotsu/arges.git
cd arges

# 完全な開発環境をセットアップ
make setup

# 仮想環境をアクティベート
source venv/bin/activate
```

## よく使用する開発コマンド

### インストール・セットアップ

```bash
# 開発依存関係をインストール
make install-dev

# 完全な開発環境セットアップ（venv作成 + 依存関係インストール + pre-commit設定）
make setup
```

### テスト

```bash
# テストを実行
make test

# カバレッジ付きでテストを実行
make test-cov
```

### コード品質

```bash
# すべてのコードをフォーマット（Python + Markdown）
make format

# Pythonコードのみフォーマット
make format-python

# Markdownファイルのみフォーマット
make format-markdown

# すべてのコードをリント（Python + Markdown）
make lint

# Pythonコードのみリント
make lint-python

# Markdownファイルのみリント
make lint-markdown

# 型チェック
make type-check
make lint
```

### ビルド・クリーン

```bash
# パッケージをビルド
make build

# プロジェクトをクリーン
make clean
```

### ドキュメント

```bash
# ドキュメントをローカルで配信
make serve-docs
```

### アプリケーション実行

```bash
# チャットセッションを開始
make run-chat
```

## プロジェクト構造

```text
arges/
├── src/arges/          # メインパッケージ
│   ├── __init__.py     # パッケージ初期化
│   ├── cli.py          # コマンドラインインターフェース
│   ├── client.py       # メインクライアント
│   ├── config.py       # 設定管理
│   └── models.py       # データモデル
├── tests/              # テストスイート
│   ├── __init__.py
│   ├── test_cli.py     # CLI テスト
│   └── test_client.py  # クライアントテスト
├── docs/               # ドキュメント
│   ├── index.md        # メインドキュメント
│   └── development.md  # 開発ガイド（このファイル）
├── scripts/            # 開発スクリプト
├── config/             # 設定ファイル
├── pyproject.toml      # プロジェクト設定
├── Makefile           # 開発コマンド
├── mkdocs.yml         # ドキュメント設定
├── requirements.txt   # 依存関係
└── README.md          # プロジェクト概要
```

## 開発ワークフロー

### 1. 機能開発

```bash
# 新しいブランチを作成
git checkout -b feature/new-feature

# 開発
# ... コード変更 ...

# テストを実行
make test

# コードをフォーマット・リント
make format
make lint

# コミット（pre-commitフックが自動実行される）
git add .
git commit -m "Add new feature"
```

### 2. テスト

```bash
# 全テストを実行
make test

# カバレッジ付きテスト
make test-cov

# 特定のテストを実行
pytest tests/test_specific.py
```

### 3. コード品質チェック

プロジェクトでは以下のツールを使用しています：

- **Black**: コードフォーマッター
- **isort**: インポート文の整理
- **flake8**: リンター
- **mypy**: 型チェック
- **pre-commit**: Git フック

```bash
# 全ての品質チェックを実行
make lint

# 個別実行
black src/ tests/
isort src/ tests/
flake8 src/ tests/
mypy src/
```

## 設定ファイル

### pyproject.toml

プロジェクトの主要な設定ファイル：

- パッケージメタデータ
- 依存関係
- ツール設定（black, isort, mypy など）

### Makefile

開発タスクの定義：

- インストール・セットアップ
- テスト実行
- コード品質チェック
- ビルド・デプロイ

### .pre-commit-config.yaml

Pre-commit フックの設定（自動的にインストールされます）

## コントリビューション

### プルリクエストのガイドライン

1. **ブランチ命名**: `feature/`, `bugfix/`, `hotfix/` プレフィックスを使用
2. **テスト**: 新機能には適切なテストを追加
3. **ドキュメント**: APIの変更時はドキュメントも更新
4. **コード品質**: `make lint` を通すこと
5. **コミットメッセージ**: 明確で説明的なメッセージ

### Issue テンプレート

<!-- TODO: Issue テンプレートが必要な場合は .github/ISSUE_TEMPLATE/ を作成 -->

## トラブルシューティング

### よくある問題

#### 仮想環境の問題

```bash
# 仮想環境を削除して再作成
rm -rf venv
make setup
```

#### 依存関係の問題

```bash
# pip をアップグレード
pip install --upgrade pip

# 依存関係を再インストール
make clean
make install-dev
```

#### テストの失敗

```bash
# 詳細な出力でテストを実行
pytest -v

# 特定のテストのみ実行
pytest tests/test_specific.py::test_function -v
```
