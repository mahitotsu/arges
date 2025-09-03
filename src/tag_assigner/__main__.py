"""
タグ割り当てCLIモジュール

使用例:
python -m tag_assigner --tags data/tags.json --csv "data/DevOps-All-QAs (Jira).csv" --column "要約" --output results.csv
"""

import argparse
import json
from pathlib import Path
from .tag_assigner import TagAssigner


def main():
    parser = argparse.ArgumentParser(description="CSVファイルの各レコードにタグを割り当て")
    
    parser.add_argument(
        '--tags', '-t',
        required=True,
        help='タグ定義JSONファイルのパス (例: data/tags.json)'
    )
    
    parser.add_argument(
        '--csv', '-c',
        required=True,
        help='入力CSVファイルのパス'
    )
    
    parser.add_argument(
        '--column', '-col',
        required=True,
        help='タグ割り当て対象の列名 (例: "要約")'
    )
    
    parser.add_argument(
        '--output', '-o',
        help='出力CSVファイルのパス (指定しない場合はサマリーのみ表示)'
    )
    
    parser.add_argument(
        '--summary', '-s',
        action='store_true',
        help='タグ割り当て結果のサマリーを表示'
    )

    # セマンティックオプション
    parser.add_argument(
        '--semantic',
        action='store_true',
        help='意味ベース（SentenceTransformer）でタグを割り当てる'
    )
    parser.add_argument(
        '--model',
        default='paraphrase-multilingual-MiniLM-L12-v2',
        help='SentenceTransformerのモデル名（多言語対応を推奨）'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.35,
        help='セマンティック類似度の閾値（0〜1, 例: 0.35）'
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=3,
        help='各カテゴリで最大いくつまでタグを付与するか'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=256,
        help='セマンティックモード時のエンコード/評価のバッチサイズ（既定: 256）'
    )
    parser.add_argument(
        '--progress',
        action='store_true',
        help='処理進捗を表示する（総件数に対する%とETA）'
    )
    parser.add_argument(
        '--progress-interval',
        type=int,
        default=1000,
        help='進捗表示の間隔（件数）。セマンティック時はバッチ単位で表示'
    )
    
    args = parser.parse_args()
    
    try:
        # タグ割り当て器を初期化
        print(f"タグ定義を読み込んでいます: {args.tags}")
        assigner = TagAssigner(
            args.tags,
            semantic=args.semantic,
            model_name=args.model,
            threshold=args.threshold,
            top_k=args.top_k,
            batch_size=args.batch_size,
        )
        
        # CSVを処理
        print(f"CSVファイルを処理しています: {args.csv}")
        print(f"対象列: {args.column}")
        
        results = assigner.process_csv(
            csv_file=args.csv,
            target_column=args.column,
            output_file=args.output,
            show_progress=args.progress,
            progress_interval=args.progress_interval,
        )
        
        print(f"処理完了: {len(results)} レコードを処理しました")
        
        if args.output:
            print(f"結果を保存しました: {args.output}")
        
        # サマリーを表示
        if args.summary or not args.output:
            print("\n=== タグ割り当てサマリー ===")
            summary = assigner.generate_summary(results)
            print_summary(summary)
    
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        return 1
    
    return 0


def print_summary(summary):
    """サマリーを見やすく表示"""
    print(f"総レコード数: {summary['total_records']}")
    print()
    
    for tag_name, stats in summary['tag_statistics'].items():
        print(f"【{tag_name}】")
        print(f"  タグ付きレコード数: {stats['records_with_tags']}")
        print(f"  カバレッジ: {stats['coverage_percentage']}%")
        
        if stats['tag_value_counts']:
            print("  タグ値別件数:")
            sorted_counts = sorted(
                stats['tag_value_counts'].items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            for value, count in sorted_counts[:10]:  # 上位10位まで表示
                print(f"    {value}: {count}件")
        print()


if __name__ == '__main__':
    exit(main())
