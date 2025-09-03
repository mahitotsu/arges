"""
Tag assignment CLI module

Example:
python -m tag_assigner --tags data/tags.json --csv "data/DevOps-All-QAs (Jira).csv" --column "要約" --output results.csv
"""

import argparse
import json
from pathlib import Path
from .tag_assigner import TagAssigner


def main():
    parser = argparse.ArgumentParser(description="Assign tags to each record of a CSV file")
    
    parser.add_argument(
        '--tags', '-t',
        required=True,
    help='Path to tag definition JSON file (e.g., data/tags.json)'
    )
    
    parser.add_argument(
        '--csv', '-c',
        required=True,
    help='Path to input CSV file'
    )
    
    parser.add_argument(
        '--column', '-col',
        required=True,
    help='Target column name for tag assignment (e.g., "要約")'
    )
    
    parser.add_argument(
        '--output', '-o',
    help='Path to output CSV file (if omitted, summary will be printed only)'
    )
    
    parser.add_argument(
        '--summary', '-s',
        action='store_true',
    help='Print summary of tag assignment'
    )

    # Semantic options
    parser.add_argument(
        '--semantic',
        action='store_true',
        help='Assign tags using semantic similarity (SentenceTransformer)'
    )
    parser.add_argument(
        '--model',
        default='paraphrase-multilingual-MiniLM-L12-v2',
        help='SentenceTransformer model name (multilingual recommended)'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.35,
        help='Threshold for semantic similarity (0–1, e.g., 0.35)'
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=3,
        help='Maximum number of tags per category'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=256,
        help='Batch size for encoding/evaluation in semantic mode (default: 256)'
    )
    parser.add_argument(
        '--progress',
        action='store_true',
        help='Show progress during processing (percentage and ETA)'
    )
    parser.add_argument(
        '--progress-interval',
        type=int,
        default=1000,
        help='Progress print interval (rows). In semantic mode, prints per batch.'
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize tag assigner
        print(f"Loading tag definitions: {args.tags}")
        assigner = TagAssigner(
            args.tags,
            semantic=args.semantic,
            model_name=args.model,
            threshold=args.threshold,
            top_k=args.top_k,
            batch_size=args.batch_size,
        )
        
        # Process CSV
        print(f"Processing CSV file: {args.csv}")
        print(f"Target column: {args.column}")
        
        results = assigner.process_csv(
            csv_file=args.csv,
            target_column=args.column,
            output_file=args.output,
            show_progress=args.progress,
            progress_interval=args.progress_interval,
        )
        
        print(f"Done: processed {len(results)} records")
        
        if args.output:
            print(f"Saved results to: {args.output}")
        
        # Print summary
        if args.summary or not args.output:
            print("\n=== Tag assignment summary ===")
            summary = assigner.generate_summary(results)
            print_summary(summary)
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return 1
    
    return 0


def print_summary(summary):
    """Pretty-print summary"""
    print(f"Total records: {summary['total_records']}")
    print()
    
    for tag_name, stats in summary['tag_statistics'].items():
        print(f"[{tag_name}]")
        print(f"  Records with tags: {stats['records_with_tags']}")
        print(f"  Coverage: {stats['coverage_percentage']}%")
        
        if stats['tag_value_counts']:
            print("  Counts per tag value:")
            sorted_counts = sorted(
                stats['tag_value_counts'].items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            for value, count in sorted_counts[:10]:  # show top 10
                print(f"    {value}: {count}")
        print()


if __name__ == '__main__':
    exit(main())
