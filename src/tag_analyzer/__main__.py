from __future__ import annotations

import argparse
from pathlib import Path

from .tag_analyzer import AnalyzeOptions, analyze_csv_to_markdown


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze tag-assigned CSV and output a Markdown time-series report")
    parser.add_argument("--csv", "-c", required=True, help="Path to tag-assigned CSV file")
    parser.add_argument("--output", "-o", default="reports/tag_report.md", help="Path to output Markdown file")
    parser.add_argument("--date-column", "-d", default="作成日", help="Date column name in CSV (tries to infer if missing)")
    parser.add_argument("--date-format", default=None, help="Date format string if needed (e.g., %Y-%m-%d)")
    parser.add_argument("--freq", "-f", default="W", help="Resample frequency (D/W/M/Q)")
    parser.add_argument("--top-n", type=int, default=10, help="Top N tag values per category to plot")

    args = parser.parse_args(argv)

    opts = AnalyzeOptions(
        date_column=args.date_column,
        date_format=args.date_format,
        freq=args.freq,
        top_n=args.top_n,
    )
    md = analyze_csv_to_markdown(args.csv, args.output, options=opts)
    out = Path(args.output).resolve()
    print(f"Written report: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
