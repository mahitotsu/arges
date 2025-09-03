"""Time-series tag analysis (Japanese-friendly).

Provides CLI and library to analyze tag-assigned CSVs and output
Markdown reports with Japanese-capable matplotlib figures.
"""

__all__ = [
    "analyze_csv_to_markdown",
]

from .tag_analyzer import analyze_csv_to_markdown  # noqa: E402,F401
