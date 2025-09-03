from __future__ import annotations

import os
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import matplotlib
import pandas as pd

# Use non-interactive backend
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402


# Library code should not configure logging; users/CLI can configure handlers.
logger = logging.getLogger(__name__)


@dataclass
class AnalyzeOptions:
    date_column: str = "作成日"
    date_format: Optional[str] = None  # e.g., "%Y-%m-%d". If None, pandas will infer
    freq: str = "W"  # resample frequency (D/W/M/Q)
    top_n: int = 10  # top N tag values to chart per category
    figure_width: float = 9.0
    figure_height: float = 4.8


def _ensure_japanese_font() -> None:
    """Configure a Japanese-capable font if available.

    Strategy:
    1) Try common JP font family names from the font cache.
    2) If not found, scan system font files (Noto/IPA/Source Han, etc.), register the first hit, and use it.
    """
    from matplotlib import font_manager, rcParams

    # 0) Explicit override via env var (family name or file path)
    env_font = os.getenv("ARGES_JP_FONT") or os.getenv("MPL_JP_FONT")
    if env_font:
        try:
            if os.path.isfile(env_font):
                font_manager.fontManager.addfont(env_font)
                prop = font_manager.FontProperties(fname=env_font)
                name = prop.get_name()
            else:
                name = env_font
            if name:
                rcParams["font.family"] = [name]
                rcParams["font.sans-serif"] = [name]
                rcParams["axes.unicode_minus"] = False
                return
        except Exception:
            pass

    # 1) Short-circuit if user explicitly configured a non-generic, non-DejaVu family
    current_family = rcParams.get("font.family")
    if current_family:
        fam_list = current_family if isinstance(current_family, (list, tuple)) else [current_family]
        def is_generic(x: str) -> bool:
            return str(x).strip().lower() in {"sans-serif", "serif", "monospace"}
        if any((not is_generic(f)) and ("dejavu" not in str(f).lower()) for f in fam_list):
            rcParams["axes.unicode_minus"] = False
            return

    candidate_names = [
        "Noto Sans CJK JP",
        "Noto Sans JP",
        "Source Han Sans JP",
        "IPAPGothic",
        "IPAexGothic",
        "TakaoPGothic",
        "VL PGothic",
        "Yu Gothic",
        "Hiragino Sans",
        "Hiragino Kaku Gothic ProN",
    ]

    chosen_name: Optional[str] = None
    chosen_path: Optional[str] = None

    # Pass 1: pick by family name from existing cache
    for fe in font_manager.fontManager.ttflist:
        try:
            fname = getattr(fe, "name", None)
            fpath = getattr(fe, "fname", None)
        except Exception:
            fname, fpath = None, None
        if not fname:
            continue
        for cand in candidate_names:
            if cand.lower().replace(" ", "") in str(fname).lower().replace(" ", ""):
                chosen_name = str(fname)
                chosen_path = str(fpath) if fpath else None
                break
        if chosen_name:
            break

    # Pass 2: search font files by path if not found in cache
    if not chosen_name:
        # Common JP font filename hints
        filename_hints = [
            "NotoSansCJKjp",  # OTF
            "NotoSansJP",     # TTF
            "SourceHanSansJP",
            "IPAGothic",
            "IPAexGothic",
            "TakaoPGothic",
            "VL-PGothic",
        ]
        try:
            # Search known system font locations
            system_fonts = set(font_manager.findSystemFonts(fontext="ttf") + font_manager.findSystemFonts(fontext="otf"))
        except Exception:
            system_fonts = set()
        # Also probe a few common directories explicitly
        extra_dirs = [
            "/usr/share/fonts", "/usr/local/share/fonts", str(Path.home() / ".local/share/fonts"), str(Path.home() / ".fonts")
        ]
        for d in extra_dirs:
            try:
                system_fonts.update(font_manager.findSystemFonts(d, fontext="ttf"))
                system_fonts.update(font_manager.findSystemFonts(d, fontext="otf"))
            except Exception:
                pass

        # Pick the first matching file
        for fpath in sorted(system_fonts):
            low = os.path.basename(fpath).lower()
            if any(hint.lower() in low for hint in filename_hints):
                try:
                    font_manager.fontManager.addfont(fpath)
                    prop = font_manager.FontProperties(fname=fpath)
                    name = prop.get_name()
                    if name:
                        chosen_name = name
                        chosen_path = fpath
                        break
                except Exception:
                    continue

    # Apply if found
    if chosen_name:
        rcParams["font.family"] = [chosen_name]
        rcParams["font.sans-serif"] = [chosen_name] + candidate_names  # allow fallbacks
        rcParams["axes.unicode_minus"] = False
    else:
        # Keep going with DejaVu (figures will warn). Provide at least proper minus rendering.
        rcParams["axes.unicode_minus"] = False
        # Emit a concise hint via logger (CLI may surface it as stderr depending on handler)
        logger.warning(
            "Japanese font not found. Set ARGES_JP_FONT to a JP font name or path, or install fonts-noto-cjk."
        )


def _melt_tag_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert columns like "タグ_問題種別" with semicolon-separated values
    into long-form rows: [date, category, value].
    """
    tag_cols = [c for c in df.columns if c.startswith("タグ_")]
    if not tag_cols:
        return pd.DataFrame(columns=["date", "category", "value"])  # empty

    rows: List[dict] = []
    for _, r in df.iterrows():
        date = r["__date__"]
        for col in tag_cols:
            cell = r.get(col)
            # Skip pandas NA/NaT and empty strings to avoid 'nan' noise
            if pd.isna(cell):
                continue
            raw = str(cell).strip()
            if not raw or raw.lower() in {"nan", "nat"}:
                continue
            cat = col.replace("タグ_", "")
            vals = []
            for x in raw.split(";"):
                x = x.strip()
                if not x:
                    continue
                lx = x.lower()
                if lx in {"nan", "nat", "none", "null"}:
                    continue
                vals.append(x)
            for val in vals:
                rows.append({"date": date, "category": cat, "value": val})
    return pd.DataFrame(rows)


def analyze_csv_to_markdown(
    csv_file: str | os.PathLike[str],
    output_md: str | os.PathLike[str],
    *,
    options: AnalyzeOptions | None = None,
) -> str:
    """Analyze a tag-assigned CSV and write a Markdown report with figures.

    Returns the generated Markdown as a string.
    """
    options = options or AnalyzeOptions()
    _ensure_japanese_font()

    csv_path = Path(csv_file)
    out_md_path = Path(output_md)
    out_md_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)

    # Pick a date column: prefer explicit setting; fallbacks common in Jira exports
    date_col = options.date_column if options.date_column in df.columns else None
    if date_col is None:
        for c in ["作成日", "作成日時", "作成", "Created", "Created At", "created", "created_at", "日付"]:
            if c in df.columns:
                date_col = c
                break
    if date_col is None:
        # create synthetic date if missing
        df["__date__"] = pd.to_datetime("today").normalize()
    else:
        # Normalize Japanese AM/PM if present
        ser = df[date_col].astype(str).str.replace(r"\s+", " ", regex=True)
        ser = ser.str.replace("午前", "AM").str.replace("午後", "PM")
        # First try: user-specified format
        dt = pd.to_datetime(ser, format=options.date_format, errors="coerce")
        # Fallback 1: infer with dateutil
        if dt.isna().mean() > 0.8:
            dt = pd.to_datetime(ser, errors="coerce", infer_datetime_format=True)
        # Fallback 2: try dayfirst
        if dt.isna().mean() > 0.8:
            try:
                dt = pd.to_datetime(ser, errors="coerce", dayfirst=True)
            except Exception:
                pass
        # If still most are NaT, set today as a synthetic date to enable counts (single bucket)
        if dt.isna().mean() > 0.95:
            dt = pd.Series(pd.to_datetime("today").normalize(), index=df.index)
        df["__date__"] = dt
        # drop rows without a valid date (keep synthetic)
        df = df.dropna(subset=["__date__"]).copy()

    # Melt tag columns
    long_df = _melt_tag_columns(df)
    if long_df.empty:
        # Diagnose reason
        tag_cols = [c for c in df.columns if c.startswith("タグ_")]
        if not tag_cols:
            msg = "タグ列（'タグ_...'）が見つかりませんでした。元CSVを確認してください。"
        else:
            msg = "有効な日付またはタグ値が見つからず集計できませんでした。日付形式や列名を確認してください。"
        md = f"# タグ時系列レポート\n\n{msg}\n"
        out_md_path.write_text(md, encoding="utf-8")
        return md

    # Resample by frequency per category/value
    long_df = long_df.sort_values("date")
    long_df.set_index("date", inplace=True)

    # Compute per-category top values and build charts
    figs_dir = out_md_path.parent / "figs"
    figs_dir.mkdir(parents=True, exist_ok=True)

    md_parts: List[str] = []
    md_parts.append("# タグ時系列レポート\n")
    md_parts.append(f"- 入力: `{csv_path}`\n")
    md_parts.append(f"- 粒度: {options.freq}\n")
    md_parts.append("")

    sns.set_style("whitegrid")
    # seaborn can reset font.family to generic; re-apply JP font
    _ensure_japanese_font()

    # Overall trend: total tagged items per period
    total_series = long_df.assign(n=1).resample(options.freq)["n"].sum().dropna()
    fig, ax = plt.subplots(figsize=(options.figure_width, options.figure_height))
    total_series.plot(ax=ax)
    ax.set_title("全タグ出現数の推移")
    ax.set_xlabel("日付")
    ax.set_ylabel("件数")
    fig.tight_layout()
    total_png = figs_dir / "overall_trend.png"
    fig.savefig(total_png, dpi=150)
    plt.close(fig)
    md_parts.append("## 全体傾向\n")
    md_parts.append(f"![overall]({(total_png.relative_to(out_md_path.parent)).as_posix()})\n")

    # Per category: top N values time series
    for cat, g in long_df.groupby("category"):
        counts = g.assign(n=1).groupby("value")["n"].sum().sort_values(ascending=False)
        top_values = list(counts.head(options.top_n).index)
        sub = g[g["value"].isin(top_values)].copy()

        if sub.empty:
            continue

        pivot = (
            sub.assign(n=1)
            .groupby([pd.Grouper(freq=options.freq), "value"])  # resample then value
            ["n"].sum()
            .unstack(fill_value=0)
        )
        # Remove any rows that are entirely NaN (should be none due to fill_value), just in case
        pivot = pivot.dropna(how="all")

        fig, ax = plt.subplots(figsize=(options.figure_width, options.figure_height))
        pivot.plot(ax=ax)
        ax.set_title(f"{cat}: 上位{len(top_values)}タグの推移")
        ax.set_xlabel("日付")
        ax.set_ylabel("件数")
        fig.tight_layout()
        fname = f"category_{cat}.png".replace("/", "-")
        path_png = figs_dir / fname
        fig.savefig(path_png, dpi=150)
        plt.close(fig)

        md_parts.append(f"## {cat}\n")
        md_parts.append(f"![{cat}]({(path_png.relative_to(out_md_path.parent)).as_posix()})\n")

    # Save Markdown
    md = "\n".join(md_parts) + "\n"
    out_md_path.write_text(md, encoding="utf-8")
    return md


__all__ = ["AnalyzeOptions", "analyze_csv_to_markdown"]
