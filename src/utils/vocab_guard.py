"""
Vocabulary guard — blocks forbidden causal/evaluative language from all artifacts.
Import and call check() or check_file() from any pipeline step before writing to disk.
"""

import re
import sys
from pathlib import Path

FORBIDDEN_PATTERNS: list[tuple[str, str]] = [
    (r"\bbest\s+region\b", "evaluative: 'best region'"),
    (r"\boptimal\s+location\b", "evaluative: 'optimal location'"),
    (r"\btop\s+region\b", "evaluative: 'top region'"),
    (r"\branked\b", "evaluative: 'ranked'"),
    (r"\bcauses\b", "causal: 'causes'"),
    (r"\beffect\s+of\s+\w+\s+on\b", "causal: 'effect of X on'"),
    (r"\bdriven\s+by\b", "causal: 'driven by'"),
    (r"\bleads\s+to\b", "causal: 'leads to'"),
    (r"\brecommend\s+locat", "prescriptive: 'recommend locating'"),
    (r"\bideal\s+for\b", "prescriptive: 'ideal for'"),
    (r"\bsuitable\s+for\s+investment\b", "prescriptive: 'suitable for investment'"),
    (r"\bsite\s+selection\s+score\b", "prohibited output type: 'site selection score'"),
    (r"\bcomposite\s+ranking\b", "prohibited output type: 'composite ranking'"),
    (r"\bwe\s+recommend\b", "prescriptive: 'we recommend'"),
    (r"\bshould\s+locate\b", "prescriptive: 'should locate'"),
    (r"\bpredict(?:s|ed|ion)?\s+(?!.*exploratory|.*descriptive)", "causal: predictive claim"),
]


class VocabViolation(ValueError):
    pass


def check(text: str, source_hint: str = "<string>") -> None:
    """Raise VocabViolation if text contains any forbidden pattern."""
    violations: list[str] = []
    for pattern, label in FORBIDDEN_PATTERNS:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            violations.append(f"  [{label}] matched: {matches[:3]}")
    if violations:
        msg = f"VocabGuard violations in {source_hint}:\n" + "\n".join(violations)
        raise VocabViolation(msg)


def check_file(path: str | Path) -> None:
    """Read a file and run check() on its full contents."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"vocab_guard: file not found: {p}")
    text = p.read_text(encoding="utf-8", errors="replace")
    check(text, source_hint=str(p))


def check_dataframe_strings(df, source_hint: str = "<dataframe>") -> None:
    """Check all string columns of a pandas DataFrame."""
    import pandas as pd

    for col in df.select_dtypes(include="object").columns:
        for val in df[col].dropna().astype(str):
            check(val, source_hint=f"{source_hint}[{col}]")


if __name__ == "__main__":
    # CLI usage: python vocab_guard.py <file_path>
    if len(sys.argv) < 2:
        print("Usage: python vocab_guard.py <file_path>")
        sys.exit(1)
    try:
        check_file(sys.argv[1])
        print(f"vocab_guard: PASS — {sys.argv[1]}")
    except VocabViolation as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)
