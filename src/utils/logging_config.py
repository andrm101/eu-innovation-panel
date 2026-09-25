"""
Structured JSON logging for all pipeline steps.
Every log entry includes: step_id, input_hash, output_hash, runtime_ms,
random_seed_used, timestamp.
"""

import hashlib
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import structlog


def configure_logging(log_level: str = "INFO") -> None:
    """Call once at the top of any pipeline script."""
    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.BoundLogger,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=False,
    )


def file_sha256(path: str | Path) -> str:
    """Compute SHA-256 hash of a file, returning first 16 hex chars."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def dataframe_hash(df: Any) -> str:
    """Stable hash of a pandas DataFrame shape + column names + first/last row."""
    import pandas as pd

    if not isinstance(df, pd.DataFrame):
        return "non-dataframe"
    sig = f"{df.shape}|{list(df.columns)}|{df.iloc[0].tolist() if len(df) > 0 else []}|{df.iloc[-1].tolist() if len(df) > 0 else []}"
    return hashlib.sha256(sig.encode()).hexdigest()[:16]


@contextmanager
def pipeline_step(
    step_id: str,
    random_seed: int = 42,
    input_artifact: str | Path | None = None,
):
    """
    Context manager that wraps a pipeline step, logging entry/exit with timing.

    Usage:
        with pipeline_step("P2.T9", input_artifact="data/silver/...") as log:
            # do work
            log.info("unit_norm_complete", rows_processed=240)
    """
    configure_logging()
    logger = structlog.get_logger(step_id)

    input_hash = file_sha256(input_artifact) if input_artifact and Path(input_artifact).exists() else "N/A"
    start = time.perf_counter()

    logger.info(
        "step_start",
        step_id=step_id,
        input_hash=input_hash,
        random_seed_used=random_seed,
    )

    bound = logger.bind(step_id=step_id)
    try:
        yield bound
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        bound.info("step_complete", runtime_ms=elapsed_ms, status="ok")
    except Exception as exc:
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        bound.error("step_failed", runtime_ms=elapsed_ms, error=str(exc))
        raise
