"""
Phase 10 — Pipeline audit: seed check, hash verification, vocab guard.
Usage: python scripts/pipeline_audit.py [--mode=full|hashes|seeds]
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

np.random.seed(42)

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


GOLD_ARTIFACTS = [
    "data/gold/region_profiles_gold.parquet",
]
PROFILE_GLOB = "profiles/*.pdf"
MANUSCRIPT_PDF = "manuscript/main.pdf"
EXPECTED_HASHES_FILE = ROOT / "expected_hashes.json"


def sha256_file(path: Path) -> str:
    """Content hash for reproducibility checking.

    Parquet files are hashed on their canonical data content (a deterministic
    CSV serialization), not raw file bytes -- pyarrow embeds non-data metadata
    (e.g. a write timestamp) that differs between two runs of the identical
    pipeline with the identical seed, which made raw-byte hashing falsely FAIL
    on runs that produced numerically identical output (verified directly:
    re-running scripts/p4_pca_scores.py twice in a row on unchanged inputs
    produced two files with different sha256 but zero differing cell values).
    Everything else (PDFs, etc.) is hashed on raw bytes as before.
    """
    if path.suffix == ".parquet":
        import pandas as pd
        df = pd.read_parquet(path)
        content = df.to_csv(index=True).encode("utf-8")
        return hashlib.sha256(content).hexdigest()

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def generate_expected_hashes() -> dict:
    """Compute sha256 for every tracked artifact and write expected_hashes.json.

    Run this once a pipeline output is considered a trusted reference (e.g.
    after a full `make reproduce` run whose results have been reviewed) --
    every subsequent `make audit` / --mode=hashes run then verifies the
    current artifacts still match this frozen reference.
    """
    import glob as glob_mod

    targets = list(GOLD_ARTIFACTS)
    targets += sorted(str(Path(p).relative_to(ROOT)) for p in glob_mod.glob(str(ROOT / PROFILE_GLOB)))
    targets += [MANUSCRIPT_PDF]

    hashes = {}
    missing = []
    for rel_path in targets:
        p = ROOT / rel_path
        if p.exists():
            hashes[rel_path] = sha256_file(p)
        else:
            missing.append(rel_path)

    EXPECTED_HASHES_FILE.write_text(json.dumps(hashes, indent=2, sort_keys=True))
    return {"written": str(EXPECTED_HASHES_FILE), "n_hashed": len(hashes), "missing": missing}


def run_hash_check() -> dict:
    if not EXPECTED_HASHES_FILE.exists():
        return {"status": "SKIP", "reason": "expected_hashes.json not yet generated"}

    expected = json.loads(EXPECTED_HASHES_FILE.read_text())
    results = {}
    all_match = True
    for rel_path, expected_hash in expected.items():
        p = ROOT / rel_path
        if not p.exists():
            results[rel_path] = {"status": "MISSING"}
            all_match = False
        else:
            actual = sha256_file(p)
            match = actual == expected_hash
            results[rel_path] = {"status": "MATCH" if match else "MISMATCH", "actual": actual[:16]}
            if not match:
                all_match = False

    return {"status": "PASS" if all_match else "FAIL", "artifacts": results}


def run_seed_audit() -> dict:
    """Delegate to seed_check.py and capture output."""
    import subprocess
    scripts_dir = ROOT / "scripts"
    result = subprocess.run(
        [sys.executable, str(ROOT / "src/utils/seed_check.py"), str(scripts_dir)],
        capture_output=True, text=True
    )
    return {
        "status": "PASS" if result.returncode == 0 else "FAIL",
        "violations": result.stderr.strip().splitlines() if result.returncode != 0 else [],
    }


def run_vocab_guard() -> dict:
    from src.utils.vocab_guard import check_file, VocabViolation
    import glob

    violations = []
    for pattern in ["analysis/*.md", "profiles/*.md", "manuscript/main.md"]:
        for fp in glob.glob(str(ROOT / pattern)):
            try:
                check_file(fp)
            except VocabViolation as e:
                violations.append(str(e))

    return {
        "status": "PASS" if not violations else "FAIL",
        "violations": violations,
    }


def main(mode: str = "full") -> None:
    report = {}

    if mode in ("full", "hashes"):
        report["hash_check"] = run_hash_check()

    if mode in ("full", "seeds"):
        report["seed_audit"] = run_seed_audit()

    if mode == "full":
        report["vocab_guard"] = run_vocab_guard()

    all_pass = all(
        v.get("status") in ("PASS", "SKIP")
        for v in report.values()
    )
    report["overall_status"] = "PASS" if all_pass else "FAIL"

    output = ROOT / "pipeline_audit.json"
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["full", "hashes", "seeds", "generate-hashes"], default="full")
    args = parser.parse_args()
    if args.mode == "generate-hashes":
        result = generate_expected_hashes()
        print(json.dumps(result, indent=2))
        sys.exit(0)
    main(args.mode)
