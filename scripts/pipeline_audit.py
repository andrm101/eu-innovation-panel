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
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


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
    parser.add_argument("--mode", choices=["full", "hashes", "seeds"], default="full")
    args = parser.parse_args()
    main(args.mode)
