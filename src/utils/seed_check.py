"""
AST-based seed auditor — scans Python scripts to verify all random_state usages
are explicitly set to 42 and np.random.seed(42) is called at module top-level.
"""

import ast
import json
import sys
from pathlib import Path


REQUIRED_SEED = 42

# sklearn estimator parameter names that accept a random seed
RANDOM_STATE_PARAMS = {"random_state", "seed"}

# Top-level seed call patterns we accept
SEED_CALL_PATTERNS = [
    ("np.random.seed", REQUIRED_SEED),
    ("numpy.random.seed", REQUIRED_SEED),
    ("random.seed", REQUIRED_SEED),
]


class SeedViolation(ValueError):
    pass


def _extract_seed_calls(tree: ast.AST) -> list[dict]:
    """Find all top-level seed() calls and return their metadata."""
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func_str = ast.unparse(node.func) if hasattr(ast, "unparse") else ""
            for pattern, _ in SEED_CALL_PATTERNS:
                if pattern in func_str:
                    args = [ast.literal_eval(a) for a in node.args if isinstance(a, ast.Constant)]
                    found.append({"call": func_str, "args": args, "lineno": node.lineno})
    return found


def _extract_random_state_usages(tree: ast.AST) -> list[dict]:
    """Find all keyword arguments named random_state or seed in function calls."""
    usages = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            for kw in node.keywords:
                if kw.arg in RANDOM_STATE_PARAMS:
                    try:
                        val = ast.literal_eval(kw.value)
                    except Exception:
                        val = None
                    usages.append({
                        "lineno": node.lineno,
                        "param": kw.arg,
                        "value": val,
                        "call": ast.unparse(node.func) if hasattr(ast, "unparse") else "?",
                    })
    return usages


def audit_file(path: str | Path) -> dict:
    """Audit a single Python file. Returns audit result dict."""
    p = Path(path)
    source = p.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source, filename=str(p))
    except SyntaxError as e:
        return {"file": str(p), "parse_error": str(e), "passed": False}

    seed_calls = _extract_seed_calls(tree)
    rs_usages = _extract_random_state_usages(tree)

    violations = []
    for u in rs_usages:
        if u["value"] != REQUIRED_SEED:
            violations.append(
                f"  Line {u['lineno']}: {u['call']}({u['param']}={u['value']!r}) — "
                f"expected {REQUIRED_SEED}"
            )

    # Only require np.random.seed if the file uses randomness at all
    has_seed_call = any(c["args"] == [REQUIRED_SEED] for c in seed_calls)
    needs_seed_call = len(rs_usages) > 0 or any(
        kw in source for kw in ["np.random", "random.shuffle", "random.choice"]
    )

    if needs_seed_call and not has_seed_call:
        violations.append(f"  Missing np.random.seed({REQUIRED_SEED}) at module level")

    return {
        "file": str(p),
        "seed_calls": seed_calls,
        "random_state_usages": rs_usages,
        "violations": violations,
        "passed": len(violations) == 0,
    }


def audit_directory(directory: str | Path, pattern: str = "**/*.py") -> dict:
    """Audit all Python files in a directory."""
    d = Path(directory)
    results = {}
    for fp in sorted(d.glob(pattern)):
        if ".venv" in fp.parts or "__pycache__" in fp.parts:
            continue
        results[str(fp)] = audit_file(fp)
    return results


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "."
    p = Path(target)
    if p.is_file():
        result = audit_file(p)
        results = {str(p): result}
    else:
        results = audit_directory(p)

    manifest = {}
    all_passed = True
    for fp, res in results.items():
        manifest[fp] = res.get("random_state_usages", [])
        if not res["passed"]:
            all_passed = False
            print(f"FAIL: {fp}", file=sys.stderr)
            for v in res.get("violations", []):
                print(v, file=sys.stderr)

    print(json.dumps(manifest, indent=2))
    sys.exit(0 if all_passed else 1)
