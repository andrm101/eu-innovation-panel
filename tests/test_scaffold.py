"""
Phase 0 quality gate tests — all must pass before Phase 1 begins.
Run: uv run pytest tests/test_scaffold.py -v
"""

import importlib
from pathlib import Path

ROOT = Path(__file__).parent.parent


# ── Importability tests ────────────────────────────────────────────────────────

def test_vocab_guard_importable():
    from src.utils.vocab_guard import check, check_file, VocabViolation
    assert callable(check)
    assert callable(check_file)


def test_vocab_guard_blocks_forbidden_token():
    from src.utils.vocab_guard import check, VocabViolation
    import pytest
    with pytest.raises(VocabViolation):
        check("This is the best region for investment.")


def test_vocab_guard_passes_clean_text():
    from src.utils.vocab_guard import check
    # Should not raise
    check(
        "R&D expenditure is associated with patent density in this archetype. "
        "This is an exploratory, non-causal analysis."
    )


def test_seed_check_importable():
    from src.utils.seed_check import audit_file, audit_directory
    assert callable(audit_file)


def test_logging_config_importable():
    from src.utils.logging_config import configure_logging, pipeline_step
    assert callable(configure_logging)


def test_bronze_schema_importable():
    from src.contracts.bronze_schema import eurostat_long_bronze, bronze_manifest_schema
    assert eurostat_long_bronze.name == "eurostat_long_bronze"


def test_silver_schema_importable():
    from src.contracts.silver_schema import silver_schema
    assert silver_schema.name == "silver_layer"
    # Primary key is nuts2_code
    assert silver_schema.index.name == "nuts2_code"


def test_gold_schema_importable():
    from src.contracts.gold_schema import gold_schema
    assert gold_schema.name == "gold_layer"
    assert "archetype_id" in gold_schema.columns
    assert "score_cost" in gold_schema.columns


# ── File existence tests ───────────────────────────────────────────────────────

def test_pyproject_exists():
    assert (ROOT / "pyproject.toml").exists()


def test_dockerfile_exists():
    assert (ROOT / "Dockerfile").exists()


def test_makefile_exists():
    assert (ROOT / "Makefile").exists()


def test_pre_commit_config_exists():
    assert (ROOT / ".pre-commit-config.yaml").exists()


def test_claude_md_exists():
    assert (ROOT / "CLAUDE.md").exists()


def test_env_example_exists():
    assert (ROOT / ".env.example").exists()


def test_ci_workflow_exists():
    assert (ROOT / ".github" / "workflows" / "ci.yml").exists()


def test_contracts_directory_has_three_schemas():
    contracts = ROOT / "src" / "contracts"
    schemas = list(contracts.glob("*_schema.py"))
    assert len(schemas) == 3, f"Expected 3 schema files, found {len(schemas)}: {schemas}"


def test_data_directory_structure():
    for subdir in ["raw/eurostat", "raw/cordis", "raw/oecd", "raw/epo",
                   "bronze", "silver", "gold"]:
        assert (ROOT / "data" / subdir).exists(), f"Missing: data/{subdir}"


def test_claude_md_contains_forbidden_token_list():
    content = (ROOT / "CLAUDE.md").read_text()
    assert "best region" in content
    assert "causes" in content
    assert "non-causal" in content
    assert "nuts2_code" in content


def test_claude_md_contains_seed_policy():
    content = (ROOT / "CLAUDE.md").read_text()
    assert "seed(42)" in content or "random_state=42" in content
