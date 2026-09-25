.PHONY: setup test pipeline dashboard reproduce audit clean help

PYTHON := python
UV := uv
SCRIPTS := scripts

help:
	@echo "EU Regional Innovation Panel — available targets:"
	@echo "  make setup      Install dependencies and pre-commit hooks"
	@echo "  make test       Run unit tests"
	@echo "  make pipeline   Run full data pipeline (Phases 1-6)"
	@echo "  make dashboard  Launch Plotly Dash app on localhost:8050"
	@echo "  make reproduce  Full clean-room pipeline run (requires Docker)"
	@echo "  make audit      Seed check + hash verification + vocab guard"
	@echo "  make clean      Remove generated artifacts (keeps raw data)"

# ── Setup ─────────────────────────────────────────────────────────────────────
setup:
	$(UV) sync
	$(UV) run pre-commit install
	@echo "Setup complete. Activate env with: source .venv/bin/activate (Linux/Mac)"
	@echo "On Windows: .venv\\Scripts\\activate"

# ── Tests ─────────────────────────────────────────────────────────────────────
test:
	$(UV) run pytest -v

# ── Pipeline phases ───────────────────────────────────────────────────────────
pipeline: p1 p2 p3 p4 p5 p6
	@echo "Pipeline complete."

p1:
	@echo "=== Phase 1: Data Inventory ==="
	$(UV) run $(PYTHON) $(SCRIPTS)/p1_data_inventory.py

p2:
	@echo "=== Phase 2: Ingestion & Harmonization ==="
	$(UV) run $(PYTHON) $(SCRIPTS)/p2_ingest_harmonize.py

p3:
	@echo "=== Phase 3: EDA ==="
	$(UV) run $(PYTHON) $(SCRIPTS)/p3_eda.py

p4:
	@echo "=== Phase 4: Dimensionality Reduction ==="
	$(UV) run $(PYTHON) $(SCRIPTS)/p4_pca_scores.py

p5:
	@echo "=== Phase 5: Clustering ==="
	$(UV) run $(PYTHON) $(SCRIPTS)/p5_clustering.py

p6:
	@echo "=== Phase 6: Barriers & Enablers ==="
	$(UV) run $(PYTHON) $(SCRIPTS)/p6_barriers_enablers.py

p7-profiles:
	@echo "=== Phase 8: Regional Profiles ==="
	$(UV) run $(PYTHON) $(SCRIPTS)/p8_render_profiles.py

# ── Dashboard ─────────────────────────────────────────────────────────────────
dashboard:
	@echo "Launching dashboard on http://localhost:8050"
	$(UV) run $(PYTHON) dashboard/app.py

# ── Reproducibility ───────────────────────────────────────────────────────────
# All of these are bind-mounted (not just data/raw + outputs) because the
# pipeline scripts write directly to repo-relative paths (data/bronze,
# data/silver, data/gold, analysis, figures, profiles, manuscript, reports) --
# an earlier version of this target only mounted data/raw and an unused
# outputs/ directory nothing writes to, so `docker run --rm` silently
# discarded every actual pipeline artifact on container exit.
reproduce:
	docker build -t eu-innov:latest .
	docker run --rm \
		-v "$$(pwd)/data/raw:/app/data/raw:ro" \
		-v "$$(pwd)/data/bronze:/app/data/bronze" \
		-v "$$(pwd)/data/silver:/app/data/silver" \
		-v "$$(pwd)/data/gold:/app/data/gold" \
		-v "$$(pwd)/analysis:/app/analysis" \
		-v "$$(pwd)/figures:/app/figures" \
		-v "$$(pwd)/profiles:/app/profiles" \
		-v "$$(pwd)/manuscript:/app/manuscript" \
		-v "$$(pwd)/reports:/app/reports" \
		eu-innov:latest make pipeline

# ── Audit ─────────────────────────────────────────────────────────────────────
audit:
	@echo "=== Seed audit ==="
	$(UV) run $(PYTHON) src/utils/seed_check.py scripts/
	@echo "=== Schema import check ==="
	$(UV) run $(PYTHON) -c "from src.contracts.silver_schema import silver_schema; from src.contracts.gold_schema import gold_schema; print('schemas: OK')"
	@echo "=== Vocab guard on analysis artifacts ==="
	$(UV) run $(PYTHON) -c "\
from src.utils.vocab_guard import check_file; \
import glob; \
[check_file(f) for f in glob.glob('analysis/*.md') + glob.glob('profiles/*.md')]; \
print('vocab_guard: OK')"
	@echo "=== Hash verification ==="
	$(UV) run $(PYTHON) scripts/pipeline_audit.py --mode=hashes
	@echo "Audit complete."

# ── Clean ─────────────────────────────────────────────────────────────────────
clean:
	rm -rf data/bronze/ data/silver/ data/gold/ data/processed/
	rm -rf figures/eda/ figures/clustering/
	rm -rf analysis/*.csv analysis/*.json analysis/*.md
	rm -rf profiles/*.md profiles/*.pdf profiles/*.json
	rm -rf manuscript/main.pdf
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@echo "Clean complete. Raw data in data/raw/ preserved."
