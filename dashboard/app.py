"""
Phase 7 — Interactive Dashboard entry point (stub).
Launch with: uv run python dashboard/app.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import dash
from dash import html

app = dash.Dash(
    __name__,
    use_pages=True,
    title="EU Regional Innovation Ecosystems — Exploratory Analysis — No Rankings or Recommendations",
)

app.layout = html.Div([
    html.H1(
        "EU Regional Innovation Ecosystems — Exploratory Analysis — No Rankings or Recommendations",
        style={"fontFamily": "sans-serif", "padding": "12px", "background": "#f5f5f5"},
    ),
    html.P(
        "Dashboard not yet implemented. Complete Phases 1–6 first.",
        style={"padding": "12px"},
    ),
    dash.page_container,
])

if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    app.run(
        host=os.getenv("DASH_HOST", "localhost"),
        port=int(os.getenv("DASH_PORT", 8050)),
        debug=os.getenv("DASH_DEBUG", "false").lower() == "true",
    )
