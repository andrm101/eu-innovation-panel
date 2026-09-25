# Multi-stage build: builder installs Python deps, runtime is lean
# Pins: Python 3.11-slim, GDAL 3.8+, PROJ 9.3+, pandoc 3.1+

# ── Stage 1: system dependencies + GDAL ──────────────────────────────────────
FROM osgeo/gdal:ubuntu-small-3.8.4 AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3-pip \
    python3.11-venv \
    libspatialindex-dev \
    libpq-dev \
    curl \
    wget \
    ripgrep \
    make \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install pandoc 3.1.x from GitHub release (apt version is too old)
RUN wget -q https://github.com/jgm/pandoc/releases/download/3.1.13/pandoc-3.1.13-1-amd64.deb \
    -O /tmp/pandoc.deb \
    && dpkg -i /tmp/pandoc.deb \
    && rm /tmp/pandoc.deb

# Install TeX for PDF rendering
RUN apt-get update && apt-get install -y --no-install-recommends \
    texlive-xetex \
    texlive-fonts-recommended \
    texlive-latex-extra \
    && rm -rf /var/lib/apt/lists/*

# ── Stage 2: Python dependencies ─────────────────────────────────────────────
FROM base AS builder

# Install uv for fast, reproducible installs
RUN pip install --no-cache-dir uv==0.1.42

WORKDIR /app
COPY pyproject.toml ./
# uv sync installs from lockfile if present, otherwise resolves fresh
COPY uv.lock* ./
RUN uv venv /app/.venv --python python3.11 \
    && uv pip install --python /app/.venv/bin/python -e ".[dev]"

# ── Stage 3: runtime ──────────────────────────────────────────────────────────
FROM base AS runtime

WORKDIR /app

# Copy venv from builder
COPY --from=builder /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"

# Copy project source
COPY . .

# Verify key tools
RUN python -c "import pandas, geopandas, pandera, sklearn, plotly, dash, esda; print('deps OK')" \
    && pandoc --version | head -1 \
    && gdalinfo --version

EXPOSE 8050

CMD ["make", "pipeline"]
