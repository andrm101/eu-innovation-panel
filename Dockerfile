# Multi-stage build: builder installs Python deps, runtime is lean
# Pins: Python 3.11-slim, GDAL 3.8+, PROJ 9.3+, pandoc 3.1+

# ── Stage 1: system dependencies + GDAL ──────────────────────────────────────
FROM ghcr.io/osgeo/gdal:ubuntu-small-3.8.4 AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Ubuntu 22.04's own jammy-updates repo only carries python3.11 3.11.0~rc1
# (a pre-release build), which fails this project's requires-python = ">=3.11"
# check under PEP 440 ordering (an rc sorts before its final release). The
# deadsnakes PPA carries the actual final 3.11.x release. Adding the PPA
# manually (curl + gpg --dearmor) rather than via add-apt-repository, since
# this minimal base image lacks gpg-agent and add-apt-repository's key-import
# path fails without it.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates gnupg curl \
    && curl -fsSL https://keyserver.ubuntu.com/pks/lookup?op=get\&search=0xBA6932366A755776 \
       -o /tmp/deadsnakes.asc \
    && gpg --dearmor -o /etc/apt/trusted.gpg.d/deadsnakes.gpg /tmp/deadsnakes.asc \
    && rm /tmp/deadsnakes.asc \
    && echo "deb https://ppa.launchpadcontent.net/deadsnakes/ppa/ubuntu jammy main" \
       > /etc/apt/sources.list.d/deadsnakes.list \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3-pip \
    python3.11-venv \
    libspatialindex-dev \
    libpq-dev \
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

# The builder stage's `uv` binary (installed via pip into that stage's system
# Python) isn't part of /app/.venv, so it doesn't survive the COPY below --
# every Makefile target invokes $(UV) run ..., so runtime needs its own uv too.
RUN pip install --no-cache-dir uv==0.1.42

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
