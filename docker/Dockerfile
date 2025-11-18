FROM python:3.13-slim

LABEL org.opencontainers.image.source=https://github.com/OpenMined/beaver
LABEL org.opencontainers.image.description="A library for data visitation and eager execution of remote bioinformatics workflows."
LABEL org.opencontainers.image.licenses=Apache-2.0

# Install uv for fast package management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Set working directory
WORKDIR /app

# Copy Python package files
COPY python/pyproject.toml python/README.md ./
COPY python/src ./src

# Install beaver
RUN uv pip install --system --no-cache .

# Set the default command to show help
CMD ["beaver", "--help"]
