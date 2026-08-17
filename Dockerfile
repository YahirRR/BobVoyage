# BobVoyage — Mission Control MCP Server
# Exposes the full BobVoyage intelligence pipeline via Streamable HTTP MCP.
#
# Build:
#   docker build -t bobvoyage .
#
# Run (demo mode — no internet required):
#   docker run -p 8080:8080 -e BOBVOYAGE_DATA_PROVIDER=local bobvoyage
#
# Run (live mode — NOAA + NASA DONKI):
#   docker run -p 8080:8080 \
#     -e BOBVOYAGE_DATA_PROVIDER=noaa \
#     -e BOBVOYAGE_NASA_API_KEY=<your-key> \
#     bobvoyage
#
# MCP endpoint : http://localhost:8080/mcp
# Health check : http://localhost:8080/health
#
# Security note: this image does not include authentication.
# Place behind an authenticated reverse proxy (nginx, Caddy, ALB, etc.)
# and serve over HTTPS in production. Never embed secrets in the image.

FROM python:3.11-slim

# Metadata
LABEL org.opencontainers.image.title="BobVoyage"
LABEL org.opencontainers.image.description="Space Weather Intelligence MCP Server"
LABEL org.opencontainers.image.licenses="MIT"

# Working directory
WORKDIR /app

# Install Python dependencies first (layer-cached separately from source)
COPY requirements.txt pyproject.toml ./
# Install without dev extras (no pytest in production image)
RUN pip install --no-cache-dir \
    "mcp>=1.0.0,<2.0.0" \
    "pandas>=2.0.0" \
    "fastapi>=0.110.0" \
    "uvicorn>=0.29.0" \
    "starlette>=0.36.0"

# Copy source code and data
COPY src/ ./src/
COPY data/ ./data/

# Install the package in non-editable mode
RUN pip install --no-cache-dir --no-deps -e .

# Runtime environment defaults
ENV BOBVOYAGE_DATA_PROVIDER=local
ENV BOBVOYAGE_MCP_HOST=0.0.0.0
ENV BOBVOYAGE_MCP_PORT=8080
ENV BOBVOYAGE_MCP_PATH=/mcp

# Expose the MCP HTTP port
EXPOSE 8080

# Run the Streamable HTTP MCP server
CMD ["uvicorn", "bobvoyage.mcp_http:app", \
     "--host", "0.0.0.0", \
     "--port", "8080", \
     "--workers", "1"]
