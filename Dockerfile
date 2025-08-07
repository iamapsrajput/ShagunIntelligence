# Multi-stage build for Shagun Intelligence Trading System
FROM python:3.11.9-slim as backend-builder

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="/root/.local/bin:$PATH"

# Copy poetry files
COPY pyproject.toml poetry.lock ./

# Install Python dependencies
RUN poetry config virtualenvs.create false \
    && poetry install --only=main

# Frontend build stage
FROM node:18-alpine as frontend-builder

WORKDIR /app/dashboard

# Copy package files
COPY dashboard/package*.json ./

# Install dependencies
RUN npm ci

# Copy frontend source
COPY dashboard/ ./

# Build frontend
ENV VITE_API_BASE_URL=http://localhost:8000
ENV NODE_OPTIONS="--max-old-space-size=4096"
RUN npm run build

# Final runtime stage
FROM python:3.11.9-slim

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy Python dependencies from builder
COPY --from=backend-builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=backend-builder /usr/local/bin /usr/local/bin

# Copy application code
COPY . /app/

# Copy built frontend
COPY --from=frontend-builder /app/dashboard/dist /app/dashboard/dist

# Create non-root user for security
RUN useradd -m -u 1000 trader && \
    chown -R trader:trader /app && \
    mkdir -p /app/data /app/logs && \
    chown -R trader:trader /app/data /app/logs

USER trader

# Expose ports
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Start command
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
