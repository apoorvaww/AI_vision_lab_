# ===================================================================
# Dockerfile: AI VisionLab (Cross-Platform / AMD64 & ARM64)
# ===================================================================

FROM python:3.11-slim-bookworm

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOST=0.0.0.0 \
    PORT=5000 \
    FLASK_ENV=production

# Install system dependencies
# - libgomp1: OpenMP runtime required by XGBoost and ML algorithms
# - libglib2.0-0, libsm6, libxext6, libxrender1: graphics and imaging support
# - curl, ca-certificates: container healthchecks and secure model downloading
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Set up application workspace
WORKDIR /app

# Install Python package dependencies first for optimal Docker layer caching
COPY requirements.txt /app/
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

# Create an unprivileged user for security
RUN groupadd -g 10001 appuser && \
    useradd -u 10001 -g appuser -s /bin/bash -m appuser

# Copy application source code
COPY . /app

# Ensure persistent data directories exist and set appropriate permissions
RUN mkdir -p /app/webapp_data/uploads /app/webapp_data/results && \
    chown -R appuser:appuser /app

# Switch to unprivileged user
USER appuser

# Expose the application port
EXPOSE 5000

# Container healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:5000/api/health || exit 1

# Production WSGI server
# Uses 1 worker with multiple threads to share in-memory job coordination safely
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--threads", "4", "--timeout", "180", "wsgi:app"]
