FROM python:3.11-slim

# Metadata
LABEL maintainer="openenv-email-triage"
LABEL version="1.0.0"
LABEL description="Email Triage OpenEnv — real-world email management environment"

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Create empty __init__ files for package imports
RUN touch data/__init__.py tasks/__init__.py tests/__init__.py

# Expose port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Default: run the API server
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
