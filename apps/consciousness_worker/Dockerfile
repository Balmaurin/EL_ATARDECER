FROM python:3.11

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install additional dependencies for Dapr and consciousness
RUN pip install --no-cache-dir \
    dapr \
    fastapi \
    uvicorn \
    redis

# Copy project files
COPY packages ./packages
COPY apps/consciousness_worker ./apps/consciousness_worker
COPY data ./data

# Set PYTHONPATH
ENV PYTHONPATH=/app

# Expose port
EXPOSE 8002

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8002/health || exit 1

# Run the consciousness worker
CMD ["python", "-m", "uvicorn", "apps.consciousness_worker.main:app", "--host", "0.0.0.0", "--port", "8002"]

