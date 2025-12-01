FROM python:3.11

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY packages ./packages
COPY apps/consciousness_server ./apps/consciousness_server
COPY data ./data
COPY tools ./tools
COPY config ./config

# Set PYTHONPATH
ENV PYTHONPATH=/app

# Expose port
EXPOSE 8001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8001/api/consciousness/health || exit 1

# Run the consciousness API server
CMD ["python", "-m", "uvicorn", "apps.consciousness_server.main:app", "--host", "0.0.0.0", "--port", "8001"]

