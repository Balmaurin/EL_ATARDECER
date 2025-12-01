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
COPY apps ./apps
COPY data ./data
COPY scripts ./scripts
COPY models ./models
COPY config ./config

# Set PYTHONPATH
ENV PYTHONPATH=/app

# Create initialization script for ADK and Federation Server
RUN echo '#!/bin/bash\n\
    echo "=== EL-AMANECER V4 Container Startup ==="\n\
    echo "🎯 Google ADK Integration: $ADK_ENABLED"\n\
    echo "🤖 HACK-MEMORI Auto Mode: $HACK_MEMORI_AUTO_MODE"\n\
    echo "🚀 Starting Federation Gateway on port 8000..."\n\
    uvicorn apps.backend.run_federation:app --host 0.0.0.0 --port 8000' > /app/startup.sh && chmod +x /app/startup.sh

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# Run the server via startup script
CMD ["/app/startup.sh"]
