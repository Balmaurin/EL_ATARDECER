FROM python:3.11

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install llama-cpp-python (requires build tools)
RUN pip install --no-cache-dir \
    llama-cpp-python \
    fastapi \
    uvicorn \
    pydantic

# Copy project files
COPY modelsLLM ./modelsLLM

# Set PYTHONPATH
ENV PYTHONPATH=/app

# Expose port
EXPOSE 8003

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8003/health || exit 1

# Run the LLM service
CMD ["python", "-m", "uvicorn", "apps.llm_service.main:app", "--host", "0.0.0.0", "--port", "8003"]

