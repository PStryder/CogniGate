FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ src/

# Create config directories
RUN mkdir -p /etc/cognigate/profiles /etc/cognigate/plugins/sinks

# Set Python path
ENV PYTHONPATH=/app/src

# Default environment variables
ENV COGNIGATE_HOST=0.0.0.0
ENV COGNIGATE_PORT=8000
ENV COGNIGATE_CONFIG_DIR=/etc/cognigate
ENV COGNIGATE_PLUGINS_DIR=/etc/cognigate/plugins
ENV COGNIGATE_PROFILES_DIR=/etc/cognigate/profiles

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/health')" || exit 1

# Run the application
CMD ["python", "-m", "cognigate.main"]
