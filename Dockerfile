FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ src/

# Create config directories and receipt storage
RUN mkdir -p /etc/cognigate/profiles /etc/cognigate/plugins/sinks /var/lib/cognigate/receipts

# Create non-root user for security
RUN groupadd -g 1000 cognigate && \
    useradd -m -u 1000 -g cognigate cognigate && \
    chown -R cognigate:cognigate /etc/cognigate /app /var/lib/cognigate

# Set Python path
ENV PYTHONPATH=/app/src

# Default environment variables
ENV COGNIGATE_HOST=0.0.0.0
ENV COGNIGATE_PORT=8000
ENV COGNIGATE_CONFIG_DIR=/etc/cognigate
ENV COGNIGATE_PLUGINS_DIR=/etc/cognigate/plugins
ENV COGNIGATE_PROFILES_DIR=/etc/cognigate/profiles
ENV COGNIGATE_STANDALONE_MODE=true
ENV COGNIGATE_RECEIPT_STORAGE_DIR=/var/lib/cognigate/receipts

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/health')" || exit 1

# Switch to non-root user
USER cognigate

# Run the application
CMD ["python", "-m", "cognigate.main"]
