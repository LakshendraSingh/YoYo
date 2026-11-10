# YoYo Autonomous Agent Container Image
FROM python:3.11-slim

LABEL maintainer="YoYo Team"
LABEL description="YoYo: Sandboxed Autonomous AI Agent for Non-Interactive Batch Environments"

# Configure non-interactive container runtime
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    CI=true \
    LC_ALL=C.UTF-8 \
    LANG=C.UTF-8

# Install minimal base utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create unprivileged batch worker user
RUN useradd -m -s /bin/bash batchworker
WORKDIR /workspace

# Install yoyo
COPY . /opt/yoyo
RUN pip install --no-cache-dir /opt/yoyo

USER batchworker

ENTRYPOINT ["yoyo"]
CMD ["--help"]
