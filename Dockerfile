# Z-Image Turbo - App image (code seulement)
# Utilise une base image (deps + modele) construite via Dockerfile.base

ARG BASE_IMAGE=swarzox/z-image-turbo-base:latest
FROM ${BASE_IMAGE}

# Make `--model-path ./z-image-turbo` work inside the container by symlinking to the HF cache
# baked in the base image (when present).
RUN set -eu; \
    mkdir -p /app; \
    if [ -e /app/z-image-turbo ]; then \
      echo "/app/z-image-turbo already exists; skipping HF cache symlink."; \
      exit 0; \
    fi; \
    snapshots_dir="/models/models--Tongyi-MAI--Z-Image-Turbo/snapshots"; \
    if [ -d "$snapshots_dir" ]; then \
      snap="$(ls -1 "$snapshots_dir" | head -n 1)"; \
      if [ -n "$snap" ] && [ -d "$snapshots_dir/$snap" ]; then \
        ln -sfn "$snapshots_dir/$snap" /app/z-image-turbo; \
      fi; \
    fi

COPY server.py /app/server.py
COPY worker.py /app/worker.py
COPY entrypoint.sh /app/entrypoint.sh

RUN chmod +x /app/entrypoint.sh

# Vast.ai uses onstart.sh instead of Docker ENTRYPOINT
# Create onstart.sh that calls our entrypoint
RUN echo '#!/bin/bash' > /app/onstart.sh && \
    echo 'exec /app/entrypoint.sh >> /var/log/onstart.log 2>&1' >> /app/onstart.sh && \
    chmod +x /app/onstart.sh

WORKDIR /app
EXPOSE 5000 5001

ENTRYPOINT ["/app/entrypoint.sh"]
