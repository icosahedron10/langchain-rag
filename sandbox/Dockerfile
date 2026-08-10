# Optional sandbox image for SANDBOX_MODE=docker.
#
# Build:  docker build -t ragchat-sandbox:latest sandbox/
#
# The application starts this image with: no network, read-only root
# filesystem, dropped capabilities, no privilege escalation, a tmpfs /tmp,
# and the per-session workspace mounted read/write at /workspace.
FROM python:3.12-slim

RUN apt-get update \
    && apt-get install --yes --no-install-recommends coreutils \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir matplotlib numpy pandas

RUN groupadd --gid 10001 sandbox \
    && useradd --create-home --uid 10001 --gid 10001 sandbox
USER sandbox

# /tmp is a tmpfs at runtime; keep caches off the read-only root filesystem.
ENV HOME=/tmp \
    MPLCONFIGDIR=/tmp/matplotlib

WORKDIR /workspace
