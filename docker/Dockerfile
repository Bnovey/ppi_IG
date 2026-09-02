# Build: docker build -t igv -f docker/Dockerfile .   (context = repo root)
# Run:   docker run --gpus all --shm-size=32g --ipc=host igv
FROM nvidia/cuda:12.6.2-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Python 3.11 from deadsnakes; bootstrap pip via get-pip.py because the
# distro python3-pip package is bound to the system python3.10.
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common curl ca-certificates && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-dev python3.11-distutils && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11 && \
    ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    rm -rf /var/lib/apt/lists/*

# torch first (from the cu126 wheel index), then boltz
RUN python3 -m pip install --no-cache-dir \
        torch==2.7.1+cu126 --index-url https://download.pytorch.org/whl/cu126
RUN python3 -m pip install --no-cache-dir boltz==2.2.1

# git is needed at RUNTIME, not just build time: provenance._git_commit shells
# out to `git rev-parse HEAD` from inside the container to stamp every artifact
# sidecar. Without the binary the call fails silently and every artifact
# records git_commit: null -- reproducibility lost with no error raised.
# Deliberately its own layer *after* torch/boltz so editing it does not
# invalidate those multi-GB cached layers.
RUN apt-get update && apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app
RUN python3 -m pip install --no-cache-dir -e ".[dev]"

# The pipeline bind-mounts the host repo over /app, where it is owned by the
# host user while the container runs as root. Git refuses to operate on a
# repo it considers "dubious ownership", which would again yield a null
# git_commit in provenance. Trust /app explicitly.
RUN git config --global --add safe.directory /app

CMD ["python3", "-m", "pytest", "tests/", "-v"]
