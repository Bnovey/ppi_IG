# Build: docker build -t igv docker/
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

WORKDIR /app
COPY .. /app
RUN python3 -m pip install --no-cache-dir -e ".[dev]"

CMD ["python3", "-m", "pytest", "tests/", "-v"]
