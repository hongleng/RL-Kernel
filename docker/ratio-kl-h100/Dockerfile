FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime

LABEL org.opencontainers.image.source="https://github.com/hongleng/RL-Kernel"

RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/* \
    && python -m pip install --no-cache-dir pytest

WORKDIR /opt/RL-Kernel
COPY . .

CMD ["bash"]
