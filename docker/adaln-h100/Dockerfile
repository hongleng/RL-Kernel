FROM ghcr.io/astral-sh/uv:0.13.4 AS uv
FROM nvidia/cuda:12.6.3-devel-ubuntu22.04

COPY --from=uv /uv /usr/local/bin/uv
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates gcc-11 g++-11 git libnuma1 openssh-server tmux \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/gcc-11 /usr/local/bin/gcc \
    && ln -s /usr/bin/g++-11 /usr/local/bin/g++ \
    && ln -s /usr/bin/g++-11 /usr/local/bin/c++
ENV UV_PYTHON_INSTALL_DIR=/opt/python
RUN uv python install 3.11.17 && uv venv --python 3.11.17 /opt/venv
ENV PATH=/opt/venv/bin:/usr/local/cuda-12.6/bin:${PATH}
ENV CC=gcc-11 CXX=g++-11 CUDA_HOME=/usr/local/cuda-12.6
COPY requirements.txt /opt/image-requirements.txt
ENV UV_INDEX_URL=https://pypi.org/simple \
    UV_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu126 \
    UV_INDEX_STRATEGY=unsafe-best-match UV_LINK_MODE=copy
# Separate large packages so a failed pull retries one smaller layer.
# Cache mounts retain build downloads without shipping the uv cache.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /opt/venv/bin/python --no-deps \
    -c /opt/image-requirements.txt nvidia-cublas-cu12
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /opt/venv/bin/python --no-deps \
    -c /opt/image-requirements.txt nvidia-cudnn-cu12
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /opt/venv/bin/python --no-deps \
    -c /opt/image-requirements.txt nvidia-cufft-cu12 nvidia-curand-cu12 nvidia-cusolver-cu12
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /opt/venv/bin/python --no-deps \
    -c /opt/image-requirements.txt nvidia-cusparse-cu12 nvidia-cusparselt-cu12
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /opt/venv/bin/python --no-deps \
    -c /opt/image-requirements.txt nvidia-nccl-cu12 nvidia-nvshmem-cu12
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /opt/venv/bin/python --no-deps \
    -c /opt/image-requirements.txt torch
# Resolve and check the complete pinned environment after staged installs.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /opt/venv/bin/python -r /opt/image-requirements.txt

WORKDIR /opt/RL-Kernel
COPY . .
RUN python -c 'import json,sys,torch,triton; e=json.load(open("acceptance/environment.json")); assert sys.version_info[:3] == tuple(map(int,e["python"].split("."))); assert torch.__version__ == e["torch"] and torch.version.cuda == e["cuda"]; assert triton.__version__ == e["triton"]; from triton.backends.nvidia.compiler import CUDAOptions; assert "enable_reflect_ftz" in CUDAOptions.__dataclass_fields__' \
    && nvcc --version && compute-sanitizer --version && ninja --version \
    && uv pip check --python /opt/venv/bin/python \
    && python acceptance/selfcheck.py \
    && python -m compileall -q acceptance \
    && python -m pytest tests/validation/operators/test_adaln_gtest.py -q -k independent_gold_uses_contract_token_order \
    && uv pip freeze --python /opt/venv/bin/python > /opt/image-packages.txt \
    && mkdir -p /run/sshd /workspace

# The user runs acceptance on H100 from the RunPod terminal.
CMD ["sleep", "infinity"]
