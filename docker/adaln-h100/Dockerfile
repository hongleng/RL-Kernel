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
RUN uv pip install --python /opt/venv/bin/python \
    --index-url https://pypi.org/simple \
    --extra-index-url https://download.pytorch.org/whl/cu126 \
    --index-strategy unsafe-best-match \
    torch==2.12.1+cu126 triton==3.7.1 pytest==9.1.1 ninja==1.13.2 \
    setuptools==78.1.0 numpy==2.3.5 packaging==26.3 tabulate==0.9.0 wheel==0.45.1

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
