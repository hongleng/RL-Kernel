FROM ghcr.io/hongleng/rl-kernel/adaln-h100@sha256:682fae9afbb852c75c7b519e9b80645e6528c0d8d034d29ed4bd35f0a5073ce4 AS runtime

RUN uv pip install --python /opt/venv/bin/python \
    --index-url https://pypi.org/simple --no-deps tzdata==2026.5

# Reuse the pinned, split dependency layers; keep old shared source outside imports.
WORKDIR /opt/RL-Kernel-indexed
COPY . .
RUN mkdir -p /opt/adaln-indexed-image-checks \
    && python image-acceptance/selfcheck.py \
    && nvcc --version && compute-sanitizer --version && ninja --version \
    && uv pip check --python /opt/venv/bin/python \
    && python -m pytest .cache/adaln-indexed/test_benchmark_adaln_indexed.py \
       .cache/adaln-indexed/test_ticket0607.py tests/reference/test_adaln_indexed.py \
       tests/runtime/test_adaln_strict_policy.py -q \
       -k 'not explicit_indexed_activation_resolves_without_enabling_cpu_select01 and not activated_strict_rechecks_actual_domain_before_execution and not ordinary_gpu_resolution_rechecks_shared_cpu_device_with_real_fallback' \
       --junitxml=/opt/adaln-indexed-image-checks/cpu.xml \
    && python .cache/adaln-indexed/run_ticket0607.py --prepare \
       --output /opt/adaln-indexed-image-checks/candidate \
    && python image-acceptance/selfcheck.py /opt/adaln-indexed-image-checks/candidate \
    && uv pip freeze --python /opt/venv/bin/python > /opt/adaln-indexed-image-checks/image-packages.txt
CMD ["sleep", "infinity"]

FROM scratch AS evidence
COPY --from=runtime /opt/RL-Kernel-indexed/image-acceptance/ /
COPY --from=runtime /opt/RL-Kernel-indexed/Dockerfile /Dockerfile
COPY --from=runtime /opt/adaln-indexed-image-checks/cpu.xml /cpu.xml
COPY --from=runtime /opt/adaln-indexed-image-checks/image-packages.txt /image-packages.txt
COPY --from=runtime /opt/adaln-indexed-image-checks/candidate/node-lock.json /node-lock.json
COPY --from=runtime /opt/adaln-indexed-image-checks/candidate/performance-plan.json /performance-plan.json
COPY --from=runtime /opt/adaln-indexed-image-checks/candidate/candidate.json /candidate.json
COPY --from=runtime /opt/adaln-indexed-image-checks/candidate/source-sha256.json /source-sha256.json
