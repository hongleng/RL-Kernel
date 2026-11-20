FROM docker.io/pytorch/pytorch@sha256:9859f8978cdfad549d72baa41d0b0bb7a5b46210a1446e09bf32600a968badb8 AS environment
RUN python -m pip install --no-cache-dir pytest==8.3.5 ninja==1.11.1.4 tabulate==0.9.0
ENV PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1

FROM environment AS build
COPY source /opt/task19-source
COPY tools /opt/task19-h100
WORKDIR /opt/task19-source
# One native compilation in the entire image and validation lifecycle.
RUN python -I /opt/task19-h100/run.py preflight --repo /opt/task19-source \
    && python -I /opt/task19-h100/test_run.py \
    && python -I /opt/task19-h100/build_image.py

# Copy only the one binary and its provenance; object files stay in build stage.
FROM environment AS runtime
COPY source /opt/task19-source
COPY tools /opt/task19-h100
COPY --from=build /opt/task19-source/rl_engine/_C*.so /opt/task19-source/rl_engine/
COPY --from=build /opt/task19-image-build /opt/task19-image-build
ENV PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /opt/task19-source
RUN python -I /opt/task19-h100/build_image.py --check
CMD ["sleep", "infinity"]
