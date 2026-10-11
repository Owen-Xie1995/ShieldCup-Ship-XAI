# Linux CPU runtime recipe; actual build and test receipts accompany image.tar.
ARG BASE_IMAGE=python:3.12.9-slim-bookworm
FROM ${BASE_IMAGE}

ARG BASE_IMAGE
ARG BUILD_PROFILE=cpu
ARG TARGETPLATFORM
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

LABEL org.opencontainers.image.title="Ship XAI black_occlusion4_soft15" \
      org.opencontainers.image.version="20261006-lock-cpu" \
      ship_xai.base_image="${BASE_IMAGE}" \
      ship_xai.profile="${BUILD_PROFILE}" \
      ship_xai.platform="${TARGETPLATFORM}" \
      ship_xai.torch_index="${TORCH_INDEX_URL}" \
      ship_xai.weights_sha256="eede97896edcb34840d998647cfd9db76be8ad82b3c99db7d3375781ab30e473" \
      ship_xai.wheel_sha256="43986f644c383f255e73dcbd2305e311f15fd70b4aeec54deb346dab5d5cbfd7"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLBACKEND=Agg \
    NVIDIA_TF32_OVERRIDE=0

WORKDIR /opt/submission

# Quantus declares opencv-python; these are its Linux shared-library requirements.
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-core.txt requirements-torch.txt requirements-runtime.txt requirements-lock-runtime.txt ./
RUN python -m pip install --no-cache-dir --only-binary=:all: --no-deps \
      --index-url "${TORCH_INDEX_URL}" \
      --constraint requirements-core.txt --requirement requirements-torch.txt \
    && python -m pip install --no-cache-dir --only-binary=:all: \
      --index-url https://pypi.org/simple \
      --constraint requirements-core.txt --constraint requirements-lock-runtime.txt --requirement requirements-runtime.txt

COPY . /opt/submission/
RUN python -c "from pathlib import Path; import hashlib; pairs=[('checkpoints/best_resnet50.pth','eede97896edcb34840d998647cfd9db76be8ad82b3c99db7d3375781ab30e473'),('dist/kaggle-0.1.0-py3-none-any.whl','43986f644c383f255e73dcbd2305e311f15fd70b4aeec54deb346dab5d5cbfd7')]; [(print(p), None if hashlib.sha256(Path(p).read_bytes()).hexdigest()==h else (_ for _ in ()).throw(RuntimeError('SHA256 mismatch: '+p))) for p,h in pairs]" \
    && python -m pip install --no-cache-dir --no-deps ./dist/kaggle-0.1.0-py3-none-any.whl \
    && python -m pip check \
    && python -m pip freeze > /opt/submission/runtime-requirements-resolved.txt

ENTRYPOINT ["python", "src/runner.py"]
CMD ["--help"]
