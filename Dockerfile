# ---- ONNX Runtime GPU (aarch64 + sm_121) ----
# Prebuilt out-of-band by Dockerfile.onnxruntime — a scratch image holding only
# the wheel. The compile takes ~45-60 minutes, so it is deliberately not part of
# this build; see the ONNX Runtime section of the README.
#
#   docker compose build onnxruntime      # once, before the first build here
#
# A "pull access denied / not found" error on this line means that image does not
# exist locally yet — run the command above rather than retrying the build.
ARG ONNXRUNTIME_IMAGE=onnxruntime-gb10:1.28.0-cu131-sm121
FROM ${ONNXRUNTIME_IMAGE} AS onnxruntime-wheels

# CUDA 13.1 for Blackwell GB10 (sm_121 / compute_121)
# CUDA 12.8 only supports up to sm_120, but GB10 is sm_121.
# "devel" includes nvcc so we can compile CUDA extensions like SageAttention.
FROM nvidia/cuda:13.1.1-devel-ubuntu24.04

ARG DEBIAN_FRONTEND=noninteractive
# Pinned refs, not moving branches: a rebuild must reproduce the same image.
# Bump these deliberately (and rebuild) rather than tracking a branch — a broken
# SageAttention build is only discoverable at runtime.
ARG COMFYUI_REF=v0.31.0
ARG SAGEATTN_REF=d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5
# Host uid/gid the container runs as, so bind-mounted files are not root-owned.
ARG PUID=1000
ARG PGID=1000

# Base system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates \
    python3 python3-pip python3-venv python3-dev \
    build-essential ninja-build cmake pkg-config \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 libxcb1 \
    libtcmalloc-minimal4 \
    libcudnn9-cuda-13 \
    && rm -rf /var/lib/apt/lists/*

# ---- Non-root user ----
# Everything below this line runs as `comfy`, so /opt/venv, /opt/ComfyUI and the
# bind mounts are owned by the host user instead of root. This has to happen
# BEFORE the heavy layers: a `chown -R` after ~15 GB of torch and ComfyUI has
# been installed would duplicate all of it into a new layer. Here /opt is still
# effectively empty, so taking ownership is free.
#
# Ubuntu 24.04 ships a stock `ubuntu` user already holding uid/gid 1000, so
# whatever currently occupies PUID/PGID is removed before we reuse the ids.
RUN set -eux; \
    if getent passwd ${PUID} >/dev/null; then \
        userdel -r "$(getent passwd ${PUID} | cut -d: -f1)" 2>/dev/null || true; \
    fi; \
    if getent group ${PGID} >/dev/null; then \
        groupdel "$(getent group ${PGID} | cut -d: -f1)" 2>/dev/null || true; \
    fi; \
    groupadd -g ${PGID} comfy; \
    useradd -m -u ${PUID} -g ${PGID} -s /bin/bash comfy; \
    mkdir -p /opt; \
    chown -R ${PUID}:${PGID} /opt

USER comfy
# Set explicitly rather than relying on Docker resolving it from /etc/passwd:
# compose runs the container with a numeric `user:`, and a stray HOME=/ would
# send pip/HF/pixi caches somewhere unwritable.
ENV HOME=/home/comfy

# Create venv (keeps python deps isolated inside container)
ENV VENV=/opt/venv
RUN python3 -m venv $VENV
ENV PATH="$VENV/bin:$PATH"

# Upgrade packaging tools.
# nvidia-ml-py, NOT pynvml: the PyPI `pynvml` package is deprecated and states
# outright that it no longer maintains the `pynvml` module — it just depends on
# nvidia-ml-py and wraps it in a FutureWarning shim, which torch trips on every
# `import pynvml` in torch/cuda/__init__.py. nvidia-ml-py is NVIDIA's official
# binding, provides the same module, and is pure Python (py3-none-any).
# Needed because ComfyUI does not pull an NVML binding itself and Crystools
# pins `pynvml; platform_machine != 'aarch64'`, i.e. skips it on GB10.
RUN pip install -U pip setuptools wheel nvidia-ml-py

# ---- PyTorch (ARM64 + CUDA 13.0) ----
# PyTorch cu130 wheels work with CUDA 13.1.x runtime.
RUN pip install --index-url https://download.pytorch.org/whl/cu130 \
    torch torchvision

# ---- ComfyUI ----
# No `|| true` on the checkout: a ref that does not resolve must fail the build
# rather than silently leave the image on the clone's default branch.
RUN git clone https://github.com/comfyanonymous/ComfyUI.git /opt/ComfyUI && \
    git -C /opt/ComfyUI -c advice.detachedHead=false checkout ${COMFYUI_REF}

RUN pip install -r /opt/ComfyUI/requirements.txt

# ---- ONNX Runtime GPU ----
# Installed after ComfyUI's requirements so it wins over any CPU onnxruntime a
# dependency dragged in (both packages provide the same `onnxruntime` module).
# The dist name is onnxruntime-gpu, so custom node requirements are satisfied.
# --chown matters: without it the copied tree is root-owned, and the `rm -rf`
# below runs as `comfy` in a sticky /tmp, which would fail the build.
COPY --from=onnxruntime-wheels --chown=comfy:comfy /wheels /tmp/wheels
RUN pip uninstall -y onnxruntime || true && \
    pip install /tmp/wheels/onnxruntime_gpu-*.whl && \
    rm -rf /tmp/wheels

# ---- Comfy Kitchen Blackwell Optimization ----
# Installed by requirements.txt above as the upstream publishes an aarch64 wheel now.
# (comfy_kitchen-*-cp312-abi3-manylinux_*_aarch64.whl) from 0.2.25 onward.

# ---- ComfyUI-Manager ----
# Handled at runtime by entrypoint.sh (clones if missing in mounted volume)
# This ensures latest version on each container start

# ---- Dependency constraints ----
# Custom nodes pip-install their own requirements at runtime, and ComfyUI-Manager
# installs more on demand. Without a floor, any node pinning `torch`, `numpy` or
# `triton` silently downgrades the carefully built CUDA 13 stack — and the
# breakage only shows up later as "no kernel image available" at sampler time.
#
# Pin the ABI-critical packages to exactly what this image ships. A node asking
# for an incompatible version now fails loudly on that one requirement instead.
# Deliberately NOT pinned: transformers, safetensors, opencv and friends, which
# nodes legitimately need to move forward.
#
# Built from importlib.metadata rather than `pip freeze`, because freeze emits
# `onnxruntime-gpu @ file:///...` and `sageattention @ git+https://...` for the
# two packages installed from a wheel and from git — and pip rejects direct URLs
# inside a constraints file.
RUN python -c "import importlib.metadata as m; d={x.metadata['Name'].lower(): x.version for x in m.distributions() if x.metadata['Name']}; ns=('torch','torchvision','torchaudio','triton','numpy','onnxruntime-gpu','sageattention','comfy-kitchen'); print('\n'.join(f'{n}=={d[n]}' for n in ns if n in d))" > /opt/constraints.txt

# PIP_CONSTRAINT covers pip; UV_CONSTRAINT covers ComfyUI-Manager, which shells
# out to `uv pip` and does not read the pip variable.
ENV PIP_CONSTRAINT=/opt/constraints.txt
ENV UV_CONSTRAINT=/opt/constraints.txt

# ---- SageAttention ----
# GB10 is compute capability 12.1 (sm_121).
# CUDA 13.1 NVCC supports sm_121, so we compile directly for it.
#
# "+PTX" also embeds PTX alongside the sm_121 cubin, as a fallback for future
# architectures that have no matching cubin. Note this has nothing to do with
# Triton: TORCH_CUDA_ARCH_LIST only tells nvcc what to target for extensions
# compiled here at build time. Triton JITs from its own source at runtime and
# never reads this variable — see TRITON_PTXAS_PATH below for that half.
ENV TORCH_CUDA_ARCH_LIST="12.1+PTX"
ENV CUDA_HOME=/usr/local/cuda

# ---- Triton ptxas ----
# Triton ships its own ptxas under triton/backends/nvidia/bin/, and the one
# bundled with current PyTorch wheels is CUDA 12.8-era. It does not know the
# sm_121a target Triton asks for on GB10, so every Triton compile dies with:
#
#   ptxas fatal : Value 'sm_121a' is not defined for option 'gpu-name'
#
# which is what torch.compile / TorchInductor was disabled over. This base image
# is CUDA 13.1 devel, whose ptxas understands sm_121a natively — so just point
# Triton at it. entrypoint.sh proves this works before enabling torch.compile.
#
# Do NOT "fix" this with TRITON_OVERRIDE_ARCH or a ptxas wrapper that rewrites
# the arch flag: those compile fine and then the driver rejects the kernels.
ENV TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas

# Build/install SageAttention from repo with sm_121 support.
# No `|| true`: a failed compile used to ship silently and only surface as a
# workflow dying at the first attention op.
RUN pip install --no-build-isolation "git+https://github.com/thu-ml/SageAttention@${SAGEATTN_REF}"

# Pin SageAttention too, now that it exists. It could not be part of the
# constraints file above — that has to be written *before* this install so the
# build itself is constrained, and at that point sageattention isn't installed.
# Without this line a custom node requiring `sageattention` happily pulls the
# PyPI wheel over this sm_121 build; the runtime probe would then catch it and
# fall back to PyTorch attention, i.e. a silent loss of the fast path.
RUN python -c "import importlib.metadata as m; print(f'sageattention=={m.version(\"sageattention\")}')" >> /opt/constraints.txt

# ---- Build manifest ----
# Lets a running container say exactly what it is, and gives entrypoint.sh a
# stable marker to hash so custom node deps are reinstalled after a rebuild.
RUN { echo "comfyui=$(git -C /opt/ComfyUI rev-parse HEAD)"; \
      echo "comfyui_ref=${COMFYUI_REF}"; \
      echo "sageattention=${SAGEATTN_REF}"; \
      echo "torch=$(python -c 'import torch; print(torch.__version__)')"; \
      date -u +build=%Y-%m-%dT%H:%M:%SZ; } > /opt/image-id

# Expose ComfyUI
EXPOSE 8188

# Entry script handles runtime updates / flags.
# The chmod stays: the file is mode 644 in git, and COPY preserves source mode.
COPY --chown=comfy:comfy entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
