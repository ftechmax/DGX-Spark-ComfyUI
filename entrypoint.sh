#!/usr/bin/env bash
set -euo pipefail

COMFY_DIR="/opt/ComfyUI"
PORT="${COMFYUI_PORT:-8188}"
FLAGS="${COMFYUI_FLAGS:---listen 0.0.0.0 --port ${PORT}}"

echo "[entrypoint] User:   $(id -un) ($(id -u):$(id -g))"
echo "[entrypoint] Python: $(python --version)"
echo "[entrypoint] Torch:  $(python -c 'import torch; print(torch.__version__)')"
echo "[entrypoint] CUDA:   $(python -c 'import torch; print(torch.version.cuda)')"
echo "[entrypoint] Flags:  ${FLAGS}"

# Build manifest written by the Dockerfile — says exactly which ComfyUI and
# SageAttention commits this image was built from.
if [[ -f /opt/image-id ]]; then
    while read -r line; do
        echo "[entrypoint] Image:  ${line}"
    done < /opt/image-id
fi

# Ensure ComfyUI-Manager exists in mounted custom_nodes
# Check for __init__.py to detect corrupted/partial installs
if [[ ! -f "${COMFY_DIR}/custom_nodes/ComfyUI-Manager/__init__.py" ]]; then
    echo "[entrypoint] ComfyUI-Manager missing or corrupted, cloning latest..."
    rm -rf "${COMFY_DIR}/custom_nodes/ComfyUI-Manager" 2>/dev/null || true
    git clone https://github.com/ltdrdata/ComfyUI-Manager.git \
        "${COMFY_DIR}/custom_nodes/ComfyUI-Manager" || true
fi

# The image ships an onnxruntime-gpu built from source for sm_121 (no aarch64
# wheel exists on PyPI). Detect it so custom node requirements can't replace it.
if python -c 'import importlib.metadata as m; m.version("onnxruntime-gpu")' 2>/dev/null; then
    HAVE_ORT_GPU=1
    echo "[entrypoint] ONNX:   $(python -c 'import onnxruntime; print(onnxruntime.get_available_providers())' 2>&1 | tail -1)"
else
    HAVE_ORT_GPU=0
    echo "[entrypoint] onnxruntime-gpu missing, custom nodes will fall back to CPU ONNX"
fi

# Install any requirements from custom nodes.
# If the file fails as a whole, retry per package so one unavailable pin doesn't
# take out the rest of that node's dependencies.
install_reqs() {
    local req="$1"
    local patched
    patched="$(mktemp)"

    if [[ "$HAVE_ORT_GPU" == 1 ]]; then
        # Drop every onnxruntime requirement: the CPU package installs over the
        # GPU one (same module), and pinned versions have no aarch64 wheel.
        #
        # PIP_CONSTRAINT does not cover this case. `onnxruntime` and
        # `onnxruntime-gpu` are separate distributions that happen to provide the
        # same import name, so a constraint pinning one says nothing about the
        # other — only dropping the requirement outright protects the GPU build.
        sed -E '/^[[:space:]]*onnxruntime(-gpu)?([[:space:]<>=!~;[].*)?$/s/^/# pinned by image: /' "$req" > "$patched"
    else
        # No GPU build available — the CPU package at least has aarch64 wheels.
        sed -E 's/^([[:space:]]*)onnxruntime-gpu/\1onnxruntime/' "$req" > "$patched"
    fi

    if ! pip install -q -r "$patched"; then
        echo "[entrypoint] Bulk install failed for $req, retrying per package..."
        while read -r line; do
            if [[ "$line" =~ ^[[:space:]]*(#|-|$) ]]; then
                continue
            fi
            if ! pip install -q "$line"; then
                echo "[entrypoint] Skipped (unavailable): $line"
                # Remembered so the skip-stamp is not written for an install
                # that only partly succeeded — otherwise the next start would
                # skip it and the missing package would look permanent.
                REQS_FAILED=1
            fi
        done < "$patched"
    fi

    rm -f "$patched"
}

# This loop used to run unconditionally on every start: ~12 pip resolutions
# against PyPI before ComfyUI even booted, so restarts were slow, needed the
# network, and a transient failure could quietly change the environment.
#
# Hash the requirements files and skip the pass when nothing has changed.
#
# The stamp deliberately lives INSIDE the venv, not in the mounted user/ volume.
# These packages are installed into /opt/venv, which is part of the container's
# writable layer rather than a volume — so `up --force-recreate`, a rebuild, or
# any `down`/`up` cycle resets the venv to its image state and the packages are
# gone. A stamp on a mounted volume would outlive the venv it describes and skip
# an install the fresh container genuinely needs, which shows up as ComfyUI
# Manager silently losing GitPython and uv. Keeping the stamp beside the venv
# ties it to exactly the right lifetime:
#   docker restart / up (container kept)  -> stamp survives, install skipped
#   up --force-recreate / rebuild / down  -> stamp gone with the venv, reinstall
#
# The ABI-critical packages are separately floored by PIP_CONSTRAINT
# (/opt/constraints.txt, written at build time), so a node pinning an older
# torch/numpy/triton now fails on that one requirement instead of silently
# downgrading the image underneath ComfyUI.
REQS_STAMP="${VENV:-/opt/venv}/.custom-node-reqs.sha256"
REQS_HASH="$( { cat "${COMFY_DIR}"/custom_nodes/*/requirements.txt 2>/dev/null || true; } \
    | sha256sum | cut -d' ' -f1)"

if [[ "${FORCE_REQS_INSTALL:-0}" != 1 && "$REQS_HASH" == "$(cat "$REQS_STAMP" 2>/dev/null || true)" ]]; then
    echo "[entrypoint] Custom node deps unchanged, skipping install (set FORCE_REQS_INSTALL=1 to override)"
else
    REQS_FAILED=0
    for req in "${COMFY_DIR}"/custom_nodes/*/requirements.txt; do
        if [[ -f "$req" ]]; then
            echo "[entrypoint] Installing deps from: $req"
            install_reqs "$req"
        fi
    done
    if [[ "$REQS_FAILED" == 0 ]]; then
        echo "$REQS_HASH" > "$REQS_STAMP"
    else
        echo "[entrypoint] Some deps could not be installed — not stamping, will retry next start"
    fi
fi

# SageAttention live kernel gate.
#
# The image compiles SageAttention for sm_121 blind — no GPU exists at build
# time — and the build is wrapped in `|| true`, so a failed compile ships
# silently. ComfyUI does not check either: --use-sage-attention is accepted
# unconditionally and sageattn() is first called from attention.py at the first
# attention op, which is inside the sampler. A broken or pip-shadowed build
# therefore shows up as a workflow that hangs or dies at the KSampler with no
# hint that attention is the cause.
#
# So prove the kernel actually runs on THIS GPU before we launch. If it doesn't,
# drop the flag and start on PyTorch attention (slower, but a working server)
# rather than handing the user another sampler-time mystery.
if [[ "$FLAGS" == *--use-sage-attention* ]]; then
    if python - <<'PY' >/dev/null 2>&1
import torch
from sageattention import sageattn
q = torch.randn(1, 8, 1024, 128, dtype=torch.float16, device="cuda")
o = sageattn(q, q, q, tensor_layout="HND")
torch.cuda.synchronize()
assert o.shape == q.shape and torch.isfinite(o).all()
PY
    then
        echo "[entrypoint] Sage:   kernel verified live on this GPU"

        # The gate above proves one shape (fp16, head_dim 128). ComfyUI dispatches
        # sageattn per attention op and comfy/ldm/modules/attention.py catches any
        # failure, prints "Error running sage attention:" and silently falls back
        # to PyTorch attention for that op. So sage can be "working" at startup and
        # still be dead in the workflow you actually run — with --bf16-unet the
        # real calls arrive as bf16, which the gate never tested.
        #
        # Report the common dtype/head_dim combinations so that fallback is
        # visible here rather than only as unexplained slowness. Advisory only:
        # an unsupported shape is normal and must not drop the flag.
        python - <<'PY' || true
import torch
from sageattention import sageattn
for dtype in (torch.float16, torch.bfloat16):
    ok = []
    for head_dim in (64, 96, 128):
        q = torch.randn(1, 8, 1024, head_dim, dtype=dtype, device="cuda")
        try:
            o = sageattn(q, q, q, tensor_layout="HND")
            torch.cuda.synchronize()
            assert o.shape == q.shape and torch.isfinite(o).all()
            ok.append(str(head_dim))
        except Exception:
            pass
    name = str(dtype).rsplit(".", 1)[-1]
    print(f"[entrypoint] Sage:   {name} head_dim {','.join(ok) if ok else 'NONE (falls back to PyTorch)'}")
PY
    else
        echo "[entrypoint] ================================================================"
        echo "[entrypoint] WARNING: the SageAttention sm_121 kernel FAILED on this GPU."
        echo "[entrypoint] Dropping --use-sage-attention and falling back to PyTorch"
        echo "[entrypoint] attention so the server still starts. Diagnose with:"
        echo "[entrypoint]   docker compose exec comfyui python -c 'from sageattention import sageattn'"
        echo "[entrypoint] Usually a failed build (SAGEATTN_REF), a pip install that"
        echo "[entrypoint] shadowed it with a PyPI wheel carrying no GB10 kernel, or a"
        echo "[entrypoint] ptxas/driver mismatch ('no kernel image available')."
        echo "[entrypoint] ================================================================"
        FLAGS="${FLAGS//--use-sage-attention/}"
    fi
fi

# Triton / torch.compile gate.
#
# Triton on GB10 historically failed because PyTorch's bundled ptxas is CUDA
# 12.8-era and rejects the sm_121a target ("ptxas fatal: Value 'sm_121a' is not
# defined"). TorchInductor then either errors or thrashes on repeated failed
# compiles, which is why this image used to hard-disable torch.compile.
#
# The Dockerfile now points TRITON_PTXAS_PATH at the CUDA 13.1 ptxas from the
# base image, which does know sm_121a. Rather than trust that, actually compile
# something on this GPU and enable torch.compile only if it works.
#
# The probe drives torch.compile rather than a bare @triton.jit kernel, for two
# reasons. It tests the thing being gated end to end — Dynamo, Inductor codegen,
# Triton, ptxas, driver — instead of just one layer. And a bare @triton.jit
# function cannot be probed this way at all: the decorator calls
# inspect.getsource() on it, which raises for anything fed in over stdin, so a
# heredoc probe dies with "@jit functions should be defined in a Python file"
# before it ever reaches the GPU and reports a Triton failure that isn't real.
# Inductor writes its generated kernels to real files, so it is unaffected.
#
# Set TORCH_COMPILE_DISABLE yourself to skip the probe and pin the behaviour.
if [[ -n "${TRITON_PTXAS_PATH:-}" && ! -x "${TRITON_PTXAS_PATH}" ]]; then
    echo "[entrypoint] Triton: ${TRITON_PTXAS_PATH} missing, falling back to the bundled ptxas"
    unset TRITON_PTXAS_PATH
fi

if [[ -n "${TORCH_COMPILE_DISABLE:-}" ]]; then
    echo "[entrypoint] Triton: probe skipped, TORCH_COMPILE_DISABLE=${TORCH_COMPILE_DISABLE} was set explicitly"
    export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-$TORCH_COMPILE_DISABLE}"
elif python - <<'PY' >/dev/null 2>&1
import torch
import torch._dynamo.config as dynamo_config

# Without this, a disabled Dynamo would make torch.compile hand back the eager
# module, the probe would pass, and we would "enable" a torch.compile that
# silently compiles nothing.
assert dynamo_config.disable is False, "dynamo is disabled; probe would prove nothing"

model = torch.nn.Sequential(
    torch.nn.Linear(1024, 1024),
    torch.nn.SiLU(),
    torch.nn.Linear(1024, 1024),
).cuda().bfloat16()

x = torch.randn(8, 1024, device="cuda", dtype=torch.bfloat16)
out = torch.compile(model, fullgraph=True)(x)
torch.cuda.synchronize()
assert out.shape == (8, 1024) and torch.isfinite(out).all()
PY
then
    echo "[entrypoint] Triton: torch.compile verified on this GPU (ptxas: ${TRITON_PTXAS_PATH:-bundled}), enabled"
    export TORCH_COMPILE_DISABLE=0
    export TORCHDYNAMO_DISABLE=0
else
    echo "[entrypoint] Triton: torch.compile FAILED on this GPU — disabling it."
    echo "[entrypoint]         TorchCompileModel nodes will silently no-op. The probe hides"
    echo "[entrypoint]         its output; re-run it by hand to see the actual error:"
    echo "[entrypoint]           docker compose exec comfyui python -c \"import torch; print(torch.compile(lambda t: t * 2)(torch.ones(8, device='cuda')))\""
    echo "[entrypoint]           docker compose exec comfyui \${TRITON_PTXAS_PATH:-ptxas} --version"
    export TORCH_COMPILE_DISABLE=1
    export TORCHDYNAMO_DISABLE=1
fi

echo "[entrypoint] Launching with: ${FLAGS}"
exec python "${COMFY_DIR}/main.py" ${FLAGS}
