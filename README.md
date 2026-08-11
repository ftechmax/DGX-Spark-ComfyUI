# 🚀 ComfyUI on DGX Spark (Blackwell GB10)

A Docker Compose setup for running [ComfyUI](https://github.com/comfyanonymous/ComfyUI) on the **NVIDIA DGX Spark** (Grace-Blackwell GB10), with a mobile-friendly UI included.

Built specifically to handle the quirks of the **sm_121 / compute 12.1** architecture and its **unified CPU-GPU memory fabric**.

---

## ✨ Features

- **CUDA 13.1 base** — full `nvcc` support for GB10 (`sm_121`), enabling CUDA extension compilation
- **PyTorch cu130** — prebuilt ARM64 wheels from PyTorch's cu130 index
- **SageAttention 2** — compiled from source directly against sm_121 for full hardware attention acceleration
- **Comfy Kitchen** (`comfy_kitchen`) — NVFP4 quantization support for Blackwell, installed from the official aarch64 wheel pinned in ComfyUI's `requirements.txt`
- **ONNX Runtime GPU built for sm_121** — no aarch64 `onnxruntime-gpu` wheel exists on PyPI, so it's compiled from source in a separate, build-once image; keeps DWPose, rembg, InsightFace and friends off the CPU
- **Unified-memory optimized flags** — carefully tuned `COMFYUI_FLAGS` that avoid fighting the Grace-Blackwell memory fabric
- **`torch.compile` enabled** — Triton is pointed at the CUDA 13.1 `ptxas` that understands `sm_121a`, and the entrypoint verifies a real compile on the live GPU before switching it on
- **Persistent compile caches** — TorchInductor, Triton and CUDA JIT caches live on the host, so a restart doesn't re-pay compile time
- **Pinned dependencies** — ABI-critical packages are constrained at build time, so a custom node can't silently downgrade torch, numpy or triton
- **Runs as your user** — no more root-owned files appearing in your mounted `output/` and `custom_nodes/`
- **ComfyUI-Manager** — auto-installed at container startup into the mounted `custom_nodes` volume
- **ComfyUIMini** — lightweight mobile/tablet UI proxying to the ComfyUI backend (optional second service, **commented out** in `docker-compose.yml` by default)
- **Health checks** — the ComfyUI service exposes a health check endpoint, which also gates `depends_on` ordering if you enable ComfyUIMini
- **Persistent volumes** — models, custom nodes, outputs, inputs, user settings, and workflows are all mounted from the host

---

## 🗂️ Repo Structure

```
.
├── Dockerfile            # Main ComfyUI image (CUDA 13.1, PyTorch, SageAttention, Comfy Kitchen)
├── Dockerfile.onnxruntime # ONNX Runtime GPU wheel for sm_121 — built separately, once
├── docker-compose.yml    # Orchestrates comfyui + comfyuimini services
├── entrypoint.sh         # Runtime startup: installs ComfyUI-Manager, custom node deps, launches ComfyUI
├── .env.example          # Example environment file — copy to .env and customize
└── comfyuimini/
    └── Dockerfile        # Lightweight Node.js image for ComfyUIMini mobile UI
```

---

## ⚙️ Prerequisites

- NVIDIA DGX Spark (or any Grace-Blackwell system with `sm_121`)
- Docker with the NVIDIA Container Toolkit installed

---

## 🛠️ Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/your-username/DGX-Spark-ComfyUI.git
cd DGX-Spark-ComfyUI
```

### 2. Configure environment

```bash
cp .env.example .env
```

Edit `.env` and set your paths:

```env
# Where your ComfyUI models live on the host
COMFYUI_HOST_PATH=/home/user/comfyui

# Where ComfyUI data lives (custom_nodes, output, input, etc.)
COMFYUI_DATA_PATH=/home/user/comfyui
```

### 3. Build the ONNX Runtime wheel (once)

```bash
docker compose build onnxruntime
```

This compiles `onnxruntime-gpu` for sm_121 and takes **~45–60 minutes**. It is a
separate image, so you only ever do this once — see [ONNX Runtime GPU](#-onnx-runtime-gpu).

### 4. Build and start

```bash
docker compose up --build -d
```

ComfyUI will be available at **`http://<host-ip>:8188`**.

ComfyUIMini is **commented out** in `docker-compose.yml`. Uncomment the service (and the
`comfyuimini_workflows` volume at the bottom of the file) if you want it; it then comes up at
`http://<host-ip>:3000` once ComfyUI passes its health check.

---

## 🔧 Optimized Flags Explained

The default `COMFYUI_FLAGS` are tuned for the Grace-Blackwell unified memory architecture:

| Flag | Reason |
|---|---|
| `--disable-pinned-memory` | Reduces overhead on the unified memory fabric; pinned memory is counterproductive here |
| `--use-sage-attention` | Enables SageAttention compiled for sm_121. `entrypoint.sh` runs the kernel on the live GPU first and drops the flag if it fails, rather than letting the workflow die at the first sampler step |
| `--bf16-unet --bf16-vae --bf16-text-enc` | Keeps models in BF16 — Blackwell's native precision |
| `--preview-method auto` | Live previews during sampling |

Optional, if you hit the overcommit freeze on large models: `--reserve-vram 8` keeps that many
GB of the unified pool free for the OS, and `--disable-dynamic-vram` falls back to
estimate-based model loading.

> **Note:** Do **not** use `--gpu-only`. It forces a split memory model that fights the unified memory fabric on Grace-Blackwell systems.
> **Note:** Do **not** use `--highvram`. It pins every model in memory causing OOM issues.
> **Note:** Recent ComfyUI versions removed `--normalvram` (normal VRAM mode is the default now). Leaving it in `COMFYUI_FLAGS` makes startup fail with `main.py: error: unrecognized arguments: --normalvram` — drop the flag from your `.env`.

### Flags deliberately *not* used

These were in earlier versions of this setup and were removed after checking them against the
ComfyUI source. `.env.example` carries the full reasoning; the short version:

| Flag | Why it's gone |
|---|---|
| `--force-fp16` | `cli_args.py` makes `should_use_fp16()` return `True` unconditionally. `--bf16-unet` still wins for the unet, so you get bf16 weights on fp16 compute paths — and fp16 is the path that NaNs on Blackwell |
| `--dont-upcast-attention` | Makes `get_attn_precision()` return `None`, **overriding** the fp32 attention some models explicitly request. Linux already defaults to no upcasting, so it can only remove a safety net (black images) |
| `--disable-mmap` | Only existed to pair with a `comfy/utils.py` patch that has since been removed. `safe_open()` had already placed the tensor on the target device, so the patch was a no-op and the two cancelled out |

### Environment variables

| Variable | Purpose |
|---|---|
| `TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas` | Points Triton at the CUDA 13.1 `ptxas` from the base image. The one bundled with PyTorch is CUDA 12.8-era and rejects `sm_121a`, which is what broke `torch.compile` on GB10 |
| `TORCH_CUDA_ARCH_LIST=12.1+PTX` | What `nvcc` targets when building CUDA extensions here (SageAttention): an sm_121 cubin plus embedded PTX as a fallback for future architectures. It does **not** affect Triton, which JITs from its own source at runtime — `TRITON_PTXAS_PATH` is that half |
| `PIP_CONSTRAINT` / `UV_CONSTRAINT` | Point at `/opt/constraints.txt`, generated at build time, pinning torch, torchvision, torchaudio, triton, numpy, onnxruntime-gpu, sageattention and comfy-kitchen to exactly what the image ships. A custom node asking for an incompatible version now fails on that requirement instead of silently downgrading the stack. `UV_CONSTRAINT` is the one ComfyUI-Manager reads, since it shells out to `uv pip` |
| `TORCHINDUCTOR_CACHE_DIR`, `TRITON_CACHE_DIR`, `CUDA_CACHE_PATH` | Redirect the compile caches into the mounted `/opt/cache` volume. Their defaults live inside the container and are lost on every recreate |
| `CUDA_MODULE_LOADING=LAZY` | Loads CUDA modules lazily for faster model loading and more memory efficiency |
| `CUDA_CACHE_MAXSIZE` | Size cap for the CUDA JIT cache — only meaningful now that `CUDA_CACHE_PATH` is persistent |
| `TOKENIZERS_PARALLELISM=false` | Silences the HuggingFace tokenizers fork warning. Note the name: the `HF_`-prefixed spelling this used to use is read by nothing |
| `FORCE_REQS_INSTALL=1` | Forces custom node requirements to reinstall. By default the entrypoint hashes every `custom_nodes/*/requirements.txt` and skips the install when nothing changed. The stamp is kept inside the venv, so anything that resets the venv — `--force-recreate`, a rebuild, a `down`/`up` — reinstalls automatically |
| `PUID` / `PGID` | uid/gid the container runs as. Baked into the image at build time, so changing them needs a rebuild |

> **`torch.compile` is no longer hard-disabled.** `entrypoint.sh` runs a real `torch.compile` on the live GPU at startup and sets `TORCH_COMPILE_DISABLE` from the result — look for `[entrypoint] Triton:` in the logs. Set `TORCH_COMPILE_DISABLE` yourself in `.env` to skip the probe and pin the behaviour.
>
> The probe drives `torch.compile` rather than a bare `@triton.jit` kernel on purpose. It covers the whole stack — Dynamo, Inductor, Triton, ptxas, driver — and a bare `@triton.jit` function can't be probed from a heredoc at all: the decorator calls `inspect.getsource()`, which raises for code arriving on stdin, so it would report a Triton failure that isn't real.
>
> **Do not re-add** `PYTORCH_NO_CUDA_MEMORY_CACHING=1`, `CUDA_MANAGED_FORCE_DEVICE_ALLOC=1`, `CUDA_DEVICE_MAX_CONNECTIONS=1` or `CUBLAS_WORKSPACE_CONFIG`. They were documented here as "unified memory optimizations" but disabling the caching allocator turns every tensor alloc into a `cudaMalloc` and every free into a device-wide sync, thousands of times per sampler step — sampling looks permanently stuck. See the comment block in `docker-compose.yml`.

---

## 📦 Volumes

| Host path | Container path | Purpose |
|---|---|---|
| `${COMFYUI_HOST_PATH}/models` | `/opt/ComfyUI/models` | Model files (shared with host) |
| `${COMFYUI_DATA_PATH}/custom_nodes` | `/opt/ComfyUI/custom_nodes` | Custom nodes (incl. ComfyUI-Manager) |
| `${COMFYUI_DATA_PATH}/user` | `/opt/ComfyUI/user` | User settings & ComfyUI-Manager config |
| `${COMFYUI_DATA_PATH}/output` | `/opt/ComfyUI/output` | Generated images/videos |
| `${COMFYUI_DATA_PATH}/input` | `/opt/ComfyUI/input` | Input files |
| `${COMFYUI_DATA_PATH}/workflows` | `/opt/ComfyUI/workflows` | Saved workflows |
| `${COMFYUI_DATA_PATH}/wheels` | `/opt/wheels` | Optional wheel vault |
| `${COMFYUI_DATA_PATH}/cache` | `/opt/cache` | TorchInductor / Triton / CUDA JIT caches |
| `${COMFYUI_DATA_PATH}/home` | `/home/comfy` | Container HOME — `comfy-env`/pixi environments live here |

Create them before the first start, or Docker will create them for you as root:

```bash
mkdir -p "$COMFYUI_DATA_PATH"/{user,custom_nodes,output,input,workflows,wheels,cache,home}
```

---

## 👤 Running as your user

The container runs as `PUID:PGID` (default `1000:1000`) instead of root, so files it writes
into the mounted volumes belong to you. Set them in `.env` to your own `id -u` / `id -g`.

They are **build arguments** as well as runtime settings — the image creates the user and
takes ownership of `/opt` before installing anything, so changing them requires
`docker compose build`, not just a restart.

If you are switching an existing install away from root, take ownership of the data once:

```bash
sudo chown -R "$(id -u):$(id -g)" /path/to/your/comfyui
```

---

## 📱 ComfyUIMini (Mobile UI)

[ComfyUIMini](https://github.com/ImDarkTom/ComfyUIMini) is a lightweight, mobile-friendly interface that proxies requests to the ComfyUI backend over the internal Docker network.

It is **commented out** in `docker-compose.yml`. Uncomment the `comfyuimini` service and the `comfyuimini_workflows` volume to enable it; it then starts after ComfyUI passes its health check, serves on port 3000, and shares the `output` directory for gallery access.

---

## 🧩 ONNX Runtime GPU

Custom nodes that use ONNX models — `comfy-mtb`, Impact Pack, DWPose/ControlNet preprocessors, rembg, InsightFace — require `onnxruntime-gpu`, which Microsoft publishes **only for x86_64**. On the Spark pip fails with:

```
ERROR: Could not find a version that satisfies the requirement onnxruntime-gpu (from versions: none)
```

That failure also aborts the *rest* of that node's requirements file, and the usual workaround (installing the CPU `onnxruntime`) leaves every ONNX preprocessor silently running on the Grace cores.

So it is compiled from source instead — in `Dockerfile.onnxruntime`, a **separate image built out of band**:

```bash
docker compose build onnxruntime
```

| Setting | Value |
|---|---|
| `ONNXRUNTIME_REF` | `v1.28.0` — builds against CUDA 13.1 on GB10 and runs with the CUDA EP active |
| `CMAKE_CUDA_ARCHITECTURES` | `121` — **not** 120. A binary built for sm_120 has no matching cubin for GB10 |
| cuDNN | `libcudnn9-dev-cuda-13` + `libcudnn9-headers-cuda-13` at build time, `libcudnn9-cuda-13` in the final image. The headers are a *separate* package, and both land in Debian multiarch paths, so the builder symlinks them into `/usr/include` and `/usr/lib` for `--cudnn_home /usr` |
| `ONNXRUNTIME_JOBS` | `8` — nvcc peaks at several GB per translation unit; more jobs can OOM the build |
| `ONNXRUNTIME_IMAGE` | `onnxruntime-gb10:1.28.0-cu131-sm121` — the tag both halves agree on |

That build produces a `scratch` image containing nothing but the wheel (the ~10 GB build tree is discarded), which the main Dockerfile consumes with a single `COPY --from`. The compile is **~45–60 minutes** and is paid exactly once: a normal `docker compose build` never touches it, no matter what you change or how aggressively you prune the build cache.

Rebuild the wheel image only when `ONNXRUNTIME_REF` or the CUDA base image changes — bump `ONNXRUNTIME_IMAGE` in `.env` at the same time so the two never drift out of sync.

> If `docker compose build` fails on the very first instruction with `pull access denied` or `manifest unknown` for `onnxruntime-gb10:...`, the wheel image simply hasn't been built yet — run the command above. (The tag is local-only and never published, so Docker's attempt to fetch it from a registry is expected to fail.) This resolution needs the default `docker` buildx driver, which reads the local image store; a `docker-container` driver cannot see local tags.

To pull the wheel out onto the host instead — to archive it, or to install it somewhere that isn't this image:

```bash
docker build -f Dockerfile.onnxruntime --output type=local,dest=./wheels .
```

(The service sits behind a Compose profile so it stays out of the default build; naming it explicitly is enough to activate that, but older Compose versions may want `docker compose --profile onnxruntime build onnxruntime`.)

At startup `entrypoint.sh` prints the active providers, which should include `CUDAExecutionProvider`:

```
[entrypoint] ONNX:   ['CUDAExecutionProvider', 'CPUExecutionProvider']
```

It also comments out every `onnxruntime` / `onnxruntime-gpu` line in custom node requirements before installing them — both packages provide the same `onnxruntime` module, so an unpinned CPU requirement would otherwise overwrite the GPU build. If the image was built without the wheel, it falls back to rewriting those lines to the CPU package, and installs a failing requirements file package-by-package so one bad pin can't take out the rest of a node's dependencies.

> **Alternative:** prebuilt community wheels exist ([aarch64 sm_121 wheel](https://huggingface.co/Jay0515/onnxruntime-gpu-aarch64-cuda13-sm121), [shared libraries](https://github.com/Albatross1382/onnxruntime-aarch64-cuda-blackwell)) if you'd rather not spend an hour compiling — see the [NVIDIA build guide thread](https://forums.developer.nvidia.com/t/onnx-runtime-gpu-inference-on-dgx-spark-gx10-build-guide-and-prebuilt-binaries/366157). Verify checksums before installing third-party wheels.

---

## 🔄 Updating ComfyUI

`COMFYUI_REF` and `SAGEATTN_REF` control what the image is built from. Both default to a
**pinned tag/commit**, not a branch:

```env
COMFYUI_REF=v0.31.0
SAGEATTN_REF=d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5
```

Bump them deliberately and rebuild. Tracking `master`/`main` means an identical `docker compose
build` can quietly produce a different image, and a broken SageAttention compile is only
discoverable at runtime. Neither step swallows failure any more: a ref that doesn't resolve, or
a SageAttention build that fails, fails the build.

A running container reports what it was built from — `entrypoint.sh` prints `/opt/image-id`:

```
[entrypoint] Image:  comfyui=43cb4fff...
[entrypoint] Image:  sageattention=d1a57a54...
[entrypoint] Image:  torch=2.13.0+cu130
```

ComfyUI-Manager is **not** baked into the image — it is cloned fresh at container startup into the mounted `custom_nodes` volume, so you always get the latest version.

### Custom node dependencies

Custom node `requirements.txt` files used to be reinstalled on every single start. Now the
entrypoint hashes them and skips the whole pass when nothing changed, so restarts are fast and
don't depend on PyPI being reachable. Set `FORCE_REQS_INSTALL=1` to reinstall anyway.

The skip-stamp lives **inside the venv** (`/opt/venv/.custom-node-reqs.sha256`), not in a
mounted volume, and that placement is load-bearing. These packages install into `/opt/venv`,
which is part of the container's writable layer — so `up --force-recreate`, a rebuild, or a
`down`/`up` cycle throws them away. A stamp on a mounted volume would outlive the venv it
describes and skip an install the fresh container actually needs, which surfaces as
ComfyUI-Manager quietly losing `GitPython` and `uv` and failing to load.

If a package genuinely can't be installed, the stamp is not written at all, so the next start
retries instead of making the gap permanent.

---

## 📄 License

MIT
