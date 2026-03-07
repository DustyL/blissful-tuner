#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# Bleeding-edge dev environment setup for Dustin's local work.
# Installs everything from git main + PyTorch nightly cu130.
#
# Usage:
#   ./dev-setup.sh          # full install
#   ./dev-setup.sh --quick  # skip git deps, only torch + cuda
# ─────────────────────────────────────────────────────────────
set -euo pipefail

VENV="$(cd "$(dirname "$0")" && pwd)/venv"
UV="${VENV}/bin/uv"
PIP="${UV} pip"

if [ ! -f "${VENV}/bin/python" ]; then
    echo "ERROR: venv not found at ${VENV}. Create it first: uv venv ${VENV}"
    exit 1
fi

QUICK=false
if [ "${1:-}" = "--quick" ]; then
    QUICK=true
fi

echo "═══════════════════════════════════════════════════════════"
echo "  Blissful Tuner — Bleeding Edge Dev Setup (CUDA 13.x)"
echo "═══════════════════════════════════════════════════════════"

# ── Step 0: Clean up any CUDA 12.x nvidia packages ──────────
echo ""
echo "── Cleaning CUDA 12.x nvidia packages ──"
CUDA12_PKGS=(
    nvidia-cublas-cu12
    nvidia-cuda-cupti-cu12
    nvidia-cuda-nvrtc-cu12
    nvidia-cuda-runtime-cu12
    nvidia-cudnn-cu12
    nvidia-cufft-cu12
    nvidia-cufile-cu12
    nvidia-curand-cu12
    nvidia-cusolver-cu12
    nvidia-cusparse-cu12
    nvidia-cusparselt-cu12
    nvidia-nccl-cu12
    nvidia-nvjitlink-cu12
    nvidia-nvshmem-cu12
    nvidia-nvtx-cu12
)
# Silently uninstall — ignore errors for packages not installed
${PIP} uninstall "${CUDA12_PKGS[@]}" 2>/dev/null || true

# ── Step 1: Install blissful-tuner editable (no torch) ──────
echo ""
echo "── Installing blissful-tuner (editable, no torch) ──"
${PIP} install -e "$(dirname "$0")" --no-deps 2>/dev/null || true
# Install non-git deps from pyproject (these are fine from PyPI)
${PIP} install \
    av \
    toml \
    tqdm \
    ftfy \
    easydict \
    sentencepiece \
    ffmpeg-python \
    PySide6 \
    yacs \
    opencv-python \
    ascii-magic \
    tensorboard \
    ruff \
    basicsr \
    facexlib \
    cupy-cuda13x \
    lpips \
    safetensors \
    voluptuous

# ── Step 2: PyTorch nightly (CUDA 13.0) ─────────────────────
echo ""
echo "── Installing PyTorch nightly (cu130) ──"
${PIP} install --index-url https://download.pytorch.org/whl/nightly/cu130 \
    torch torchvision triton

# ── Step 3: CUDA 13.x toolkit + extras ──────────────────────
echo ""
echo "── Installing CUDA 13.x toolkit ──"
${PIP} install "cuda-toolkit[all]==13.*"
${PIP} install \
    nvidia-cudnn-cu13 \
    nvidia-nccl-cu13 \
    nvidia-nvshmem-cu13 \
    nvidia-cusparselt-cu13

if [ "$QUICK" = true ]; then
    echo ""
    echo "── Quick mode: skipping git deps ──"
    echo "Done!"
    exit 0
fi

# ── Step 4: Bleeding-edge git deps (HuggingFace ecosystem) ──
echo ""
echo "── Installing bleeding-edge HuggingFace packages ──"
${PIP} install \
    "git+https://github.com/huggingface/accelerate.git@main" \
    "git+https://github.com/huggingface/diffusers.git@main" \
    "git+https://github.com/huggingface/transformers.git@main" \
    "git+https://github.com/huggingface/safetensors.git@main" \
    "git+https://github.com/huggingface/datasets.git@main" \
    "git+https://github.com/huggingface/tokenizers.git@main#subdirectory=bindings/python" \
    "huggingface_hub[mcp,torch,dev]"

echo ""
echo "── Installing bleeding-edge core deps ──"
${PIP} install \
    "git+https://github.com/bitsandbytes-foundation/bitsandbytes.git@main" \
    "git+https://github.com/arogozhnikov/einops.git@main" \
    "git+https://github.com/python-pillow/Pillow.git@main" \
    "git+https://github.com/alecthomas/voluptuous.git@main"

echo ""
echo "── Installing bleeding-edge Blissful-specific deps ──"
${PIP} install \
    "git+https://github.com/omry/omegaconf.git@main" \
    "git+https://github.com/huggingface/pytorch-image-models.git@main" \
    "git+https://github.com/Delgan/loguru.git@main" \
    "git+https://github.com/Textualize/rich.git@main" \
    "git+https://github.com/hamdanal/rich-argparse.git@main" \
    "git+https://github.com/celsiusnarhwal/rich-tracebacks.git@main" \
    "git+https://github.com/prompt-toolkit/python-prompt-toolkit.git@main" \
    "git+https://github.com/matplotlib/matplotlib.git@main"

echo ""
echo "── Installing bleeding-edge postprocess + tools ──"
${PIP} install \
    "git+https://github.com/chaiNNer-org/spandrel.git@main" \
    "git+https://github.com/ultralytics/ultralytics.git@main" \
    "git+https://github.com/pypa/hatch.git@main"

echo ""
echo "── Installing bleeding-edge optimizers ──"
${PIP} install \
    "git+https://github.com/LoganBooker/prodigy-plus-schedule-free.git@main" \
    "git+https://github.com/KohakuBlueleaf/LyCORIS.git@main" \
    "git+https://github.com/facebookresearch/schedule_free.git@main" \
    "git+https://github.com/konstmish/prodigy.git@main" \
    "git+https://github.com/nikhilvyas/SOAP.git@main" \
    "git+https://github.com/facebookresearch/dadaptation.git@main" \
    "git+https://github.com/kozistr/pytorch_optimizer.git@main"

# ── Step 5: xformers (optional, may fail on nightly torch) ──
echo ""
echo "── Installing xformers (may fail on nightly) ──"
${PIP} install "git+https://github.com/facebookresearch/xformers.git@main#egg=xformers" || \
    echo "WARNING: xformers failed to install (common with nightly torch, not critical)"

# ── Step 6: Final cu12 cleanup (in case any dep pulled them back) ──
echo ""
echo "── Final CUDA 12.x cleanup ──"
${PIP} uninstall "${CUDA12_PKGS[@]}" 2>/dev/null || true

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  Done! All packages installed from bleeding edge."
echo "═══════════════════════════════════════════════════════════"
