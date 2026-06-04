#!/bin/bash
# install.sh — NaylisGDN sur Modal (B200, CUDA 12.8)
# Usage kernel Jupyter : !bash install.sh 2>&1 | tail -50
set -e

CUDA_HOME=/usr/local/cuda          # symlink → /usr/local/cuda-12.8 sur Modal
export CUDA_HOME
export NVTE_CUDA_INCLUDE_PATH=$CUDA_HOME/include
export PATH=$CUDA_HOME/bin:$PATH

echo "=== [1/9] NCCL ==="
apt-get install -y -q libnccl2 libnccl-dev

echo "=== [2/9] CUDA 12.8 dev libs ==="
apt-get update -q && apt-get install -y -q \
    cuda-toolkit-12-8 \
    libcusparse-dev-12-8 \
    libcublas-dev-12-8 \
    cuda-nvcc-12-8 \
    libcudnn9-dev-cuda-12

echo "  nvcc : $(nvcc --version | grep release)"
echo "  CUDA_HOME=$CUDA_HOME"

echo "=== [3/9] Ninja ==="
pip install -q ninja

echo "=== [4/9] Transformer Engine (FP8 — build 10-20 min) ==="
CUDA_HOME=$CUDA_HOME \
NVTE_CUDA_INCLUDE_PATH=$CUDA_HOME/include \
MAX_JOBS=1 \
pip install --no-build-isolation transformer-engine[pytorch]

echo "=== [5/9] FlashAttention ==="
CUDA_HOME=$CUDA_HOME \
pip install flash-attn --no-build-isolation

echo "=== [6/9] Flash Linear Attention (GDN kernels Triton) ==="
pip install -q flash-linear-attention

echo "=== [7/9] Liger Kernel (kernels Triton : RMSNorm, RoPE, SwiGLU, CrossEntropy) ==="
pip install -q liger-kernel

echo "=== [8/9] Dépendances Python ==="
pip install -q -r requirements.txt

echo "=== [9/9] Package naylisgdn ==="
pip install -q -e .

echo ""
echo "=== Vérification ==="
python -c "
import transformer_engine
print('  ✓ transformer_engine', transformer_engine.__version__)
import flash_attn
print('  ✓ flash_attn', flash_attn.__version__)
from fla.ops.gated_delta_rule import chunk_gated_delta_rule
print('  ✓ flash-linear-attention (fla)')
import liger_kernel
print('  ✓ liger_kernel', liger_kernel.__version__)
from naylisgdn import NaylisGDN
print('  ✓ naylisgdn')
print('OK — prêt pour train.py')
"
