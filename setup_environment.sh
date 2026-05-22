#!/bin/bash
set -e

echo "=== Depth Estimation Pipeline Setup ==="
echo "Jetson AGX Orin / JetPack 6.2 / CUDA 12.6"

# ── Variables ────────────────────────────────────────────────
BASE_DIR=~/depth_anything_V2
JETSON_TORCH_WHL=~/torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl

# ── System deps ──────────────────────────────────────────────
echo ""
echo ">>> Installing system dependencies..."
sudo apt install -y \
    python3-venv python3-pip \
    cuda-toolkit-12-6 \
    tensorrt \
    ffmpeg \
    code \
    libjpeg-dev zlib1g-dev libpython3-dev \
    libopenblas-dev libavcodec-dev \
    libavformat-dev libswscale-dev

# ── Add TensorRT to PATH ─────────────────────────────────────
echo 'export PATH=$PATH:/usr/src/tensorrt/bin' >> ~/.bashrc

# ── Add cusparselt to LD_LIBRARY_PATH ───────────────────────
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/murg/depth_anything_V2/depth_anything_env/lib/python3.10/site-packages/nvidia/cusparselt/lib/' >> ~/.bashrc

# ── Install uv ───────────────────────────────────────────────
echo ""
echo ">>> Installing uv..."
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env

# ── Clone repos ──────────────────────────────────────────────
echo ""
echo ">>> Cloning repositories..."
mkdir -p $BASE_DIR
cd $BASE_DIR

git clone https://github.com/DepthAnything/Depth-Anything-V2.git
git clone https://github.com/spacewalk01/depth-anything-tensorrt.git
git clone https://github.com/lpiccinelli-eth/UniDepth.git
git clone https://github.com/prs-eth/Marigold.git

# ── depth_anything_env ───────────────────────────────────────
echo ""
echo ">>> Setting up depth_anything_env..."
cd $BASE_DIR
uv venv depth_anything_env --system-site-packages
source depth_anything_env/bin/activate

uv pip install onnx opencv-python pycuda onnxscript \
               scipy transformers==4.46.0 diffusers==0.31.0 \
               huggingface_hub rerun-sdk dearpygui tqdm

# Install Jetson PyTorch
echo ">>> Installing Jetson PyTorch (download first if not present)..."
if [ -f "$JETSON_TORCH_WHL" ]; then
    UV_SKIP_WHEEL_FILENAME_CHECK=1 uv pip install $JETSON_TORCH_WHL
else
    echo "WARNING: Jetson torch wheel not found at $JETSON_TORCH_WHL"
    echo "Download from: https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/"
fi

uv pip install "numpy<2"

# Build torchvision from source
echo ">>> Building torchvision..."
cd ~
git clone --branch v0.19.1 https://github.com/pytorch/vision.git torchvision
cd torchvision
python setup.py install
cd $BASE_DIR

# Install DA2 deps
cd $BASE_DIR/Depth-Anything-V2
uv pip install -r requirements.txt

# Copy TRT export files
cp $BASE_DIR/depth-anything-tensorrt/depth_anything_v2/dpt.py \
   $BASE_DIR/Depth-Anything-V2/depth_anything_v2/dpt.py

deactivate

# ── marigold_env ─────────────────────────────────────────────
echo ""
echo ">>> Setting up marigold_env..."
cd $BASE_DIR/Marigold
uv venv marigold_env --system-site-packages
source marigold_env/bin/activate

if [ -f "$JETSON_TORCH_WHL" ]; then
    UV_SKIP_WHEEL_FILENAME_CHECK=1 uv pip install $JETSON_TORCH_WHL
fi

uv pip install "numpy<2"

cd ~/torchvision
python setup.py install
cd $BASE_DIR/Marigold

uv pip install -r requirements.txt diffusers==0.31.0 transformers==4.46.0

deactivate

# ── unidepth_env ─────────────────────────────────────────────
echo ""
echo ">>> Setting up unidepth_env..."
cd $BASE_DIR/UniDepth
uv venv unidepth_env --system-site-packages
source unidepth_env/bin/activate

uv pip install -e .
uv pip install onnxscript pycuda opencv-python

deactivate

# ── Download DA2 weights ─────────────────────────────────────
echo ""
echo ">>> Downloading DA2 weights..."
cd $BASE_DIR/Depth-Anything-V2
mkdir -p checkpoints

wget -q --show-progress \
    https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth \
    -O checkpoints/depth_anything_v2_vits.pth

wget -q --show-progress \
    https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth \
    -O checkpoints/depth_anything_v2_vitb.pth

wget -q --show-progress \
    https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth \
    -O checkpoints/depth_anything_v2_vitl.pth

echo ""
echo "=== Setup complete ==="
echo ""
echo "Next steps:"
echo "  1. Activate env:  source ~/depth_anything_V2/depth_anything_env/bin/activate"
echo "  2. Export ONNX:   python export_v2.py --encoder vitl --input-size 518"
echo "  3. Build TRT:     trtexec --onnx=depth_anything_v2_vitl.onnx --saveEngine=depth_anything_v2_vitl.engine --fp16"
echo "  4. Run inference: python ~/depth_anything_V2/depth-anything-tensorrt/python/trt_infer_video.py"