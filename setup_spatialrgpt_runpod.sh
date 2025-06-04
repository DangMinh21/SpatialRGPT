#!/bin/bash

# Ensure this script is run with `bash setup_on_runpod.sh` or similar
# It assumes you are in the root directory of the cloned SpatialRGPT repository.

# 1. Set Conda Environment Name
CONDA_ENV_NAME="spatialrgpt_finetune" # Consistent with previous discussions
PYTHON_VERSION="3.10" # As used in flash-attn wheel and pyproject.toml implies >=3.8

# 2. Initialize Conda for the current shell session
echo "Initializing Conda..."
eval "$(conda shell.bash hook)"
# Optional: Prevent base from auto-activating in new shells if desired (usually done in .bashrc)
# conda config --set auto_activate_base false

# 3. Create and Activate Conda Environment
echo "Creating Conda environment: $CONDA_ENV_NAME with Python $PYTHON_VERSION..."
conda create -n $CONDA_ENV_NAME python=$PYTHON_VERSION -y
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to create conda environment. Trying to activate if it already exists."
fi
conda activate $CONDA_ENV_NAME
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to activate conda environment '$CONDA_ENV_NAME'. Please check Conda setup."
    exit 1
fi
echo "Conda environment '$CONDA_ENV_NAME' activated."
echo "Python version: $(python --version)"
echo "Python executable: $(which python)"

# 4. Upgrade Pip
echo "Upgrading pip..."
pip install --upgrade pip

# 5. Install PyTorch (Matches pyproject.toml and FlashAttention wheel)
# Ensure CUDA version on RunPod matches (cu121 or cu122 for the flash_attn wheel)
# The flash_attn wheel uses cu122torch2.3. Your pyproject.toml uses torch==2.3.0.
# If your RunPod has CUDA 12.1, use whl/cu121. If CUDA 12.2+, use whl/cu122 for PyTorch.
# For the specific FlashAttention wheel: cu122torch2.3 implies CUDA 12.2+ for PyTorch 2.3.
# Let's assume your RunPod can support CUDA 12.2 compatible PyTorch.
echo "Installing PyTorch 2.3.0 (for CUDA 12.1 as an example, adjust for your RunPod's CUDA)..."
pip3 install torch==2.3.0 torchvision==0.18.0 torchaudio --index-url https://download.pytorch.org/whl/cu121
# OR for CUDA 12.2+ (if your RunPod has it, to match flash-attn wheel's cu122 more closely, though cu121 usually works)
# pip3 install torch==2.3.0 torchvision==0.18.0 torchaudio --index-url https://download.pytorch.org/whl/cu122
echo "PyTorch installed."

# 6. Install FlashAttention (Specific Wheel)
echo "Installing FlashAttention v2.5.8 for PyTorch 2.3 & CUDA 12.2..."
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.8/flash_attn-2.5.8+cu122torch2.3cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to install FlashAttention. Check CUDA/PyTorch/Python version compatibility with the wheel."
    # exit 1 # Optionally exit if FlashAttention is critical
fi
echo "FlashAttention installed."

# 7. Install Core Dependencies (from pyproject.toml, excluding torch/torchvision already installed)
# Note: transformers will be installed here, then specifically re-installed from git in step 8.
echo "Installing core dependencies..."
pip install \
    transformers==4.37.2 tokenizers>=0.15.2 sentencepiece==0.1.99 shortuuid \
    accelerate==0.27.2 peft>=0.9.0 bitsandbytes==0.41.0 \
    "pydantic<2,>=1" "markdown2[all]" numpy==1.26.0 scikit-learn==1.2.2 \
    gradio==3.35.2 gradio_client==0.2.9 \
    requests httpx==0.24.0 uvicorn fastapi \
    einops==0.6.1 einops-exts==0.0.4 timm==0.9.12 \
    openpyxl==3.1.2 pytorchvideo==0.1.5 decord==0.6.0 \
    datasets==2.16.1 openai==1.8.0 webdataset==0.2.86 \
    nltk==3.3 pywsd==1.2.4 opencv-python==4.8.0.74 \
    git+https://github.com/bfshi/scaling_on_scales#egg=s2wrapper \
    pycocotools \
    tyro pytest pre-commit
echo "Core dependencies installed."

# 8. Install Specific Transformers from Git (as per environment_setup.sh)
echo "Installing specific Transformers version (v4.37.2) from GitHub..."
pip install git+https://github.com/huggingface/transformers@v4.37.2
echo "Transformers from Git installed."

# 9. Install Optional Training & Evaluation Dependencies
echo "Installing optional training dependencies (deepspeed, ninja, wandb)..."
pip install deepspeed==0.9.5 ninja wandb
echo "Installing optional evaluation dependencies (mmengine, word2number, Levenshtein)..."
pip install mmengine word2number Levenshtein # nltk, pywsd already in core
echo "Optional dependencies installed."

# 10. Apply Local Patches/Replacements (CRUCIAL)
echo "Applying local replacements for transformers and deepspeed..."
SITE_PKG_PATH=$(python -c 'import site; print(site.getsitepackages()[0])')

if [ -z "$SITE_PKG_PATH" ]; then
    echo "ERROR: Could not determine site-packages path. Patches cannot be applied."
else
    echo "Site-packages path: $SITE_PKG_PATH"
    if [ -d "./llava/train/transformers_replace" ] && [ -d "$SITE_PKG_PATH/transformers" ]; then
        echo "Applying transformers replacements..."
        cp -rv ./llava/train/transformers_replace/* "$SITE_PKG_PATH/transformers/"
    else
        echo "Warning: Transformers replacement source ('./llava/train/transformers_replace') or destination ('$SITE_PKG_PATH/transformers') not found."
    fi

    if [ -d "./llava/train/deepspeed_replace" ] && [ -d "$SITE_PKG_PATH/deepspeed" ]; then
        echo "Applying deepspeed replacements..."
        cp -rv ./llava/train/deepspeed_replace/* "$SITE_PKG_PATH/deepspeed/"
    else
        echo "Warning: Deepspeed replacement source ('./llava/train/deepspeed_replace') or destination ('$SITE_PKG_PATH/deepspeed') not found."
    fi
fi
echo "Local replacements applied (if paths were valid)."

# 11. Install the Project in Editable Mode
echo "Installing 'vila' (SpatialRGPT project) in editable mode..."
pip install -e .
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to install project in editable mode. Check pyproject.toml and setup.py (if any)."
    exit 1
fi
echo "'vila' project installed in editable mode."

# 12. NLTK Data (Optional, run in Python if needed later)
# echo "To download NLTK data (punkt, wordnet, averaged_perceptron_tagger), run the following in Python later if required:"
# echo "import nltk"
# echo "nltk.download('punkt', quiet=True)"
# echo "nltk.download('wordnet', quiet=True)"
# echo "nltk.download('averaged_perceptron_tagger', quiet=True)"

echo "Environment setup script finished."
echo "Please verify key packages, e.g., by running:"
echo "python -c \"import torch; print(f'Torch: {torch.__version__}'); import transformers; print(f'Transformers: {transformers.__version__}'); import deepspeed; print(f'Deepspeed: {deepspeed.__version__}'); import flash_attn; print('FlashAttention imported.')\""