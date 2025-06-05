#!/usr/bin/env bash

# download conda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source /root/miniconda3/etc/profile.d/conda.sh

# This is required to activate conda environment in scripts
eval "$(conda shell.bash hook)"

# Create and activate a new conda environment
conda create -n spatialrgpt_finetune python=3.10 -y # pyproject.toml specifies >=3.8, 3.10 is safe
conda activate spatialrgpt_finetune

# Upgrade pip
pip install --upgrade pip

# Install CUDA toolkit (optional if your RunPod image has a compatible one)
# conda install -c nvidia cuda-toolkit -y 

# Install FlashAttention (as per environment_setup.sh)
# Make sure cu122torch2.3 matches your PyTorch version and CUDA. torch==2.3.0 is in pyproject.toml
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.8/flash_attn-2.5.8+cu122torch2.3cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

# Install dependencies from pyproject.toml (VILA project)
# We'll install them directly using pip. Note: transformers will be overridden by the git+https later.
pip install \
    torch==2.3.0 torchvision==0.18.0 torchaudio \
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

# Install optional training dependencies
pip install deepspeed==0.9.5 ninja wandb

# Install specific Transformers version from git and apply patches (as per environment_setup.sh)
# This suggests the project relies on a fork or specific commit of transformers.
pip install git+https://github.com/huggingface/transformers@v4.37.2
# The following copy commands assume you are in the root of the cloned SpatialRGPT/VILA repo
# and that the site-packages path is correctly found.
SITE_PKG_PATH=$(python -c 'import site; print(site.getsitepackages()[0])')
if [ -d "./llava/train/transformers_replace" ] && [ -d "$SITE_PKG_PATH/transformers" ]; then
    echo "Applying transformers replacements..."
    cp -rv ./llava/train/transformers_replace/* $SITE_PKG_PATH/transformers/
else
    echo "Warning: Transformers replacement paths not found or site-packages path incorrect."
fi
if [ -d "./llava/train/deepspeed_replace" ] && [ -d "$SITE_PKG_PATH/deepspeed" ]; then
    echo "Applying deepspeed replacements..."
    cp -rv ./llava/train/deepspeed_replace/* $SITE_PKG_PATH/deepspeed/
else
    echo "Warning: Deepspeed replacement paths not found or site-packages path incorrect."
fi

# Install the project itself in editable mode (if its setup.py or pyproject.toml is in root)
# This makes `llava` modules importable.
pip install -e . 