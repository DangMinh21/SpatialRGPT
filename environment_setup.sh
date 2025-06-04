# #!/usr/bin/env bash

# # This is required to activate conda environment
# eval "$(conda shell.bash hook)"

# CONDA_ENV=${1:-""}
# if [ -n "$CONDA_ENV" ]; then
#     conda create -n $CONDA_ENV python=3.10 -y
#     conda activate $CONDA_ENV
# else
#     echo "Skipping conda environment creation. Make sure you have the correct environment activated."
# fi

# # This is required to enable PEP 660 support
# pip install --upgrade pip

# # This is optional if you prefer to use built-in nvcc
# conda install -c nvidia cuda-toolkit -y

# # Install FlashAttention2
# pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.8/flash_attn-2.5.8+cu122torch2.3cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

# # Install VILA
# pip install -e .
# pip install -e ".[train]"
# pip install -e ".[eval]"

# # Install HF's Transformers
# pip install git+https://github.com/huggingface/transformers@v4.37.2
# site_pkg_path=$(python -c 'import site; print(site.getsitepackages()[0])')
# cp -rv ./llava/train/transformers_replace/* $site_pkg_path/transformers/
# cp -rv ./llava/train/deepspeed_replace/* $site_pkg_path/deepspeed/


# ---------- modify ---------------
#!/usr/bin/env bash

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

bash Miniconda3-latest-Linux-x86_64.sh
source /root/miniconda3/etc/profile.d/conda.sh
conda activate spatialrgpt_finetune
command -v conda

# Load conda into current shell
# conda config --set auto_activate_base false
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
# The environment_setup.sh also does `pip install -e ".[train]"` and `pip install -e ".[eval]"`, 
# which would install optional dependencies if defined in setup.py or pyproject.toml's project.optional-dependencies.
# We already installed them manually, but running this won't hurt if the setup files are configured for it.
# For pyproject.toml based projects, this usually means installing the current directory.