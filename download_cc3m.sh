#!/usr/bin/env bash

# Download CC3M for first stage of training process
git lfs install

git clone https://huggingface.co/datasets/liuhaotian/LLaVA-CC3M-Pretrain-595K

cd LLaVA-CC3M-Pretrain-595K

unzip image.zip