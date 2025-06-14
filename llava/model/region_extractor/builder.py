# This file is modified from https://github.com/haotian-liu/LLaVA/

import os

import torch
import torch.nn as nn
from transformers import PretrainedConfig, PreTrainedModel

from .base_extractor import RegionExtractor, RegionExtractorConfig
from .region_transformer import RegionFeatureExtractor, RegionFeatureExtractorConfig
from.region_heads import RegionClassifier, RegionClassifierConfig


def build_region_extractor(model_type_or_path: str, config: PretrainedConfig) -> PreTrainedModel:
    if model_type_or_path is None:
        return None

    ## load from pretrained model
    if config.resume_path and os.path.exists(model_type_or_path):
        print("Resuming region extractor from: ", model_type_or_path)
        return RegionExtractor.from_pretrained(model_type_or_path, config, torch_dtype=eval(config.model_dtype))
    else:
        print("Build region extractor from scratch.")
        region_extractor_cfg = RegionExtractorConfig(model_type_or_path)
        region_extractor = RegionExtractor(region_extractor_cfg, config).to(eval(config.model_dtype))
        return region_extractor
    
# def build_region_enhancer(enhancer_cfg, config):
#     """
#     Args:
#         enhancer_cfg: Cấu hình cho RegionFeatureExtractor
#         config: Cấu hình chung của model
#     """
#     return RegionFeatureExtractor(
#         dim=config.hidden_size,  # hoặc dim phù hợp
#         num_heads=enhancer_cfg.get('num_heads', 8),
#         num_transformer_layers=enhancer_cfg.get('num_transformer_layers', 6),
#         num_cross_attn_layers=enhancer_cfg.get('num_cross_attn_layers', 1),
#         dropout=enhancer_cfg.get('dropout', 0.1),
#         activation=enhancer_cfg.get('activation', 'gelu')
#     )
def build_region_enhancer(enhancer_cfg, config):
    # This pattern checks if we are resuming from a checkpoint
    if config.resume_path and os.path.exists(os.path.join(config.resume_path, "region_enhancer")):
        print("Resuming region enhancer from:", os.path.join(config.resume_path, "region_enhancer"))
        return RegionFeatureExtractor.from_pretrained(
            os.path.join(config.resume_path, "region_enhancer"),
            torch_dtype=eval(config.model_dtype)
        )
    else:
        # Build from scratch
        print("Building region enhancer from scratch.")
        cfg = RegionFeatureExtractorConfig(**enhancer_cfg)
        return RegionFeatureExtractor(cfg).to(eval(config.model_dtype))

# def build_region_classifier(classifier_cfg, config):
#     """
#     Args:
#         classifier_cfg: Cấu hình cho RegionClassifier
#         config: Cấu hình chung của model
#     """
#     return RegionClassifier(
#         in_dim=config.hidden_size,
#         num_classes=classifier_cfg.get('num_classes', 4),
#         dropout=classifier_cfg.get('dropout', 0.1)
#     )
def build_region_classifier(classifier_cfg, config):
    if config.resume_path and os.path.exists(os.path.join(config.resume_path, "region_classifier")):
        print("Resuming region classifier from:", os.path.join(config.resume_path, "region_classifier"))
        return RegionClassifier.from_pretrained(
            os.path.join(config.resume_path, "region_classifier"),
            torch_dtype=eval(config.model_dtype)
        )
    else:
        print("Building region classifier from scratch.")
        cfg = RegionClassifierConfig(**classifier_cfg)
        return RegionClassifier(cfg).to(eval(config.model_dtype))
    
    
