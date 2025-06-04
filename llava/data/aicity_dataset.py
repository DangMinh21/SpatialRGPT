# In a new file, e.g., /workspace/SpatialRGPT/llava/data/aicity_dataset.py
import copy
import json
import logging
import os
import random
import re  # For the <mask> <depth> substitution
from typing import Dict

import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

# Assuming these are accessible from the SpatialRGPT/VILA environment
from llava.constants import DEFAULT_IMAGE_TOKEN  # IMAGE_TOKEN_INDEX is used by tokenizer_image_token
from llava.train.args import DataArguments  # Or specific training_args if needed for image_processor access
from llava.mm_utils import process_image, process_depth, process_masks

# The 'preprocess' function for conversations is also in dataset.py from LLaVA
from llava.data.dataset import preprocess  # Importing from the provided dataset.py
from llava import conversation as conversation_lib
from llava.constants import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX,
)


def preprocess_multimodal(sources, data_args):
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for source in sources:
        concat_values = "".join([sentence["value"] for sentence in source])
        for sid, sentence in enumerate(source):
            # In multimodal conversations, we automatically prepend '<image>' at the start of the first sentence if it doesn't already contain one.
            if sid == 0 and DEFAULT_IMAGE_TOKEN not in concat_values:
                sentence["value"] = f"{DEFAULT_IMAGE_TOKEN}\n" + sentence["value"]
            if DEFAULT_IMAGE_TOKEN in sentence["value"]:
                sentence_chunks = [chunk.strip() for chunk in sentence["value"].split(DEFAULT_IMAGE_TOKEN)]
                sentence_chunks = [
                    chunk + " " if not (chunk.endswith("\n")) else chunk for chunk in sentence_chunks[:-1]
                ] + [sentence_chunks[-1]]
                sentence["value"] = f"{DEFAULT_IMAGE_TOKEN}\n".join(sentence_chunks).strip()

                replace_token = DEFAULT_IMAGE_TOKEN
                if "mmtag" in conversation_lib.default_conversation.version:
                    replace_token = "<Image>" + replace_token + "</Image>"
                if data_args.mm_use_im_start_end:
                    replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
                sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)
            # ensure every DEFAULT_IMAGE_TOKEN is followed by a newline character.
            # If it has one already, we don't add another one.
            if DEFAULT_IMAGE_TOKEN in sentence["value"]:
                sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, f"{DEFAULT_IMAGE_TOKEN}\n")
                sentence["value"] = sentence["value"].replace(f"{DEFAULT_IMAGE_TOKEN}\n\n", f"{DEFAULT_IMAGE_TOKEN}\n")

    return sources


class AICityLazySpatialDataset(Dataset):
    def __init__(
        self,
        data_path: str,  # Path to your processed JSONL file
        rgb_image_folder: str,  # Base path to AI City RGB images, e.g., /workspace/aicity_challenge_data_raw/train/images/
        depth_image_folder: str,  # Base path to AI City Depth images, e.g., /workspace/aicity_challenge_data_raw/train/depths/
        tokenizer: PreTrainedTokenizer,
        data_args: DataArguments,  # Contains image_processor, image_aspect_ratio etc.
        # training_args: TrainingArguments, # Not directly used in __getitem__ but often in main script
    ):
        super().__init__()
        logging.warning(f"Loading AI City Spatial Dataset from: {data_path}")

        try:
            with open(data_path, "r") as f:
                self.list_data_dict = [json.loads(line) for line in f]
        except json.JSONDecodeError:  # If it's a plain JSON list and not JSONL
            with open(data_path, "r") as f:
                self.list_data_dict = json.load(f)

        print(f"Total AI City Spatial Samples: {len(self.list_data_dict)} from {data_path}")
        logging.warning("Formatting inputs... (Lazy mode, actual processing in __getitem__)")

        self.tokenizer = tokenizer
        self.data_args = data_args
        self.rgb_image_folder = rgb_image_folder
        self.depth_image_folder = depth_image_folder
        self.enable_depth = True  # Always true for AI City Challenge dataset

    def __len__(self):
        return len(self.list_data_dict)

    # Optional: You can add lengths properties if your sampler needs them,
    # similar to LazySupervisedSpatialDataset, but often not strictly necessary.

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        # Each 'sources' dict is one line from your processed JSONL
        sources_dict = self.list_data_dict[i]

        # Make a copy for modification if preprocess_multimodal modifies in-place
        # preprocess_multimodal expects a list of conversations list: [[conv1_turn1, conv1_turn2], [conv2_turn1,...]]
        # Here, one item from list_data_dict IS one conversation
        current_conversations = copy.deepcopy(sources_dict["conversations"])

        # 1. Process Image (RGB)
        # The 'filename' key in our JSONL should be the base name like "029734"
        # AI City images are .png
        rgb_image_basename = sources_dict["image_base_filename"] + ".png"
        rgb_image, image_info = process_image(
            rgb_image_basename,
            self.data_args,  # Contains image_processor, image_aspect_ratio
            self.rgb_image_folder,
            return_info=True,  # We need image_info for process_masks
        )

        # 2. Process Depth Image
        # AI City depth images are like "029734_depth.png"
        depth_image_basename = sources_dict["image_base_filename"] + "_depth.png"
        try:
            # process_depth is designed to take a path and load/process it.
            # It uses data_args.image_processor internally.
            # Crucially, ensure the image_processor handles single-channel depth correctly
            # (usually by converting to RGB for ViTs). Standard CLIP/SigLIP processors do.
            depth_image_tensor = process_depth(depth_image_basename, self.data_args, self.depth_image_folder)
        except Exception as e:
            print(
                f"Error loading or processing depth image {depth_image_basename} for sample {sources_dict['id']}: {e}"
            )
            # Fallback: return a dummy depth tensor of the correct expected size
            # This size depends on the image_processor's output
            # Assuming image_processor outputs [C, H, W]
            img_proc_output_size = self.data_args.image_processor.size
            # For SigLIP, it might be data_args.image_processor.size['shortest_edge'] or similar logic
            # Defaulting to typical 384x384, 3 channels if size info is tricky
            h = img_proc_output_size.get("height", 384)
            w = img_proc_output_size.get("width", 384)
            depth_image_tensor = torch.zeros((3, h, w), dtype=torch.float16)  # Match image_processor output
            print(f"Using dummy depth tensor for sample {sources_dict['id']}")

        # 3. Process Masks (RLE)
        # process_masks expects `sources` to be a list containing the data dict for the current sample.
        # It also expects `image_info` which we got from process_image.
        # The RLEs are at sources_dict['rle']

        # process_masks expects a list of source dictionaries for its first argument.
        # It internally uses sources[0]['rle'] or sources[0]['bbox'].
        masks_tensor = process_masks([sources_dict], self.data_args, image_info)

        # 4. Process Conversations for Text and Labels
        # preprocess_multimodal adds <image>\n if needed, and handles <im_start/end> tokens
        # The <mask> -> <mask> <depth> substitution should already be done when creating the JSONL.
        processed_source_conversations = preprocess_multimodal(
            [current_conversations], self.data_args  # Needs to be a list of conversation lists
        )

        # preprocess applies the conversation template (e.g., LLaMA3, Vicuna)
        # and creates input_ids and labels (masking human turns in labels)
        data_dict = preprocess(
            processed_source_conversations,  # This is now a list containing ONE processed conversation
            self.tokenizer,
            has_image=True,  # We always have an image
            # no_system_prompt might be from data_args if needed by specific conv_template
            no_system_prompt=getattr(self.data_args, "vflan_no_system_prompt", False),
        )

        # preprocess returns a dict where input_ids and labels are lists of tensors (one per source).
        # Since we pass one source, we take the first element.
        data_dict = dict(input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0])

        # 5. Assemble final data_dict
        # The collator expects 'image', 'depths', 'masks'
        # Ensure tensors are unsqueezed to have a batch-like dim if process_image/depth/masks don't provide it
        # process_image and process_depth already return [C, H, W]
        # process_masks returns [num_masks, H_proc, W_proc]

        data_dict["image"] = rgb_image.unsqueeze(0)  # -> [1, C, H, W]
        data_dict["depths"] = depth_image_tensor.unsqueeze(0)  # -> [1, C, H, W]
        data_dict["masks"] = masks_tensor  # -> [num_masks, H_proc, W_proc] (DataCollator will wrap in list)

        return data_dict
