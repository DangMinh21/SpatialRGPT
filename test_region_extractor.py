# flow in prepare_inputs_labels_for_multimodal() function
import torch
from transformers import PretrainedConfig

from region_extractor import RegionExtractor, RegionExtractorConfig
from llava.model.region_extractor.region_transformer import RegionFeatureExtractor
from llava.model.region_extractor.region_heads import RegionClassifier, DistanceHead, MultipleChoiceHead, CountingHead, LeftRightHead


print("\n--- 1. Defining Inputs & Config ---")
# --- Simulation Parameters ---
BATCH_SIZE = 2
NUM_MASKS_SAMPLE_1 = 5
NUM_MASKS_SAMPLE_2 = 3
TOTAL_MASKS = NUM_MASKS_SAMPLE_1 + NUM_MASKS_SAMPLE_2
GLOBAL_IMG_PATCHES = 729 # (e.g., 27x27 grid from ada_pooling in original RegionExtractor)
MM_HIDDEN_SIZE = 1152 # Feature dimension from Vision Tower / RegionExtractor
LLM_HIDDEN_SIZE = 4096 # Feature dimension for the LLM

# --- Simulated Tensors ---
# These represent the outputs from the Vision Tower
Fimg_rgb_patches = torch.randn(BATCH_SIZE, GLOBAL_IMG_PATCHES, MM_HIDDEN_SIZE)
Fimg_depth_patches = torch.randn(BATCH_SIZE, GLOBAL_IMG_PATCHES, MM_HIDDEN_SIZE)

# This represents the list of binary masks from the dataloader
masks_list = [
    torch.rand(NUM_MASKS_SAMPLE_1, 384, 384), # 5 masks for sample 1
    torch.rand(NUM_MASKS_SAMPLE_2, 384, 384)  # 3 masks for sample 2
]

print(f"Simulated Fimg_rgb (global patches): {Fimg_rgb_patches.shape}")
print(f"Simulated Fimg_depth (global patches): {Fimg_depth_patches.shape}")
print(f"Simulated masks_list: {len(masks_list)} items, first item shape: {masks_list[0].shape}, \n\t\t\t\tsecond item shape: {masks_list[1].shape}")


# --- Model Configurations ---
region_extractor_cfg = RegionExtractorConfig(region_extractor_type="regiongpt")
config = PretrainedConfig()
config.mm_hidden_size = MM_HIDDEN_SIZE
config.hidden_size = LLM_HIDDEN_SIZE # LLM dimension

print("\n--- 2. Instantiate Modules ---")
# --- Instantiate Original and New Modules ---
# This is the original module responsible for pooling features from masks
region_extractor = RegionExtractor(region_extractor_cfg, config).eval()

region_feature_extractor_new = RegionFeatureExtractor(
    dim=MM_HIDDEN_SIZE, # Operates on 1152-dim features
    num_heads=8,        
    num_transformer_layers=6, 
    num_cross_attn_layers=1   
).eval()

# initialize head
# RegionHead for classify region
num_region_classes = 10 # Example: pallet, shelf, transporter, etc.
max_object_count = 15   # Example: max number of objects to count in a scene
region_head = RegionClassifier(
    infeatures=MM_HIDDEN_SIZE, # Takes the 1152-dim features
    nclasses=num_region_classes,
).eval()

# DistanceHead for messure the distance of 2 region
distance_head = DistanceHead(
    infeatures = MM_HIDDEN_SIZE
).eval()

# LeftRightHead for classify leftright of 2 region
leftright_head = LeftRightHead(
    infeatures=MM_HIDDEN_SIZE
).eval()

multiplechoice_head = MultipleChoiceHead(
    infeatures=MM_HIDDEN_SIZE
).eval()

counting_head = CountingHead(
    infeatures=MM_HIDDEN_SIZE
).eval()
print("Modules instantiated successfully.")

print("\n--- 3. Simulating Forward Pass Data Flow ---")
print("\n--- Step A: Feature Pooling (Simulating RegionExtractor) ---")
hres_tower_features_rgb, lres_tower_features_rgb = region_extractor.feature_refinement(Fimg_rgb_patches)
print(f"hres_tower_features_rgb (for RGB mask pooling): {hres_tower_features_rgb.shape}")
print(f"lres_tower_features_rgb (for global projector): {lres_tower_features_rgb.shape}")
print(f"Fimg_depth_patches (for depth mask pooling): {Fimg_depth_patches.shape}")


unprojected_rgb_regions_list = region_extractor.mask_pooling(hres_tower_features_rgb, masks_list, return_list=True)
unprojected_depth_regions_list = region_extractor.mask_pooling(Fimg_depth_patches, masks_list, return_list=True)
print(f"Unprojected RGB region features (sample 0 - 1): {unprojected_rgb_regions_list[0].shape, unprojected_rgb_regions_list[1].shape}") # Should be [NUM_MASKS_SAMPLE, 1152]
print(f"Unprojected Depth region features (sample 0): {unprojected_depth_regions_list[0].shape, unprojected_depth_regions_list[1].shape}")# Should be [NUM_MASKS_SAMPLE, 1152]

print("\n--- Step B: Region Interaction (RegionFeatureExtractor) ---")
# For batch processing, Need pad and stack these.
# Let's test with the first sample for simplicity.
unprojected_rgb_regions_s0 = unprojected_rgb_regions_list[0]
unprojected_depth_regions_s0 = unprojected_depth_regions_list[0]

global_context_features_s0 = lres_tower_features_rgb[0]


# module enhances region features using self- and cross-attention
enhanced_region_features_s0 = region_feature_extractor_new(
    rgb_features=unprojected_rgb_regions_s0,
    depth_features=unprojected_depth_regions_s0,
    image_features=global_context_features_s0
)
print(f"Enhanced region features (sample 0): {enhanced_region_features_s0.shape}") # Should be [2 * NUM_MASKS_SAMPLE_1, 1152]
print("\n--- Step C: Branching for LLM and Auxiliary Heads ---")

print("\n  --- Branch 1 (LLM Pathway) ---")
# Split the enhanced features back into RGB and Depth
num_masks_s0 = unprojected_rgb_regions_s0.shape[0]
enhanced_rgb_s0 = enhanced_region_features_s0[:num_masks_s0]
enhanced_depth_s0 = enhanced_region_features_s0[num_masks_s0:]
print(f"Enhanced RGB region features (sample 0): {enhanced_rgb_s0.shape}") # Should be [NUM_MASKS_SAMPLE_1, 1152]
print(f"Enhanced Depth region features (sample 0): {enhanced_depth_s0.shape}") # Should be [NUM_MASKS_SAMPLE_1, 1152]


# Pass through the original projectors to get LLM-compatible embeddings
projected_rgb_for_llm_s0 = region_extractor.rgb_projector(enhanced_rgb_s0)
projected_depth_for_llm_s0 = region_extractor.depth_projector(enhanced_depth_s0)
print(f"Projected RGB for LLM (sample 0): {projected_rgb_for_llm_s0.shape}") # Should be [num_masks, 4096]
print(f"Projected Depth for LLM (sample 0): {projected_depth_for_llm_s0.shape}") # Should be [num_masks, 4096]

# --- Branch 2: Features for the Auxiliary Task Heads ---
print("\n  --- Branch 2 (Auxiliary Heads) ---")
# The task_heads module takes the enhanced features and applies different heads.
# Let's test each head individually.

# 2.1 - Region Classifier: take all region on image and classify each region
region_class_logits = region_head(enhanced_region_features_s0)
print(f"Region Classifier inputs (sample 0): {enhanced_region_features_s0.shape}") # Should be [num_masks*2, 1152]
print(f"Region Classifier logits (sample 0): {region_class_logits.shape}") # Should be [num_masks, num_region_classes]

# 2.2.1 - Distance Head: take 2 region and predict distance for each region
# This head expects features for exactly two regions.
# Let's simulate taking the first two regions from the enhanced features.
two_region_features = (enhanced_region_features_s0[[0, 1]],enhanced_region_features_s0[[0+num_masks_s0, 1+num_masks_s0]]) # Grabbing RGB and Depth for first 2 masks
distance_prediction = distance_head(two_region_features)
print(f"Distance Head inputs (sample 0): tuple of {len(two_region_features)} - for RGB: {two_region_features[0].shape} - for Depth: {two_region_features[1].shape}") # tuple of (rgb_features, depth_features) each of shape (2, F)
print(f"Distance Head prediction (sample 0): {distance_prediction}") # Should be sclar (a single distance value)


# 2.2.2 - Left/Right Head: take 2 region and predict left/right for each region
left_right_logits = leftright_head(two_region_features)
print(f"Left/Right Head inputs (sample 0): tuple of {len(two_region_features)} - for RGB: {two_region_features[0].shape} - for Depth: {two_region_features[1].shape}") # tuple of (rgb_features, depth_features) each of shape (2, F)
print(f"Left/Right Head logits (sample 0): {left_right_logits}") # Should be [1, 2] (logits for left/right classes)


# 2.2.3 - MultipleChoice Head: This head takes features for all choices and predicts which one is correct.
all_region_features = (enhanced_region_features_s0[:num_masks_s0], enhanced_region_features_s0[num_masks_s0:])
mcq_logits = multiplechoice_head(all_region_features)
print(f"MultipleChoice Head inputs (sample 0): tuple of {len(all_region_features)} - for RGB: {all_region_features[0].shape} - for Depth: {all_region_features[1].shape}") # tuple of (rgb_features, depth_features) each of shape (Nr, F)
print(f"MultipleChoice Head logits (sample 0): {mcq_logits}") # Should be [num_masks]

# 2.2.4 - Counting Head: This head fuses all region features and predicts a count.
count_logits = counting_head(all_region_features)
print(f"Counting Head inputs (sample 0): tuple of {len(all_region_features)} - for RGB: {all_region_features[0].shape} - for Depth: {all_region_features[1].shape}") # tuple of (rgb_features, depth_features) each of shape (Nr, F)
print(f"Counting Head logits (sample 0): {count_logits}")

print("\n--- Data Flow Test Complete ---")

# ========= Output =============
# --- 1. Defining Inputs & Config ---
# Simulated Fimg_rgb (global patches): torch.Size([2, 729, 1152])
# Simulated Fimg_depth (global patches): torch.Size([2, 729, 1152])
# Simulated masks_list: 2 items, first item shape: torch.Size([5, 384, 384]), 
# 				second item shape: torch.Size([3, 384, 384])

# --- 2. Instantiate Modules ---
# Modules instantiated successfully.

# --- 3. Simulating Forward Pass Data Flow ---

# --- Step A: Feature Pooling (Simulating RegionExtractor) ---
# hres_tower_features_rgb (for RGB mask pooling): torch.Size([2, 11664, 1152])
# lres_tower_features_rgb (for global projector): torch.Size([2, 729, 1152])
# Fimg_depth_patches (for depth mask pooling): torch.Size([2, 729, 1152])
# Unprojected RGB region features (sample 0 - 1): (torch.Size([5, 1152]), torch.Size([3, 1152]))
# Unprojected Depth region features (sample 0): (torch.Size([5, 1152]), torch.Size([3, 1152]))

# --- Step B: Region Interaction (RegionFeatureExtractor) ---
# Enhanced region features (sample 0): torch.Size([10, 1152])

# --- Step C: Branching for LLM and Auxiliary Heads ---

#   --- Branch 1 (LLM Pathway) ---
# Enhanced RGB region features (sample 0): torch.Size([5, 1152])
# Enhanced Depth region features (sample 0): torch.Size([5, 1152])
# Projected RGB for LLM (sample 0): torch.Size([5, 4096])
# Projected Depth for LLM (sample 0): torch.Size([5, 4096])

#   --- Branch 2 (Auxiliary Heads) ---
# Region Classifier inputs (sample 0): torch.Size([10, 1152])
# Region Classifier logits (sample 0): torch.Size([5, 10])
# Distance Head inputs (sample 0): tuple of 2 - for RGB: torch.Size([2, 1152]) - for Depth: torch.Size([2, 1152])
# Distance Head prediction (sample 0): 0.24084824323654175
# Left/Right Head inputs (sample 0): tuple of 2 - for RGB: torch.Size([2, 1152]) - for Depth: torch.Size([2, 1152])
# Left/Right Head logits (sample 0): tensor([-0.1226,  0.1650], grad_fn=<SqueezeBackward1>)
# MultipleChoice Head inputs (sample 0): tuple of 2 - for RGB: torch.Size([5, 1152]) - for Depth: torch.Size([5, 1152])
# MultipleChoice Head logits (sample 0): tensor([-0.1161, -0.1154, -0.1120, -0.1189, -0.1098],
#        grad_fn=<SqueezeBackward1>)
# Counting Head inputs (sample 0): tuple of 2 - for RGB: torch.Size([5, 1152]) - for Depth: torch.Size([5, 1152])
# Counting Head logits (sample 0): tensor([-0.0229, -0.0239, -0.0272, -0.0175, -0.0212],
#        grad_fn=<SqueezeBackward1>)

# --- Data Flow Test Complete ---