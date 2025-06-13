import torch
import torch.nn as nn
from llava.model.region_extractor.region_transformer import RegionTransformer, CrossAttention, RegionFeatureExtractor

def test_region_transformer():
    # Test parameters
    num_regions = 3
    dim = 1152
    
    # Test case 1: 2D input (no batch dimension)
    print("\nTesting 2D input case:")
    rgb_features_2d = torch.randn(num_regions, dim)
    depth_features_2d = torch.randn(num_regions, dim)
    mask_2d = torch.zeros((1, num_regions), dtype=torch.bool)
    
    transformer = RegionTransformer(dim=dim)
    transformed_2d = transformer(rgb_features_2d, depth_features_2d, mask_2d)
    
    assert transformed_2d.shape == (2 * num_regions, dim), \
        f"2D case: Expected shape (2*{num_regions}, {dim}), got {transformed_2d.shape}"
    
    rgb_out_2d, depth_out_2d = transformer.get_region_features(transformed_2d)
    assert rgb_out_2d.shape == (num_regions, dim), \
        f"2D case: Expected RGB shape ({num_regions}, {dim}), got {rgb_out_2d.shape}"
    assert depth_out_2d.shape == (num_regions, dim), \
        f"2D case: Expected depth shape ({num_regions}, {dim}), got {depth_out_2d.shape}"
    
    # Test case 2: 3D input (with batch dimension)
    print("\nTesting 3D input case:")
    batch_size = 2
    rgb_features_3d = torch.randn(batch_size, num_regions, dim)
    depth_features_3d = torch.randn(batch_size, num_regions, dim)
    mask_3d = torch.zeros((batch_size, num_regions), dtype=torch.bool)
    
    transformed_3d = transformer(rgb_features_3d, depth_features_3d, mask_3d)
    
    assert transformed_3d.shape == (batch_size, 2 * num_regions, dim), \
        f"3D case: Expected shape ({batch_size}, 2*{num_regions}, {dim}), got {transformed_3d.shape}"
    
    rgb_out_3d, depth_out_3d = transformer.get_region_features(transformed_3d)
    assert rgb_out_3d.shape == (batch_size, num_regions, dim), \
        f"3D case: Expected RGB shape ({batch_size}, {num_regions}, {dim}), got {rgb_out_3d.shape}"
    assert depth_out_3d.shape == (batch_size, num_regions, dim), \
        f"3D case: Expected depth shape ({batch_size}, {num_regions}, {dim}), got {depth_out_3d.shape}"
    
    print("RegionTransformer test passed for both 2D and 3D cases!")

def test_cross_attention():
    # Test parameters
    num_regions = 3
    num_tokens = 10
    dim = 1152
    
    # Test case 1: 2D input
    print("\nTesting 2D input case:")
    region_features_2d = torch.randn(2 * num_regions, dim)
    image_features_2d = torch.randn(num_tokens, dim)
    mask_2d = torch.ones(num_tokens, dtype=torch.bool)
    
    cross_attn = CrossAttention(dim=dim)
    attended_2d = cross_attn(region_features_2d, image_features_2d, mask_2d)
    
    assert attended_2d.shape == (2 * num_regions, dim), \
        f"2D case: Expected shape (2*{num_regions}, {dim}), got {attended_2d.shape}"
    
    # Test case 2: 3D input
    print("\nTesting 3D input case:")
    batch_size = 2
    region_features_3d = torch.randn(batch_size, 2 * num_regions, dim)
    image_features_3d = torch.randn(batch_size, num_tokens, dim)
    mask_3d = torch.ones(batch_size, num_tokens, dtype=torch.bool)
    
    attended_3d = cross_attn(region_features_3d, image_features_3d, mask_3d)
    
    assert attended_3d.shape == (batch_size, 2 * num_regions, dim), \
        f"3D case: Expected shape ({batch_size}, 2*{num_regions}, {dim}), got {attended_3d.shape}"
    
    print("CrossAttention test passed for both 2D and 3D cases!")

def test_region_feature_extractor():
    # Test parameters
    num_regions = 3
    num_tokens = 10
    dim = 1152
    
    # Test case 1: 2D input
    print("\nTesting 2D input case:")
    rgb_features_2d = torch.randn(num_regions, dim)
    depth_features_2d = torch.randn(num_regions, dim)
    image_features_2d = torch.randn(num_tokens, dim)
    mask_2d = torch.ones(num_tokens, dtype=torch.bool)
    
    extractor = RegionFeatureExtractor(
        dim=dim,
        num_heads=8,
        num_transformer_layers=2,
        num_cross_attn_layers=1
    )
    
    enhanced_2d = extractor(rgb_features_2d, depth_features_2d, image_features_2d, mask_2d)
    
    assert enhanced_2d.shape == (2 * num_regions, dim), \
        f"2D case: Expected shape (2*{num_regions}, {dim}), got {enhanced_2d.shape}"
    
    # Test case 2: 3D input
    print("\nTesting 3D input case:")
    batch_size = 2
    rgb_features_3d = torch.randn(batch_size, num_regions, dim)
    depth_features_3d = torch.randn(batch_size, num_regions, dim)
    image_features_3d = torch.randn(batch_size, num_tokens, dim)
    mask_3d = torch.ones(batch_size, num_tokens, dtype=torch.bool)
    
    enhanced_3d = extractor(rgb_features_3d, depth_features_3d, image_features_3d, mask_3d)
    
    assert enhanced_3d.shape == (batch_size, 2 * num_regions, dim), \
        f"3D case: Expected shape ({batch_size}, 2*{num_regions}, {dim}), got {enhanced_3d.shape}"
    
    # Test case 3: No mask
    print("\nTesting without mask:")
    enhanced_no_mask = extractor(rgb_features_2d, depth_features_2d, image_features_2d)
    assert enhanced_no_mask.shape == (2 * num_regions, dim), \
        f"No mask case: Expected shape (2*{num_regions}, {dim}), got {enhanced_no_mask.shape}"
    
    print("RegionFeatureExtractor test passed for all cases!")

def test_end_to_end():
    # Test parameters
    num_regions = 3
    num_tokens = 10
    dim = 1152
    num_classes = 10
    
    # Test case 1: 2D input
    print("\nTesting 2D input case:")
    rgb_features_2d = torch.randn(num_regions, dim)
    depth_features_2d = torch.randn(num_regions, dim)
    image_features_2d = torch.randn(num_tokens, dim)
    
    extractor = RegionFeatureExtractor(
        dim=dim,
        num_heads=8,
        num_transformer_layers=2,
        num_cross_attn_layers=1
    )
    task_head = TaskHead(dim=dim, num_classes=num_classes)
    
    enhanced_2d = extractor(rgb_features_2d, depth_features_2d, image_features_2d)
    
    task_types = ['distance', 'left_right', 'multiple_choice', 'counting']
    for task_type in task_types:
        output_2d = task_head(enhanced_2d, task_type)
        print(f"2D case: End-to-end test for {task_type} passed!")
    
    # Test case 2: 3D input
    print("\nTesting 3D input case:")
    batch_size = 2
    rgb_features_3d = torch.randn(batch_size, num_regions, dim)
    depth_features_3d = torch.randn(batch_size, num_regions, dim)
    image_features_3d = torch.randn(batch_size, num_tokens, dim)
    
    enhanced_3d = extractor(rgb_features_3d, depth_features_3d, image_features_3d)
    
    for task_type in task_types:
        output_3d = task_head(enhanced_3d, task_type)
        print(f"3D case: End-to-end test for {task_type} passed!")
    
    print("All end-to-end tests passed for both 2D and 3D cases!")

if __name__ == "__main__":
    # Run individual tests
    #test_region_transformer()
    #test_cross_attention()
    test_region_feature_extractor()
    