import torch
import torch.nn as nn
from llava.model.region_extractor.region_heads import (
    RegionClassifier, RegionClassifierLoss,
    DistanceHead, DistanceHeadLoss,
    LeftRightHead, LeftRightHeadLoss,
    MultipleChoiceHead, MultipleChoiceHeadLoss,
    CountingHead, CountingHeadLoss
)

def test_region_classifier_single():
    """Test RegionClassifier with single tensor input"""
    print("\nTesting RegionClassifier with single tensor input...")
    
    # Initialize classifier
    infeatures = 1152
    nclasses = 10
    classifier = RegionClassifier(infeatures=infeatures, nclasses=nclasses)
    
    # Create input tensor (10 regions)
    Nr = 10
    x = torch.randn(2*Nr, infeatures)  # (20, 1152)
    
    # Forward pass
    logits = classifier(x)
    
    # Check output shape
    assert logits.shape == (Nr, nclasses), \
        f"Expected output shape ({Nr}, {nclasses}), got {logits.shape}"
    print("✓ Output shape is correct")
    
    # Check that output is not all zeros
    assert not torch.allclose(logits, torch.zeros_like(logits)), \
        "Output is all zeros"
    print("✓ Output is not all zeros")
    
    # Check that output is not all ones
    assert not torch.allclose(logits, torch.ones_like(logits)), \
        "Output is all ones"
    print("✓ Output is not all ones")
    
    print("All tests passed for single tensor input!")

def test_region_classifier_batch():
    """Test RegionClassifier with batch input"""
    print("\nTesting RegionClassifier with batch input...")
    
    # Initialize classifier
    infeatures = 1152
    nclasses = 10
    classifier = RegionClassifier(infeatures=infeatures, nclasses=nclasses)
    
    # Create batch input (3 images with different number of regions)
    x_list = [
        torch.randn(20, infeatures),   # 10 regions
        torch.randn(30, infeatures),   # 15 regions
        torch.randn(40, infeatures)    # 20 regions
    ]
    
    # Forward pass
    all_logits = classifier(x_list)
    
    # Check output shape
    total_regions = sum(x.shape[0]//2 for x in x_list)  # 45 regions
    assert all_logits.shape == (total_regions, nclasses), \
        f"Expected output shape ({total_regions}, {nclasses}), got {all_logits.shape}"
    print("✓ Output shape is correct")
    
    # Test split_by_image
    num_regions = [10, 15, 20]
    image_logits = RegionClassifier.split_by_image(all_logits, num_regions)
    
    # Check number of tensors
    assert len(image_logits) == len(x_list), \
        f"Expected {len(x_list)} tensors, got {len(image_logits)}"
    print("✓ Correct number of split tensors")
    
    # Check shapes of split tensors
    for i, (logits, Nr) in enumerate(zip(image_logits, num_regions)):
        assert logits.shape == (Nr, nclasses), \
            f"Expected shape ({Nr}, {nclasses}) for image {i}, got {logits.shape}"
    print("✓ All split tensors have correct shapes")
    
    # Check that outputs are not all zeros
    assert not torch.allclose(all_logits, torch.zeros_like(all_logits)), \
        "Output is all zeros"
    print("✓ Output is not all zeros")
    
    # Check that outputs are not all ones
    assert not torch.allclose(all_logits, torch.ones_like(all_logits)), \
        "Output is all ones"
    print("✓ Output is not all ones")
    
    print("All tests passed for batch input!")

def test_region_classifier_edge_cases():
    """Test RegionClassifier with edge cases"""
    print("\nTesting RegionClassifier with edge cases...")
    
    # Initialize classifier
    infeatures = 1152
    nclasses = 10
    classifier = RegionClassifier(infeatures=infeatures, nclasses=nclasses)
    
    # Test with single region
    x = torch.randn(2, infeatures)  # 1 region
    logits = classifier(x)
    assert logits.shape == (1, nclasses), \
        f"Expected shape (1, {nclasses}), got {logits.shape}"
    print("✓ Works with single region")
    
    # Test with empty batch
    x_list = []
    all_logits = classifier(x_list)
    assert all_logits.shape == (0, nclasses), \
        f"Expected shape (0, {nclasses}), got {all_logits.shape}"
    print("✓ Works with empty batch")
    
    # Test split_by_image with empty tensor
    empty_logits = torch.empty(0, nclasses)
    empty_regions = []
    image_logits = RegionClassifier.split_by_image(empty_logits, empty_regions)
    assert len(image_logits) == 0, \
        f"Expected empty list, got list of length {len(image_logits)}"
    print("✓ Works with empty tensor")
    
    # Test with all zeros input
    x = torch.zeros(20, infeatures)  # 10 regions
    logits = classifier(x)
    assert logits.shape == (10, nclasses), \
        f"Expected shape (10, {nclasses}), got {logits.shape}"
    print("✓ Works with all zeros input")
    
    # Test with all ones input
    x = torch.ones(20, infeatures)  # 10 regions
    logits = classifier(x)
    assert logits.shape == (10, nclasses), \
        f"Expected shape (10, {nclasses}), got {logits.shape}"
    print("✓ Works with all ones input")
    
    print("All edge case tests passed!")

def test_region_classifier_errors():
    """Test RegionClassifier error handling"""
    print("\nTesting RegionClassifier error handling...")
    
    # Initialize classifier
    infeatures = 1152
    nclasses = 10
    classifier = RegionClassifier(infeatures=infeatures, nclasses=nclasses)
    
    # Test with odd number of rows
    x = torch.randn(21, infeatures)  # 10.5 regions
    try:
        classifier(x)
        assert False, "Should have raised AssertionError for odd number of rows"
    except AssertionError:
        print("✓ Correctly handles odd number of rows")
    
    # Test with wrong input dimension
    x = torch.randn(20, infeatures+1)  # wrong feature dimension
    try:
        classifier(x)
        assert False, "Should have raised RuntimeError for wrong feature dimension"
    except RuntimeError:
        print("✓ Correctly handles wrong feature dimension")
    
    # Test split_by_image with wrong total regions
    logits = torch.randn(45, nclasses)
    num_regions = [10, 15, 25]  # sum is 50, but logits has 45 rows
    try:
        RegionClassifier.split_by_image(logits, num_regions)
        assert False, "Should have raised AssertionError for mismatched total regions"
    except AssertionError:
        print("✓ Correctly handles mismatched total regions")
    
    print("All error handling tests passed!")

def test_region_classifier_loss():
    """Test RegionClassifierLoss"""
    print("\nTesting RegionClassifierLoss...")
    
    # Initialize loss function
    loss_fn = RegionClassifierLoss()
    
    # Test with single tensor
    nclasses = 10
    Nr = 5
    logits = torch.randn(Nr, nclasses)
    targets = torch.randint(0, nclasses, (Nr,))
    
    loss = loss_fn(logits, targets)
    assert isinstance(loss, torch.Tensor), "Loss should be a tensor"
    assert loss.dim() == 0, "Loss should be a scalar"
    print("✓ Works with single tensor")
    
    # Test with list of tensors
    logits_list = [
        torch.randn(3, nclasses),  # 3 regions
        torch.randn(4, nclasses),  # 4 regions
        torch.randn(2, nclasses)   # 2 regions
    ]
    targets_list = [
        torch.randint(0, nclasses, (3,)),
        torch.randint(0, nclasses, (4,)),
        torch.randint(0, nclasses, (2,))
    ]
    
    loss = loss_fn(logits_list, targets_list)
    assert isinstance(loss, torch.Tensor), "Loss should be a tensor"
    assert loss.dim() == 0, "Loss should be a scalar"
    print("✓ Works with list of tensors")
    
    # Test with single tensor logits and list targets
    all_logits = torch.cat(logits_list, dim=0)
    num_regions = [3, 4, 2]
    loss = loss_fn(all_logits, targets_list, num_regions)
    assert isinstance(loss, torch.Tensor), "Loss should be a tensor"
    assert loss.dim() == 0, "Loss should be a scalar"
    print("✓ Works with single tensor logits and list targets")
    
    # Test with different reduction modes
    loss_fn_none = RegionClassifierLoss(reduction='none')
    loss_fn_sum = RegionClassifierLoss(reduction='sum')
    
    loss_none = loss_fn_none(logits, targets)
    loss_sum = loss_fn_sum(logits, targets)
    
    assert loss_none.shape == (Nr,), f"Expected shape ({Nr},), got {loss_none.shape}"
    assert loss_sum.dim() == 0, "Loss should be a scalar"
    print("✓ Works with different reduction modes")
    
    # Test error handling
    try:
        # Mismatched list lengths
        loss_fn([logits], [targets, targets])
        assert False, "Should have raised AssertionError for mismatched list lengths"
    except AssertionError:
        print("✓ Correctly handles mismatched list lengths")
        
    try:
        # Wrong target type
        loss_fn(logits_list, targets)
        assert False, "Should have raised AssertionError for wrong target type"
    except AssertionError:
        print("✓ Correctly handles wrong target type")
        
    try:
        # Missing num_regions_per_image when needed
        loss_fn(all_logits, targets_list)
        assert False, "Should have raised AssertionError for missing num_regions_per_image"
    except AssertionError:
        print("✓ Correctly handles missing num_regions_per_image")
    
    print("All loss function tests passed!")

def test_distance_head_single():
    """Test DistanceHead with single pair input"""
    print("\nTesting DistanceHead with single pair input...")
    
    # Initialize head
    infeatures = 256
    head = DistanceHead(infeatures=infeatures)
    
    # Create input tensors (2 regions)
    rgb_features = torch.randn(2, infeatures)  # (2, 256)
    depth_features = torch.randn(2, infeatures)  # (2, 256)
    
    # Forward pass
    distance = head((rgb_features, depth_features))
    
    # Check output
    assert isinstance(distance, torch.Tensor), "Output should be a tensor"
    assert distance.dim() == 0, "Output should be a scalar"
    print("✓ Output is correct scalar tensor")
    
    # Check that output is not all zeros
    assert not torch.allclose(distance, torch.zeros_like(distance)), \
        "Output is all zeros"
    print("✓ Output is not all zeros")
    
    print("All tests passed for single pair input!")

def test_distance_head_batch():
    """Test DistanceHead with batch input"""
    print("\nTesting DistanceHead with batch input...")
    
    # Initialize head
    infeatures = 256
    head = DistanceHead(infeatures=infeatures)
    
    # Create batch input (3 pairs)
    batch = [
        (torch.randn(2, infeatures), torch.randn(2, infeatures)),  # pair 1
        (torch.randn(2, infeatures), torch.randn(2, infeatures)),  # pair 2
        (torch.randn(2, infeatures), torch.randn(2, infeatures))   # pair 3
    ]
    
    # Forward pass
    distances = head(batch)
    
    # Check output
    assert isinstance(distances, torch.Tensor), "Output should be a tensor"
    assert distances.shape == (3,), f"Expected shape (3,), got {distances.shape}"
    print("✓ Output shape is correct")
    
    # Check that outputs are not all zeros
    assert not torch.allclose(distances, torch.zeros_like(distances)), \
        "Output is all zeros"
    print("✓ Output is not all zeros")
    
    print("All tests passed for batch input!")

def test_distance_head_edge_cases():
    """Test DistanceHead with edge cases"""
    print("\nTesting DistanceHead with edge cases...")
    
    # Initialize head
    infeatures = 256
    head = DistanceHead(infeatures=infeatures)
    
    # Test with empty batch
    empty_batch = []
    distances = head(empty_batch)
    assert distances.shape == (0,), f"Expected shape (0,), got {distances.shape}"
    print("✓ Works with empty batch")
    
    # Test with all zeros input
    rgb_features = torch.zeros(2, infeatures)
    depth_features = torch.zeros(2, infeatures)
    distance = head((rgb_features, depth_features))
    assert distance.dim() == 0, "Output should be a scalar"
    print("✓ Works with all zeros input")
    
    # Test with all ones input
    rgb_features = torch.ones(2, infeatures)
    depth_features = torch.ones(2, infeatures)
    distance = head((rgb_features, depth_features))
    assert distance.dim() == 0, "Output should be a scalar"
    print("✓ Works with all ones input")
    
    print("All edge case tests passed!")

def test_distance_head_errors():
    """Test DistanceHead error handling"""
    print("\nTesting DistanceHead error handling...")
    
    # Initialize head
    infeatures = 256
    head = DistanceHead(infeatures=infeatures)
    
    # Test with wrong number of regions
    rgb_features = torch.randn(3, infeatures)  # 3 regions
    depth_features = torch.randn(2, infeatures)  # 2 regions
    try:
        head((rgb_features, depth_features))
        assert False, "Should have raised RuntimeError for mismatched region counts"
    except RuntimeError:
        print("✓ Correctly handles mismatched region counts")
    
    # Test with wrong feature dimension
    rgb_features = torch.randn(2, infeatures+1)  # wrong feature dim
    depth_features = torch.randn(2, infeatures)
    try:
        head((rgb_features, depth_features))
        assert False, "Should have raised RuntimeError for wrong feature dimension"
    except RuntimeError:
        print("✓ Correctly handles wrong feature dimension")
    
    print("All error handling tests passed!")

def test_distance_head_loss():
    """Test DistanceHeadLoss"""
    print("\nTesting DistanceHeadLoss...")
    
    # Initialize loss function
    loss_fn = DistanceHeadLoss()
    
    # Test with single prediction
    prediction = torch.tensor(2.5)
    target = torch.tensor(2.0)
    
    loss = loss_fn(prediction, target)
    assert isinstance(loss, torch.Tensor), "Loss should be a tensor"
    assert loss.dim() == 0, "Loss should be a scalar"
    print("✓ Works with single prediction")
    
    # Test with batch predictions
    predictions = torch.tensor([2.5, 3.5, 1.5])
    targets = torch.tensor([2.0, 3.0, 1.0])
    
    loss = loss_fn(predictions, targets)
    assert isinstance(loss, torch.Tensor), "Loss should be a tensor"
    assert loss.dim() == 0, "Loss should be a scalar"
    print("✓ Works with batch predictions")
    
    # Test with different reduction modes
    loss_fn_none = DistanceHeadLoss(reduction='none')
    loss_fn_sum = DistanceHeadLoss(reduction='sum')
    
    loss_none = loss_fn_none(predictions, targets)
    loss_sum = loss_fn_sum(predictions, targets)
    
    assert loss_none.shape == (3,), f"Expected shape (3,), got {loss_none.shape}"
    assert loss_sum.dim() == 0, "Loss should be a scalar"
    print("✓ Works with different reduction modes")
    
    # Test error handling
    try:
        # Mismatched shapes
        loss_fn(predictions[:2], targets)  # predictions has 2 elements, targets has 3
        assert False, "Should have raised RuntimeError for mismatched shapes"
    except RuntimeError:
        print("✓ Correctly handles mismatched shapes")
    
    print("All loss function tests passed!")

def test_leftright_head_single():
    """Test LeftRightHead with single pair input"""
    print("\nTesting LeftRightHead with single pair input...")
    
    # Initialize head
    infeatures = 256
    head = LeftRightHead(infeatures=infeatures)
    
    # Create input tensors (2 regions)
    rgb_features = torch.randn(2, infeatures)  # (2, 256)
    depth_features = torch.randn(2, infeatures)  # (2, 256)
    
    # Forward pass
    logits = head((rgb_features, depth_features))
    
    # Check output
    assert isinstance(logits, torch.Tensor), "Output should be a tensor"
    assert logits.shape == (2,), f"Expected shape (2,), got {logits.shape}"
    print("✓ Output shape is correct")
    
    # Check that output is not all zeros
    assert not torch.allclose(logits, torch.zeros_like(logits)), \
        "Output is all zeros"
    print("✓ Output is not all zeros")
    
    print("All tests passed for single pair input!")

def test_leftright_head_batch():
    """Test LeftRightHead with batch input"""
    print("\nTesting LeftRightHead with batch input...")
    
    # Initialize head
    infeatures = 256
    head = LeftRightHead(infeatures=infeatures)
    
    # Create batch input (3 pairs)
    batch = [
        (torch.randn(2, infeatures), torch.randn(2, infeatures)),  # pair 1
        (torch.randn(2, infeatures), torch.randn(2, infeatures)),  # pair 2
        (torch.randn(2, infeatures), torch.randn(2, infeatures))   # pair 3
    ]
    
    # Forward pass
    logits = head(batch)
    
    # Check output
    assert isinstance(logits, torch.Tensor), "Output should be a tensor"
    assert logits.shape == (3, 2), f"Expected shape (3, 2), got {logits.shape}"
    print("✓ Output shape is correct")
    
    # Check that outputs are not all zeros
    assert not torch.allclose(logits, torch.zeros_like(logits)), \
        "Output is all zeros"
    print("✓ Output is not all zeros")
    
    print("All tests passed for batch input!")

def test_leftright_head_edge_cases():
    """Test LeftRightHead with edge cases"""
    print("\nTesting LeftRightHead with edge cases...")
    
    # Initialize head
    infeatures = 256
    head = LeftRightHead(infeatures=infeatures)
    
    # Test with empty batch
    empty_batch = []
    logits = head(empty_batch)
    assert logits.shape == (0, 2), f"Expected shape (0, 2), got {logits.shape}"
    print("✓ Works with empty batch")
    
    # Test with all zeros input
    rgb_features = torch.zeros(2, infeatures)
    depth_features = torch.zeros(2, infeatures)
    logits = head((rgb_features, depth_features))
    assert logits.shape == (2,), f"Expected shape (2,), got {logits.shape}"
    print("✓ Works with all zeros input")
    
    # Test with all ones input
    rgb_features = torch.ones(2, infeatures)
    depth_features = torch.ones(2, infeatures)
    logits = head((rgb_features, depth_features))
    assert logits.shape == (2,), f"Expected shape (2,), got {logits.shape}"
    print("✓ Works with all ones input")
    
    print("All edge case tests passed!")

def test_leftright_head_errors():
    """Test LeftRightHead error handling"""
    print("\nTesting LeftRightHead error handling...")
    
    # Initialize head
    infeatures = 256
    head = LeftRightHead(infeatures=infeatures)
    
    # Test with wrong number of regions
    rgb_features = torch.randn(3, infeatures)  # 3 regions
    depth_features = torch.randn(2, infeatures)  # 2 regions
    try:
        head((rgb_features, depth_features))
        assert False, "Should have raised RuntimeError for mismatched region counts"
    except RuntimeError:
        print("✓ Correctly handles mismatched region counts")
    
    # Test with wrong feature dimension
    rgb_features = torch.randn(2, infeatures+1)  # wrong feature dim
    depth_features = torch.randn(2, infeatures)
    try:
        head((rgb_features, depth_features))
        assert False, "Should have raised RuntimeError for wrong feature dimension"
    except RuntimeError:
        print("✓ Correctly handles wrong feature dimension")
    
    print("All error handling tests passed!")

def test_leftright_head_loss():
    """Test LeftRightHeadLoss"""
    print("\nTesting LeftRightHeadLoss...")
    
    # Initialize loss function
    loss_fn = LeftRightHeadLoss()
    
    # Test with single prediction
    logits = torch.tensor([1.0, -1.0])  # (2,)
    target = torch.tensor(0)  # LEFT class
    
    loss = loss_fn(logits, target)
    assert isinstance(loss, torch.Tensor), "Loss should be a tensor"
    assert loss.dim() == 0, "Loss should be a scalar"
    print("✓ Works with single prediction")
    
    # Test with batch predictions
    logits = torch.tensor([
        [1.0, -1.0],  # LEFT
        [-1.0, 1.0],  # RIGHT
        [1.0, -1.0]   # LEFT
    ])  # (3, 2)
    targets = torch.tensor([0, 1, 0])  # (3,)
    
    loss = loss_fn(logits, targets)
    assert isinstance(loss, torch.Tensor), "Loss should be a tensor"
    assert loss.dim() == 0, "Loss should be a scalar"
    print("✓ Works with batch predictions")
    
    # Test with different reduction modes
    loss_fn_none = LeftRightHeadLoss(reduction='none')
    loss_fn_sum = LeftRightHeadLoss(reduction='sum')
    
    loss_none = loss_fn_none(logits, targets)
    loss_sum = loss_fn_sum(logits, targets)
    
    assert loss_none.shape == (3,), f"Expected shape (3,), got {loss_none.shape}"
    assert loss_sum.dim() == 0, "Loss should be a scalar"
    print("✓ Works with different reduction modes")
    
    # Test error handling
    try:
        # Wrong target shape for single prediction
        loss_fn(logits[0], targets)  # logits (2,) but targets (3,)
        assert False, "Should have raised RuntimeError for wrong target shape"
    except RuntimeError as e:
        assert "scalar" in str(e), "Error message should mention scalar target"
        print("✓ Correctly handles wrong target shape for single prediction")
        
    try:
        # Mismatched batch sizes
        loss_fn(logits[:2], targets)  # logits (2,2) but targets (3,)
        assert False, "Should have raised RuntimeError for mismatched batch sizes"
    except RuntimeError as e:
        assert "mismatch" in str(e), "Error message should mention batch size mismatch"
        print("✓ Correctly handles mismatched batch sizes")
    
    print("All loss function tests passed!")

def test_multiplechoice_head_single():
    """Test MultipleChoiceHead with single image input"""
    print("\nTesting MultipleChoiceHead with single image input...")
    
    # Initialize head
    infeatures = 256
    head = MultipleChoiceHead(infeatures=infeatures)
    
    # Create input tensors (5 regions)
    rgb_features = torch.randn(5, infeatures)  # (5, 256)
    depth_features = torch.randn(5, infeatures)  # (5, 256)
    
    # Forward pass
    logits = head((rgb_features, depth_features))
    
    # Check output
    assert isinstance(logits, torch.Tensor), "Output should be a tensor"
    assert logits.shape == (5,), f"Expected shape (5,), got {logits.shape}"
    print("✓ Output shape is correct")
    
    # Check that output is not all zeros
    assert not torch.allclose(logits, torch.zeros_like(logits)), \
        "Output is all zeros"
    print("✓ Output is not all zeros")
    
    print("All tests passed for single image input!")

def test_multiplechoice_head_batch():
    """Test MultipleChoiceHead with batch input"""
    print("\nTesting MultipleChoiceHead with batch input...")
    
    # Initialize head
    infeatures = 256
    head = MultipleChoiceHead(infeatures=infeatures)
    
    # Create batch input (3 images with different number of regions)
    batch = [
        (torch.randn(3, infeatures), torch.randn(3, infeatures)),  # 3 regions
        (torch.randn(5, infeatures), torch.randn(5, infeatures)),  # 5 regions
        (torch.randn(4, infeatures), torch.randn(4, infeatures))   # 4 regions
    ]
    
    # Forward pass
    logits = head(batch)
    
    # Check output
    assert isinstance(logits, torch.Tensor), "Output should be a tensor"
    assert logits.shape == (3, 5), f"Expected shape (3, 5), got {logits.shape}"
    print("✓ Output shape is correct")
    
    # Check that outputs are not all zeros
    assert not torch.allclose(logits, torch.zeros_like(logits)), \
        "Output is all zeros"
    print("✓ Output is not all zeros")
    
    # Check that padding positions are masked
    attention_mask = torch.tensor([
        [True]*3 + [False]*2,  # 3 regions + 2 padding
        [True]*5,              # 5 regions
        [True]*4 + [False]*1   # 4 regions + 1 padding
    ])
    assert torch.all(logits[~attention_mask] == float('-inf')), \
        "Padding positions should be masked with -inf"
    print("✓ Padding positions are correctly masked")
    
    print("All tests passed for batch input!")

def test_multiplechoice_head_edge_cases():
    """Test MultipleChoiceHead with edge cases"""
    print("\nTesting MultipleChoiceHead with edge cases...")
    
    # Initialize head
    infeatures = 256
    head = MultipleChoiceHead(infeatures=infeatures)
    
    # Test with empty batch
    empty_batch = []
    logits = head(empty_batch)
    assert logits.shape == (0,), f"Expected shape (0,), got {logits.shape}"
    print("✓ Works with empty batch")
    
    # Test with single region
    rgb_features = torch.randn(1, infeatures)
    depth_features = torch.randn(1, infeatures)
    logits = head((rgb_features, depth_features))
    assert logits.shape == (1,), f"Expected shape (1,), got {logits.shape}"
    print("✓ Works with single region")
    
    # Test with all zeros input
    rgb_features = torch.zeros(5, infeatures)
    depth_features = torch.zeros(5, infeatures)
    logits = head((rgb_features, depth_features))
    assert logits.shape == (5,), f"Expected shape (5,), got {logits.shape}"
    print("✓ Works with all zeros input")
    
    # Test with all ones input
    rgb_features = torch.ones(5, infeatures)
    depth_features = torch.ones(5, infeatures)
    logits = head((rgb_features, depth_features))
    assert logits.shape == (5,), f"Expected shape (5,), got {logits.shape}"
    print("✓ Works with all ones input")
    
    print("All edge case tests passed!")

def test_multiplechoice_head_errors():
    """Test MultipleChoiceHead error handling"""
    print("\nTesting MultipleChoiceHead error handling...")
    
    # Initialize head
    infeatures = 256
    head = MultipleChoiceHead(infeatures=infeatures)
    
    # Test with mismatched region counts
    rgb_features = torch.randn(5, infeatures)  # 5 regions
    depth_features = torch.randn(4, infeatures)  # 4 regions
    try:
        head((rgb_features, depth_features))
        assert False, "Should have raised RuntimeError for mismatched region counts"
    except RuntimeError:
        print("✓ Correctly handles mismatched region counts")
    
    # Test with wrong feature dimension
    rgb_features = torch.randn(5, infeatures+1)  # wrong feature dim
    depth_features = torch.randn(5, infeatures)
    try:
        head((rgb_features, depth_features))
        assert False, "Should have raised RuntimeError for wrong feature dimension"
    except RuntimeError:
        print("✓ Correctly handles wrong feature dimension")
    
    print("All error handling tests passed!")

def test_multiplechoice_head_loss():
    """Test MultipleChoiceHeadLoss"""
    print("\nTesting MultipleChoiceHeadLoss...")
    
    # Initialize loss function
    loss_fn = MultipleChoiceHeadLoss()
    
    # Test with single image
    logits = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])  # (5,)
    target = torch.tensor(2)  # select region 2
    
    loss = loss_fn(logits, target)
    assert isinstance(loss, torch.Tensor), "Loss should be a tensor"
    assert loss.dim() == 0, "Loss should be a scalar"
    print("✓ Works with single image")
    
    # Test with batch
    logits = torch.tensor([
        [1.0, 2.0, 3.0, float('-inf'), float('-inf')],  # 3 regions
        [1.0, 2.0, 3.0, 4.0, 5.0],                      # 5 regions
        [1.0, 2.0, 3.0, 4.0, float('-inf')]             # 4 regions
    ])  # (3, 5)
    targets = torch.tensor([1, 3, 2])  # select regions 1, 3, 2
    
    loss = loss_fn(logits, targets)
    assert isinstance(loss, torch.Tensor), "Loss should be a tensor"
    assert loss.dim() == 0, "Loss should be a scalar"
    print("✓ Works with batch")
    
    # Test with different reduction modes
    loss_fn_none = MultipleChoiceHeadLoss(reduction='none')
    loss_fn_sum = MultipleChoiceHeadLoss(reduction='sum')
    
    loss_none = loss_fn_none(logits, targets)
    loss_sum = loss_fn_sum(logits, targets)
    
    assert loss_none.shape == (3,), f"Expected shape (3,), got {loss_none.shape}"
    assert loss_sum.dim() == 0, "Loss should be a scalar"
    print("✓ Works with different reduction modes")
    
    # Test error handling
    try:
        # Wrong target shape for single image
        loss_fn(logits[0], targets)  # logits (5,) but targets (3,)
        assert False, "Should have raised RuntimeError for wrong target shape"
    except RuntimeError as e:
        assert "scalar" in str(e), "Error message should mention scalar target"
        print("✓ Correctly handles wrong target shape for single image")
        
    try:
        # Mismatched batch sizes
        loss_fn(logits[:2], targets)  # logits (2,5) but targets (3,)
        assert False, "Should have raised RuntimeError for mismatched batch sizes"
    except RuntimeError as e:
        assert "mismatch" in str(e), "Error message should mention batch size mismatch"
        print("✓ Correctly handles mismatched batch sizes")
    
    print("All loss function tests passed!")

def test_counting_head_single():
    """Test CountingHead with single image input"""
    print("\nTesting CountingHead with single image input...")
    
    # Initialize head
    infeatures = 256
    head = CountingHead(infeatures=infeatures)
    
    # Create input tensors (5 regions)
    rgb_features = torch.randn(5, infeatures)  # (5, 256)
    depth_features = torch.randn(5, infeatures)  # (5, 256)
    
    # Forward pass
    logits = head((rgb_features, depth_features))
    
    # Check output
    assert isinstance(logits, torch.Tensor), "Output should be a tensor"
    assert logits.shape == (5,), f"Expected shape (5,), got {logits.shape}"
    print("✓ Output shape is correct")
    
    # Check that output is not all zeros
    assert not torch.allclose(logits, torch.zeros_like(logits)), \
        "Output is all zeros"
    print("✓ Output is not all zeros")
    
    print("All tests passed for single image input!")

def test_counting_head_batch():
    """Test CountingHead with batch input"""
    print("\nTesting CountingHead with batch input...")
    
    # Initialize head
    infeatures = 256
    head = CountingHead(infeatures=infeatures)
    
    # Create batch input (3 images with different number of regions)
    batch = [
        (torch.randn(3, infeatures), torch.randn(3, infeatures)),  # 3 regions
        (torch.randn(5, infeatures), torch.randn(5, infeatures)),  # 5 regions
        (torch.randn(4, infeatures), torch.randn(4, infeatures))   # 4 regions
    ]
    
    # Forward pass
    logits = head(batch)
    
    # Check output
    assert isinstance(logits, torch.Tensor), "Output should be a tensor"
    assert logits.shape == (3, 5), f"Expected shape (3, 5), got {logits.shape}"
    print("✓ Output shape is correct")
    
    # Check that outputs are not all zeros
    assert not torch.allclose(logits, torch.zeros_like(logits)), \
        "Output is all zeros"
    print("✓ Output is not all zeros")
    
    # Check that padding positions are masked
    attention_mask = torch.tensor([
        [True]*3 + [False]*2,  # 3 regions + 2 padding
        [True]*5,              # 5 regions
        [True]*4 + [False]*1   # 4 regions + 1 padding
    ])
    assert torch.all(logits[~attention_mask] == float('-inf')), \
        "Padding positions should be masked with -inf"
    print("✓ Padding positions are correctly masked")
    
    print("All tests passed for batch input!")

def test_counting_head_edge_cases():
    """Test CountingHead with edge cases"""
    print("\nTesting CountingHead with edge cases...")
    
    # Initialize head
    infeatures = 256
    head = CountingHead(infeatures=infeatures)
    
    # Test with empty batch
    empty_batch = []
    logits = head(empty_batch)
    assert logits.shape == (0,), f"Expected shape (0,), got {logits.shape}"
    print("✓ Works with empty batch")
    
    # Test with single region
    rgb_features = torch.randn(1, infeatures)
    depth_features = torch.randn(1, infeatures)
    logits = head((rgb_features, depth_features))
    assert logits.shape == (1,), f"Expected shape (1,), got {logits.shape}"
    print("✓ Works with single region")
    
    # Test with all zeros input
    rgb_features = torch.zeros(5, infeatures)
    depth_features = torch.zeros(5, infeatures)
    logits = head((rgb_features, depth_features))
    assert logits.shape == (5,), f"Expected shape (5,), got {logits.shape}"
    print("✓ Works with all zeros input")
    
    # Test with all ones input
    rgb_features = torch.ones(5, infeatures)
    depth_features = torch.ones(5, infeatures)
    logits = head((rgb_features, depth_features))
    assert logits.shape == (5,), f"Expected shape (5,), got {logits.shape}"
    print("✓ Works with all ones input")
    
    print("All edge case tests passed!")

def test_counting_head_errors():
    """Test CountingHead error handling"""
    print("\nTesting CountingHead error handling...")
    
    # Initialize head
    infeatures = 256
    head = CountingHead(infeatures=infeatures)
    
    # Test with mismatched region counts
    rgb_features = torch.randn(5, infeatures)  # 5 regions
    depth_features = torch.randn(4, infeatures)  # 4 regions
    try:
        head((rgb_features, depth_features))
        assert False, "Should have raised RuntimeError for mismatched region counts"
    except RuntimeError:
        print("✓ Correctly handles mismatched region counts")
    
    # Test with wrong feature dimension
    rgb_features = torch.randn(5, infeatures+1)  # wrong feature dim
    depth_features = torch.randn(5, infeatures)
    try:
        head((rgb_features, depth_features))
        assert False, "Should have raised RuntimeError for wrong feature dimension"
    except RuntimeError:
        print("✓ Correctly handles wrong feature dimension")
    
    print("All error handling tests passed!")

def test_counting_head_loss():
    """Test CountingHeadLoss"""
    print("\nTesting CountingHeadLoss...")
    
    # Initialize loss function
    loss_fn = CountingHeadLoss()
    
    # Test with single image
    logits = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])  # (5,)
    target = torch.tensor(2)  # count is 2
    
    loss = loss_fn(logits, target)
    assert isinstance(loss, torch.Tensor), "Loss should be a tensor"
    assert loss.dim() == 0, "Loss should be a scalar"
    print("✓ Works with single image")
    
    # Test with batch
    logits = torch.tensor([
        [1.0, 2.0, 3.0, float('-inf'), float('-inf')],  # 3 regions
        [1.0, 2.0, 3.0, 4.0, 5.0],                      # 5 regions
        [1.0, 2.0, 3.0, 4.0, float('-inf')]             # 4 regions
    ])  # (3, 5)
    targets = torch.tensor([1, 3, 2])  # counts are 1, 3, 2
    
    loss = loss_fn(logits, targets)
    assert isinstance(loss, torch.Tensor), "Loss should be a tensor"
    assert loss.dim() == 0, "Loss should be a scalar"
    print("✓ Works with batch")
    
    # Test with different reduction modes
    loss_fn_none = CountingHeadLoss(reduction='none')
    loss_fn_sum = CountingHeadLoss(reduction='sum')
    
    loss_none = loss_fn_none(logits, targets)
    loss_sum = loss_fn_sum(logits, targets)
    
    assert loss_none.shape == (3,), f"Expected shape (3,), got {loss_none.shape}"
    assert loss_sum.dim() == 0, "Loss should be a scalar"
    print("✓ Works with different reduction modes")
    
    # Test error handling
    try:
        # Wrong target shape for single image
        loss_fn(logits[0], targets)  # logits (5,) but targets (3,)
        assert False, "Should have raised RuntimeError for wrong target shape"
    except RuntimeError as e:
        assert "scalar" in str(e), "Error message should mention scalar target"
        print("✓ Correctly handles wrong target shape for single image")
        
    try:
        # Mismatched batch sizes
        loss_fn(logits[:2], targets)  # logits (2,5) but targets (3,)
        assert False, "Should have raised RuntimeError for mismatched batch sizes"
    except RuntimeError as e:
        assert "mismatch" in str(e), "Error message should mention batch size mismatch"
        print("✓ Correctly handles mismatched batch sizes")
    
    print("All loss function tests passed!")

if __name__ == "__main__":
    # Run all tests
    test_region_classifier_single()
    test_region_classifier_batch()
    test_region_classifier_edge_cases()
    test_region_classifier_errors()
    test_region_classifier_loss()
    
    test_distance_head_single()
    test_distance_head_batch()
    test_distance_head_edge_cases()
    test_distance_head_errors()
    test_distance_head_loss()
    
    test_leftright_head_single()
    test_leftright_head_batch()
    test_leftright_head_edge_cases()
    test_leftright_head_errors()
    test_leftright_head_loss()
    
    test_multiplechoice_head_single()
    test_multiplechoice_head_batch()
    test_multiplechoice_head_edge_cases()
    test_multiplechoice_head_errors()
    test_multiplechoice_head_loss()
    
    test_counting_head_single()
    test_counting_head_batch()
    test_counting_head_edge_cases()
    test_counting_head_errors()
    test_counting_head_loss()
    
    print("\nAll tests completed successfully!") 