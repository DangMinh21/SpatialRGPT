import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Union, Optional, Tuple
from torch.nn.utils.rnn import pad_sequence

class RegionClassifierLoss(nn.Module):
    def __init__(self, reduction: str = 'mean'):
        """
        Loss function for RegionClassifier using CrossEntropy.
        
        Args:
            reduction: How to reduce the loss ('none', 'mean', or 'sum')
        """
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(reduction=reduction)
        
    def forward(
        self,
        logits: Union[torch.Tensor, List[torch.Tensor]],
        targets: Union[torch.Tensor, List[torch.Tensor]],
        num_regions_per_image: Optional[List[int]] = None
    ) -> torch.Tensor:
        """
        Compute loss between predictions and targets.
        
        Args:
            logits: Either a tensor of shape (N, nclasses) or a list of B tensors
                   where N is total number of regions across all images
            targets: Either a tensor of shape (N,) or a list of B tensors
                    containing class indices for each region
            num_regions_per_image: Optional list of number of regions for each image.
                                 Required only if logits is a single tensor and targets is a list.
            
        Returns:
            Loss value
        """
        if isinstance(logits, list):
            # Handle list of tensors
            assert isinstance(targets, list), "If logits is a list, targets must also be a list"
            assert len(logits) == len(targets), "Number of logits and targets must match"
            
            # Concatenate all logits and targets
            all_logits = torch.cat(logits, dim=0)
            all_targets = torch.cat(targets, dim=0)
            
        else:
            # Handle single tensor
            if isinstance(targets, list):
                # If targets is a list but logits is a tensor, we need num_regions_per_image
                assert num_regions_per_image is not None, \
                    "num_regions_per_image is required when logits is a single tensor but targets is a list"
                all_logits = logits
                all_targets = torch.cat(targets, dim=0)
            else:
                # Both logits and targets are tensors
                all_logits = logits
                all_targets = targets
            
        # Compute loss
        return self.criterion(all_logits, all_targets)

class RegionClassifier(nn.Module):
    def __init__(
        self,
        infeatures: int,
        nclasses: int,
        hidden_dims: List[int] = [512, 256],
        dropout: float = 0.1
    ):
        """
        Region classifier for classifying regions into different types.
        
        Args:
            infeatures: Input feature dimension
            nclasses: Number of region classes
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout rate
        """
        super().__init__()
        self.infeatures = infeatures
        self.nclasses = nclasses
        
        # Build MLP layers
        layers = []
        prev_dim = infeatures
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
            
        # Output layer
        layers.append(nn.Linear(prev_dim, nclasses))
        
        self.mlp = nn.Sequential(*layers)
        
    def _fuse_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Fuse RGB and depth features by taking mean.
        
        Args:
            x: Tensor of shape (2*Nr, F)
            
        Returns:
            Fused features of shape (Nr, F)
        """
        # Split into RGB and depth features
        Nr = x.shape[0] // 2
        frgb = x[:Nr]  # (Nr, F)
        fdepth = x[Nr:]  # (Nr, F)
        
        # Fuse by taking mean
        fused = (frgb + fdepth) / 2  # (Nr, F)
        return fused
        
    def _process_batch(self, x_list: List[torch.Tensor]) -> Tuple[torch.Tensor, List[int]]:
        """
        Process a batch of tensors by concatenating them and keeping track of image indices.
        
        Args:
            x_list: List of B tensors, each of shape (2*Nr, F)
            
        Returns:
            Tuple of:
            - Concatenated features of shape (N, F) where N is total number of regions
            - List of image indices for each region
        """
        # Handle empty batch
        if not x_list:
            return torch.empty(0, self.infeatures), []
            
        # Validate input shapes
        for i, x in enumerate(x_list):
            assert x.dim() == 2, f"Expected 2D tensor for image {i}, got {x.dim()}D"
            assert x.shape[0] % 2 == 0, f"First dimension must be even for image {i}, got {x.shape[0]}"
            
        # Create image indices for each region
        image_indices = []
        for i, x in enumerate(x_list):
            Nr = x.shape[0] // 2
            image_indices.extend([i] * Nr)
            
        # Concatenate all features
        fused_features = []
        for x in x_list:
            fused = self._fuse_features(x)
            fused_features.append(fused)
            
        # Stack all fused features
        all_features = torch.cat(fused_features, dim=0)  # (N, F)
        
        return all_features, image_indices
        
    def forward(self, x: Union[torch.Tensor, List[torch.Tensor]]) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Either a tensor of shape (2*Nr, F) or a list of B tensors, each of shape (2*Nr, F)
               where Nr is number of regions, F is feature dimension
            
        Returns:
            Tensor of shape (N, nclasses) where N is total number of regions across all images
        """
        if isinstance(x, list):
            # Process batch
            all_features, _ = self._process_batch(x)
            
            # Handle empty batch
            if all_features.shape[0] == 0:
                return torch.empty(0, self.nclasses)
                
        else:
            # Process single tensor
            assert x.dim() == 2, f"Expected 2D tensor, got {x.dim()}D"
            assert x.shape[0] % 2 == 0, f"First dimension must be even, got {x.shape[0]}"
            all_features = self._fuse_features(x)
            
        # Apply MLP
        return self.mlp(all_features)
        
    @staticmethod
    def split_by_image(logits: torch.Tensor, num_regions_per_image: List[int]) -> List[torch.Tensor]:
        """
        Split logits tensor into list of tensors, one per image.
        
        Args:
            logits: Tensor of shape (N, nclasses) where N is total number of regions
            num_regions_per_image: List of number of regions for each image
            
        Returns:
            List of tensors, each of shape (Nr, nclasses) where Nr is number of regions in that image
        """
        # Handle empty case
        if logits.shape[0] == 0:
            return []
            
        assert logits.dim() == 2, f"Expected 2D tensor, got {logits.dim()}D"
        assert sum(num_regions_per_image) == logits.shape[0], \
            f"Sum of regions per image ({sum(num_regions_per_image)}) must match logits shape ({logits.shape[0]})"
            
        # Split logits into per-image tensors
        start_idx = 0
        outputs = []
        for Nr in num_regions_per_image:
            image_logits = logits[start_idx:start_idx + Nr]
            outputs.append(image_logits)
            start_idx += Nr
            
        return outputs

class DistanceHead(nn.Module):
    def __init__(
        self,
        infeatures: int,
        hidden_dim: int = 256,
        dropout: float = 0.1
    ):
        """
        Head for predicting distances between regions.
        
        Args:
            infeatures: Input feature dimension
            hidden_dim: Hidden layer dimension
            dropout: Dropout rate
        """
        super().__init__()
        self.infeatures = infeatures
        
        # Feature fusion dimension (4x input dim due to concatenation)
        fused_dim = infeatures * 4
        
        # MLP for distance prediction
        self.mlp = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        
    def _fuse_features(self, rgb_features: torch.Tensor, depth_features: torch.Tensor) -> torch.Tensor:
        """
        Fuse RGB and depth features for a pair of regions.
        
        Args:
            rgb_features: RGB features of shape (2, F)
            depth_features: Depth features of shape (2, F)
            
        Returns:
            Fused features of shape (4*F,)
        """
        # Take mean of RGB and depth features
        fused = (rgb_features + depth_features) / 2  # (2, F)
        
        # Split into individual region features
        f1, f2 = fused[0], fused[1]  # each (F,)
        
        # Create combined feature vector
        combined = torch.cat([
            f1,           # (F,)
            f2,           # (F,)
            f1 - f2,      # (F,)
            f1 * f2       # (F,)
        ])  # (4*F,)
        
        return combined
        
    def _process_batch(self, features_list: List[Tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        """
        Process a batch of region pairs.
        
        Args:
            features_list: List of B tuples, each containing (rgb_features, depth_features)
                          where each tensor has shape (2, F)
            
        Returns:
            Combined features of shape (B, 4*F)
        """
        # Handle empty batch
        if not features_list:
            return torch.empty(0, self.infeatures * 4)
            
        # Process each pair
        fused_features = []
        for rgb_features, depth_features in features_list:
            fused = self._fuse_features(rgb_features, depth_features)
            fused_features.append(fused)
            
        # Stack all features
        return torch.stack(fused_features)  # (B, 4*F)
        
    def forward(
        self,
        x: Union[Tuple[torch.Tensor, torch.Tensor], List[Tuple[torch.Tensor, torch.Tensor]]]
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Either a tuple of (rgb_features, depth_features) each of shape (2, F),
               or a list of B such tuples
            
        Returns:
            Either a scalar tensor for single pair, or a tensor of shape (B,) for batch
        """
        if isinstance(x, list):
            # Process batch
            features = self._process_batch(x)
            
            # Handle empty batch
            if features.shape[0] == 0:
                return torch.empty(0)
                
        else:
            # Process single pair
            rgb_features, depth_features = x
            features = self._fuse_features(rgb_features, depth_features)
            features = features.unsqueeze(0)  # (1, 4*F)
            
        # Predict distance
        distances = self.mlp(features)  # (B, 1) or (1, 1)
        distances = distances.squeeze(-1)  # (B,) or (1,)
        
        # For single pair, return scalar
        if not isinstance(x, list):
            distances = distances.squeeze(0)  # scalar
            
        return distances

class DistanceHeadLoss(nn.Module):
    def __init__(self, reduction: str = 'mean'):
        """
        Loss function for DistanceHead using MSE.
        
        Args:
            reduction: How to reduce the loss ('none', 'mean', or 'sum')
        """
        super().__init__()
        self.criterion = nn.MSELoss(reduction=reduction)
        
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute loss between predictions and targets.
        
        Args:
            predictions: Either a scalar tensor or a tensor of shape (B,)
            targets: Either a scalar tensor or a tensor of shape (B,)
            
        Returns:
            Loss value
        """
        # Ensure predictions and targets have the same shape
        if predictions.shape != targets.shape:
            raise RuntimeError(
                f"Predictions shape {predictions.shape} does not match targets shape {targets.shape}"
            )
            
        return self.criterion(predictions, targets)

class LeftRightHead(nn.Module):
    def __init__(
        self,
        infeatures: int,
        hidden_dim: int = 256,
        dropout: float = 0.1
    ):
        """
        Head for predicting left-right relationship between regions.
        
        Args:
            infeatures: Input feature dimension
            hidden_dim: Hidden layer dimension
            dropout: Dropout rate
        """
        super().__init__()
        self.infeatures = infeatures
        
        # Feature fusion dimension (4x input dim due to concatenation)
        fused_dim = infeatures * 4
        
        # MLP for left-right prediction
        self.mlp = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2)  # 2 classes: LEFT, RIGHT
        )
        
    def _fuse_features(self, rgb_features: torch.Tensor, depth_features: torch.Tensor) -> torch.Tensor:
        """
        Fuse RGB and depth features for a pair of regions.
        
        Args:
            rgb_features: RGB features of shape (2, F)
            depth_features: Depth features of shape (2, F)
            
        Returns:
            Fused features of shape (4*F,)
        """
        # Take mean of RGB and depth features
        fused = (rgb_features + depth_features) / 2  # (2, F)
        
        # Split into individual region features
        f1, f2 = fused[0], fused[1]  # each (F,)
        
        # Create combined feature vector
        combined = torch.cat([
            f1,           # (F,)
            f2,           # (F,)
            f1 - f2,      # (F,)
            f1 * f2       # (F,)
        ])  # (4*F,)
        
        return combined
        
    def _process_batch(self, features_list: List[Tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        """
        Process a batch of region pairs.
        
        Args:
            features_list: List of B tuples, each containing (rgb_features, depth_features)
                          where each tensor has shape (2, F)
            
        Returns:
            Combined features of shape (B, 4*F)
        """
        # Handle empty batch
        if not features_list:
            return torch.empty(0, self.infeatures * 4)
            
        # Process each pair
        fused_features = []
        for rgb_features, depth_features in features_list:
            fused = self._fuse_features(rgb_features, depth_features)
            fused_features.append(fused)
            
        # Stack all features
        return torch.stack(fused_features)  # (B, 4*F)
        
    def forward(
        self,
        x: Union[Tuple[torch.Tensor, torch.Tensor], List[Tuple[torch.Tensor, torch.Tensor]]]
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Either a tuple of (rgb_features, depth_features) each of shape (2, F),
               or a list of B such tuples
            
        Returns:
            Either a tensor of shape (2,) for single pair, or a tensor of shape (B, 2) for batch
        """
        if isinstance(x, list):
            # Process batch
            features = self._process_batch(x)
            
            # Handle empty batch
            if features.shape[0] == 0:
                return torch.empty(0, 2)
                
        else:
            # Process single pair
            rgb_features, depth_features = x
            features = self._fuse_features(rgb_features, depth_features)
            features = features.unsqueeze(0)  # (1, 4*F)
            
        # Predict left-right relationship
        logits = self.mlp(features)  # (B, 2) or (1, 2)
        
        # For single pair, remove batch dimension
        if not isinstance(x, list):
            logits = logits.squeeze(0)  # (2,)
            
        return logits

class LeftRightHeadLoss(nn.Module):
    def __init__(self, reduction: str = 'mean'):
        """
        Loss function for LeftRightHead using CrossEntropy.
        
        Args:
            reduction: How to reduce the loss ('none', 'mean', or 'sum')
        """
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(reduction=reduction)
        
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute loss between predictions and targets.
        
        Args:
            logits: Either a tensor of shape (2,) for single pair,
                   or a tensor of shape (B, 2) for batch
            targets: Either a scalar tensor (0 or 1) for single pair,
                    or a tensor of shape (B,) for batch
            
        Returns:
            Loss value
        """
        # Ensure logits and targets have correct shapes
        if logits.dim() == 1:  # single pair
            if targets.dim() != 0:
                raise RuntimeError(
                    f"For single pair, targets should be scalar, got shape {targets.shape}"
                )
            logits = logits.unsqueeze(0)  # (1, 2)
            targets = targets.unsqueeze(0)  # (1,)
        else:  # batch
            if logits.shape[0] != targets.shape[0]:
                raise RuntimeError(
                    f"Batch size mismatch: logits {logits.shape[0]} vs targets {targets.shape[0]}"
                )
            
        return self.criterion(logits, targets)

class MultipleChoiceHead(nn.Module):
    def __init__(
        self,
        infeatures: int,
        hidden_dims: List[int] = [512, 256],
        dropout: float = 0.1,
        fusion_type: str = 'mean'  # 'mean' or 'concat'
    ):
        """
        Head for multiple choice prediction.
        
        Args:
            infeatures: Input feature dimension
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout rate
            fusion_type: How to fuse RGB and depth features ('mean' or 'concat')
        """
        super().__init__()
        self.infeatures = infeatures
        self.fusion_type = fusion_type
        
        # Feature fusion dimension
        if fusion_type == 'mean':
            fused_dim = infeatures
        else:  # concat
            fused_dim = infeatures * 2
            
        # Build MLP layers
        layers = []
        prev_dim = fused_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
            
        # Output layer (single value per region)
        layers.append(nn.Linear(prev_dim, 1))
        
        self.mlp = nn.Sequential(*layers)
        
    def _fuse_features(self, rgb_features: torch.Tensor, depth_features: torch.Tensor) -> torch.Tensor:
        """
        Fuse RGB and depth features.
        
        Args:
            rgb_features: RGB features of shape (Nr, F)
            depth_features: Depth features of shape (Nr, F)
            
        Returns:
            Fused features of shape (Nr, F) or (Nr, 2*F)
        """
        if self.fusion_type == 'mean':
            return (rgb_features + depth_features) / 2
        else:  # concat
            return torch.cat([rgb_features, depth_features], dim=-1)
            
    def _process_batch(self, features_list: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process a batch of images.
        
        Args:
            features_list: List of B tuples, each containing (rgb_features, depth_features)
                          where each tensor has shape (Nr, F)
            
        Returns:
            Tuple of:
            - Padded features of shape (B, N_max, F)
            - Attention mask of shape (B, N_max)
        """
        # Handle empty batch
        if not features_list:
            return torch.empty(0, 0, self.infeatures), torch.empty(0, 0, dtype=torch.bool)
            
        # Process each image
        fused_features = []
        for rgb_features, depth_features in features_list:
            fused = self._fuse_features(rgb_features, depth_features)
            fused_features.append(fused)
            
        # Pad sequences
        features_padded = pad_sequence(fused_features, batch_first=True)  # (B, N_max, F)
        
        # Create attention mask
        attention_mask = torch.tensor([
            [True]*f.shape[0] + [False]*(features_padded.shape[1] - f.shape[0])
            for f in fused_features
        ])  # (B, N_max)
        
        return features_padded, attention_mask
        
    def forward(
        self,
        x: Union[Tuple[torch.Tensor, torch.Tensor], List[Tuple[torch.Tensor, torch.Tensor]]]
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Either a tuple of (rgb_features, depth_features) each of shape (Nr, F),
               or a list of B such tuples
            
        Returns:
            Either a tensor of shape (Nr,) for single image, or a tensor of shape (B, N_max) for batch
        """
        if isinstance(x, list):
            # Process batch
            features, attention_mask = self._process_batch(x)
            
            # Handle empty batch
            if features.shape[0] == 0:
                return torch.empty(0)
                
            # Apply MLP to each region
            B, N_max, F = features.shape
            features = features.view(-1, F)  # (B*N_max, F)
            logits = self.mlp(features)  # (B*N_max, 1)
            logits = logits.view(B, N_max)  # (B, N_max)
            
            # Mask padding positions
            logits = logits.masked_fill(~attention_mask, float('-inf'))
            
        else:
            # Process single image
            rgb_features, depth_features = x
            features = self._fuse_features(rgb_features, depth_features)
            logits = self.mlp(features)  # (Nr, 1)
            logits = logits.squeeze(-1)  # (Nr,)
            
        return logits

class MultipleChoiceHeadLoss(nn.Module):
    def __init__(self, reduction: str = 'mean'):
        """
        Loss function for MultipleChoiceHead using CrossEntropy.
        
        Args:
            reduction: How to reduce the loss ('none', 'mean', or 'sum')
        """
        super().__init__()
        self.reduction = reduction
        
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute loss between predictions and targets.
        
        Args:
            logits: Either a tensor of shape (Nr,) for single image,
                   or a tensor of shape (B, N_max) for batch
            targets: Either a scalar tensor for single image,
                    or a tensor of shape (B,) for batch
            
        Returns:
            Loss value
        """
        # Ensure logits and targets have correct shapes
        if logits.dim() == 1:  # single image
            if targets.dim() != 0:
                raise RuntimeError(
                    f"For single image, targets should be scalar, got shape {targets.shape}"
                )
            logits = logits.unsqueeze(0)  # (1, Nr)
            targets = targets.unsqueeze(0)  # (1,)
        else:  # batch
            if logits.shape[0] != targets.shape[0]:
                raise RuntimeError(
                    f"Batch size mismatch: logits {logits.shape[0]} vs targets {targets.shape[0]}"
                )
            
        return F.cross_entropy(logits, targets, reduction=self.reduction)

class CountingHead(MultipleChoiceHead):
    def __init__(
        self,
        infeatures: int,
        hidden_dims: List[int] = [512, 256],
        dropout: float = 0.1,
        fusion_type: str = 'mean'  # 'mean' or 'concat'
    ):
        """
        Head for counting regions in an image.
        Inherits from MultipleChoiceHead since counting is essentially
        selecting the correct number from a set of possible numbers.
        
        Args:
            infeatures: Input feature dimension
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout rate
            fusion_type: How to fuse RGB and depth features ('mean' or 'concat')
        """
        super().__init__(
            infeatures=infeatures,
            hidden_dims=hidden_dims,
            dropout=dropout,
            fusion_type=fusion_type
        )

class CountingHeadLoss(MultipleChoiceHeadLoss):
    def __init__(self, reduction: str = 'mean'):
        """
        Loss function for CountingHead.
        Inherits from MultipleChoiceHeadLoss since counting is essentially
        selecting the correct number from a set of possible numbers.
        
        Args:
            reduction: How to reduce the loss ('none', 'mean', or 'sum')
        """
        super().__init__(reduction=reduction)
