import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
from transformers import PreTrainedModel, PretrainedConfig

class CrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor, context: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Add batch dimension if not present
        if x.dim() == 2:
            x = x.unsqueeze(0)  # [N, C] -> [1, N, C]
        if context.dim() == 2:
            context = context.unsqueeze(0)  # [M, C] -> [1, M, C]
            
        # Ensure inputs are 3D
        assert x.dim() == 3, f"Expected 3D tensor for x, got {x.dim()}D"
        assert context.dim() == 3, f"Expected 3D tensor for context, got {context.dim()}D"
            
        # Handle mask for batched inputs
        if mask is not None:
            if mask.dim() == 1:
                mask = mask.unsqueeze(0)  # [M] -> [1, M]
            # Convert to key_padding_mask format (True for padding positions)
            key_padding_mask = ~mask
        else:
            key_padding_mask = None
            
        # Cross attention
        attn_out, _ = self.attn(query=x, key=context, value=context, key_padding_mask=key_padding_mask)
        x = self.norm1(x + attn_out)
        
        # Feed forward
        x = self.norm2(x + self.ffn(x))
        
        # Remove batch dimension if input was 2D
        if x.shape[0] == 1:
            x = x.squeeze(0)
            
        return x

class RegionTransformer(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        num_layers: int = 6,
        dropout: float = 0.1,
        activation: str = 'gelu'
    ):
        super().__init__()
        self.dim = dim
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=dim * 4,
            dropout=dropout,
            activation=activation,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Layer normalization
        self.norm = nn.LayerNorm(dim)
        
    def forward(
        self,
        rgb_features: torch.Tensor,
        depth_features: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Add batch dimension if not present
        if rgb_features.dim() == 2:
            rgb_features = rgb_features.unsqueeze(0)  # [N, C] -> [1, N, C]
        if depth_features.dim() == 2:
            depth_features = depth_features.unsqueeze(0)  # [N, C] -> [1, N, C]
            
        # Ensure inputs are 3D
        assert rgb_features.dim() == 3, f"Expected 3D tensor for rgb_features, got {rgb_features.dim()}D"
        assert depth_features.dim() == 3, f"Expected 3D tensor for depth_features, got {depth_features.dim()}D"
        
        # Combine RGB and depth features along sequence dimension
        combined = torch.cat([rgb_features, depth_features], dim=1)  # [B, 2*N, C]
        
        # Apply transformer
        if mask is not None:
            # Create padding mask for the combined sequence
            # Shape should be [batch_size, seq_len]
            src_key_padding_mask = torch.zeros((combined.size(0), combined.size(1)), dtype=torch.bool)
        else:
            src_key_padding_mask = None
            
        transformed = self.transformer(combined, src_key_padding_mask=src_key_padding_mask)
        
        # Remove batch dimension if input was 2D
        if transformed.shape[0] == 1:
            transformed = transformed.squeeze(0)
        
        return transformed
    
    def get_region_features(self, transformed: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Add batch dimension if not present
        if transformed.dim() == 2:
            transformed = transformed.unsqueeze(0)  # [2*N, C] -> [1, 2*N, C]
            
        # Split back into RGB and depth features
        N = transformed.shape[1] // 2
        rgb_features = transformed[:, :N]
        depth_features = transformed[:, N:]
        
        # Remove batch dimension if input was 2D
        if rgb_features.shape[0] == 1:
            rgb_features = rgb_features.squeeze(0)
            depth_features = depth_features.squeeze(0)
            
        return rgb_features, depth_features



class RegionFeatureExtractorConfig(PretrainedConfig):
    model_type = "region_feature_extractor"
    # You can add default config values here if needed
    def __init__(self, dim=1152, num_heads=8, num_transformer_layers=6, num_cross_attn_layers=1, dropout= 0.1, activation= 'gelu', **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.num_heads = num_heads
        self.num_transformer_layers = num_transformer_layers
        self.num_cross_attn_layers = num_cross_attn_layers
        self.dropout = dropout
        self.activation = activation

class RegionFeatureExtractor(PreTrainedModel):
    def __init__(self, config: RegionFeatureExtractorConfig):
        self,
        self.dim = config.dim,
        self.num_heads = config.num_heads,
        self.num_transformer_layers = config.num_transformer_layers,
        self.num_cross_attn_layers = config.num_cross_attn_layers,
        self.dropout = config.dropout,
        self.activation = config.activation
        super().__init__()
        
        # Region transformer for processing RGB and depth features
        self.region_transformer = RegionTransformer(
            dim=self.dim,
            num_heads=self.num_heads,
            num_layers=self.num_transformer_layers,
            dropout=self.dropout,
            activation=self.activation
        )
        
        # Cross attention for attending to image features
        self.cross_attention = CrossAttention(
            dim=self.dim,
            num_heads=self.num_heads,
            dropout=self.dropout
        )
        
        # Final layer normalization
        self.norm = nn.LayerNorm(self.dim)
        
    def forward(
        self,
        rgb_features: torch.Tensor,
        depth_features: torch.Tensor,
        image_features: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            rgb_features: RGB features of regions [N, C] or [B, N, C]
            depth_features: Depth features of regions [N, C] or [B, N, C]
            image_features: Global image features [M, C] or [B, M, C]
            mask: Optional mask for image features [M] or [B, M]
            
        Returns:
            Enhanced region features [N, C] or [B, N, C]
        """
        # Check input dimensions
        is_batched = rgb_features.dim() == 3
        
        if is_batched:
            # 3D input case [B, N, C]
            B, N, C = rgb_features.shape
            assert depth_features.shape == (B, N, C), "RGB and depth features must have same shape"
            assert image_features.shape[0] == B, "Image features must have same batch size"
            
            # Process through transformer
            transformed = self.region_transformer(rgb_features, depth_features, mask)
            
            # Get processed features
            rgb_out, depth_out = self.region_transformer.get_region_features(transformed)
            
            # Combine features
            region_features = torch.cat([rgb_out, depth_out], dim=1)  # [B, 2*N, C]
            
            # Apply cross attention
            enhanced = self.cross_attention(region_features, image_features, mask)
            
            # Split back into RGB and depth
            rgb_enhanced = enhanced[:, :N]
            depth_enhanced = enhanced[:, N:]
            
            # Combine enhanced features
            final_features = torch.cat([rgb_enhanced, depth_enhanced], dim=1)  # [B, 2*N, C]
            
        else:
            # 2D input case [N, C]
            N, C = rgb_features.shape
            assert depth_features.shape == (N, C), "RGB and depth features must have same shape"
            
            # Process through transformer
            transformed = self.region_transformer(rgb_features, depth_features, mask)
            
            # Get processed features
            rgb_out, depth_out = self.region_transformer.get_region_features(transformed)
            
            # Combine features
            region_features = torch.cat([rgb_out, depth_out], dim=0)  # [2*N, C]
            
            # Apply cross attention
            enhanced = self.cross_attention(region_features, image_features, mask)
            
            # Split back into RGB and depth
            rgb_enhanced = enhanced[:N]
            depth_enhanced = enhanced[N:]
            
            # Combine enhanced features
            final_features = torch.cat([rgb_enhanced, depth_enhanced], dim=0)  # [2*N, C]
        
        # Apply final normalization
        final_features = self.norm(final_features)
        
        return final_features

