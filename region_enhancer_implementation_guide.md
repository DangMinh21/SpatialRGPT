# Hướng dẫn tích hợp RegionFeatureExtractor vào SpatialRGPT

## Tổng quan
Tài liệu này hướng dẫn cách tích hợp RegionFeatureExtractor - một module mới để nâng cấp khả năng xử lý region features trong SpatialRGPT.

## Các bước thực hiện

### 1. Cập nhật cấu hình model
File: `llava/model/language_model/llava_llama.py`

```python
class LlavaLlamaConfig(LlavaConfig):
    model_type = "llava_llama"
    
    def __init__(
        self,
        # ... các tham số hiện tại ...
        enable_region: bool = True,
        enable_depth: bool = True,
        # Thêm cấu hình mới cho region enhancer
        enable_region_enhancer: bool = True,
        region_enhancer_cfg: dict = {
            'num_heads': 8,
            'num_transformer_layers': 6,
            'num_cross_attn_layers': 1,
            'dropout': 0.1,
            'activation': 'gelu'
        },
        **kwargs
    ):
        super().__init__(
            # ... các tham số hiện tại ...
            enable_region=enable_region,
            enable_depth=enable_depth,
            enable_region_enhancer=enable_region_enhancer,
            region_enhancer_cfg=region_enhancer_cfg,
            **kwargs
        )
```

### 2. Thêm hàm build_region_enhancer
File: `llava/model/region_extractor/builder.py`

```python
def build_region_enhancer(enhancer_cfg, config):
    """
    Args:
        enhancer_cfg: Cấu hình cho RegionFeatureExtractor
        config: Cấu hình chung của model
    """
    return RegionFeatureExtractor(
        dim=config.hidden_size,  # hoặc dim phù hợp
        num_heads=enhancer_cfg.get('num_heads', 8),
        num_transformer_layers=enhancer_cfg.get('num_transformer_layers', 6),
        num_cross_attn_layers=enhancer_cfg.get('num_cross_attn_layers', 1),
        dropout=enhancer_cfg.get('dropout', 0.1),
        activation=enhancer_cfg.get('activation', 'gelu')
    )
```

### 3. Tích hợp vào LlavaMetaForCausalLM
File: `llava/model/llava_arch.py`

```python
class LlavaMetaForCausalLM(ABC):
    def __init__(self, config):
        super().__init__()
        # ... các khởi tạo khác ...
        
        # Khởi tạo RegionFeatureExtractor
        if hasattr(config, "enable_region_enhancer") and config.enable_region_enhancer:
            self.region_enhancer = build_region_enhancer(
                config.region_enhancer_cfg,
                config
            )
        else:
            self.region_enhancer = None

    def prepare_inputs_labels_for_multimodal(self, ...):
        # ... code hiện tại ...
        
        if hasattr(self.config, "enable_region") and self.config.enable_region:
            hres_tower_features, lres_tower_features = self.get_region_extractor().feature_refinement(tower_features)
            
            if hasattr(self.config, "enable_depth") and self.config.enable_depth and depths is not None:
                depth_features = self.get_vision_tower()(depths).to(self.device)
                mask_embeds, depth_embeds = self.get_region_extractor()(hres_tower_features, depth_features, masks)
            else:
                mask_embeds, depth_embeds = self.get_region_extractor()(hres_tower_features, None, masks)
                
            # Thêm xử lý qua RegionFeatureExtractor
            if self.region_enhancer is not None:
                enhanced_features = self.region_enhancer(
                    rgb_features=mask_embeds,
                    depth_features=depth_embeds if depth_embeds is not None else None,
                    image_features=tower_features,
                    mask=masks
                )
                # Cập nhật mask_embeds và depth_embeds với enhanced_features
                mask_embeds, depth_embeds = self.split_enhanced_features(enhanced_features)

    def split_enhanced_features(self, enhanced_features):
        """
        Tách enhanced_features thành mask_embeds và depth_embeds
        """
        if enhanced_features is None:
            return None, None
            
        # Logic tách features dựa trên cấu trúc output của RegionFeatureExtractor
        N = enhanced_features.shape[1] // 2
        mask_embeds = enhanced_features[:, :N]
        depth_embeds = enhanced_features[:, N:]
        
        return mask_embeds, depth_embeds
```

### 4. Cập nhật cấu hình training
File: `llava/train/args.py`

```python
class ModelArguments:
    # ... các tham số hiện tại ...
    enable_region_enhancer: bool = field(
        default=True,
        metadata={"help": "Whether to use region enhancer"}
    )
    region_enhancer_cfg: dict = field(
        default_factory=lambda: {
            'num_heads': 8,
            'num_transformer_layers': 6,
            'num_cross_attn_layers': 1,
            'dropout': 0.1,
            'activation': 'gelu'
        },
        metadata={"help": "Configuration for region enhancer"}
    )
```

### 5. Cập nhật training script
File: `llava/train/train.py`

```python
def train():
    # ... code hiện tại ...
    
    # Khởi tạo model với region-enhancer
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        torch_dtype=torch.float16,
    )
    
    # Đảm bảo region-enhancer được khởi tạo
    if model_args.enable_region_enhancer:
        model.region_enhancer = build_region_enhancer(
            model_args.region_enhancer_cfg,
            config
        )
    
    # Cập nhật optimizer để train các tham số mới
    optimizer = torch.optim.AdamW(
        model.parameters(),  # Sẽ bao gồm cả tham số của region-enhancer
        lr=training_args.learning_rate
    )
```

## Lưu ý quan trọng

1. **Tính tương thích**:
   - Đảm bảo RegionFeatureExtractor có thể xử lý cả batch và từng ảnh
   - Xử lý các trường hợp đặc biệt (masks=None, depths=None)
   - Giữ nguyên interface input/output

2. **Hiệu suất**:
   - Tối ưu hóa việc xử lý batch
   - Giảm thiểu bộ nhớ sử dụng
   - Cân nhắc việc sử dụng mixed precision training

3. **Testing**:
   - Thêm unit tests cho RegionFeatureExtractor
   - Test tính tương thích với các thành phần khác
   - Test các trường hợp đặc biệt

4. **Logging**:
   - Thêm logging để theo dõi quá trình training
   - Log các metric quan trọng
   - Log các trường hợp lỗi

## Các bước tiếp theo

1. Implement RegionFeatureExtractor
2. Thêm các test cases
3. Cập nhật documentation
4. Thêm các metric đánh giá
5. Tối ưu hóa hiệu suất

## Tài liệu tham khảo

1. [SpatialRGPT Documentation](link_to_docs)
2. [RegionFeatureExtractor Design](link_to_design)
3. [Training Guide](link_to_training_guide) 