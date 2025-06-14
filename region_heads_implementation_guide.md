# Hướng dẫn tích hợp Region Heads với Dynamic Routing

## Tổng quan
Tài liệu này hướng dẫn cách tích hợp các region heads (Distance, LeftRight, MultipleChoice, Counting) với cơ chế dynamic routing dựa trên task type của mỗi sample. Mỗi sample có một câu hỏi thuộc một trong 4 loại task, và sẽ được định tuyến đến head tương ứng.

## Các bước thực hiện

### 1. Cập nhật Dataset hiện có
File: `llava/data/dataset.py`

```python
class RegionDataset(LlavaDataset):  # Kế thừa từ dataset hiện có
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_types = {
            'distance': 0,
            'left_right': 1,
            'multiple_choice': 2,
            'counting': 3
        }
    
    def __getitem__(self, idx):
        # Lấy data từ dataset gốc
        data = super().__getitem__(idx)
        
        # Bổ sung thêm task_type và label
        data['task_type'] = self.get_task_type(data)
        data['label'] = self.extract_label(data, data['task_type'])
        
        return data
    
    def get_task_type(self, sample):
        """
        Xác định task type dựa trên câu hỏi
        """
        question = sample['question'].lower()
        
        # Distance task: hỏi về khoảng cách
        if any(word in question for word in ['distance', 'far', 'close', 'near']):
            return self.task_types['distance']
            
        # LeftRight task: hỏi về left/right
        elif any(word in question for word in ['left', 'right']):
            return self.task_types['left_right']
            
        # Counting task: hỏi về số lượng
        elif any(phrase in question for phrase in ['how many', 'count', 'number of']):
            return self.task_types['counting']
            
        # MultipleChoice task: hỏi về lựa chọn region
        else:
            return self.task_types['multiple_choice']
    
    def extract_label(self, sample, task_type):
        """
        Trích xuất label tương ứng với task type
        """
        if task_type == self.task_types['distance']:
            # Label là số thực (khoảng cách)
            return float(sample['distance_label'])
            
        elif task_type == self.task_types['left_right']:
            # Label là 0 (left) hoặc 1 (right)
            return 0 if sample['direction'] == 'left' else 1
            
        elif task_type == self.task_types['multiple_choice']:
            # Label là chỉ số của region cần chọn
            return int(sample['selected_region_idx'])
            
        elif task_type == self.task_types['counting']:
            # Label là số lượng cần đếm
            return int(sample['count'])
            
        else:
            raise ValueError(f"Unknown task type: {task_type}")

# Cập nhật DataCollator hiện có
class RegionDataCollator(LlavaDataCollator):
    def __call__(self, batch):
        # Lấy batch từ collator gốc
        batch = super().__call__(batch)
        
        # Bổ sung thêm task_types và labels
        batch['task_types'] = torch.tensor([item['task_type'] for item in batch])
        batch['labels'] = torch.tensor([item['label'] for item in batch])
        
        return batch
```

### 2. Tích hợp Region Heads
File: `llava/model/region_heads.py`

```python
class RegionHeads(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.heads = nn.ModuleDict()
        
        # Khởi tạo các heads
        if config.region_heads_cfg['distance']['enabled']:
            self.heads['distance'] = DistanceHead(
                in_dim=config.hidden_size,
                dropout=config.region_heads_cfg['distance']['dropout']
            )
            self.distance_loss = nn.MSELoss()  # Loss cho regression
            
        if config.region_heads_cfg['left_right']['enabled']:
            self.heads['left_right'] = LeftRightHead(
                in_dim=config.hidden_size,
                dropout=config.region_heads_cfg['left_right']['dropout']
            )
            self.left_right_loss = nn.CrossEntropyLoss()  # Loss cho classification
            
        if config.region_heads_cfg['multiple_choice']['enabled']:
            self.heads['multiple_choice'] = MultipleChoiceHead(
                in_dim=config.hidden_size,
                dropout=config.region_heads_cfg['multiple_choice']['dropout']
            )
            self.multiple_choice_loss = nn.CrossEntropyLoss()  # Loss cho classification
            
        if config.region_heads_cfg['counting']['enabled']:
            self.heads['counting'] = CountingHead(
                in_dim=config.hidden_size,
                dropout=config.region_heads_cfg['counting']['dropout']
            )
            self.counting_loss = nn.CrossEntropyLoss()  # Loss cho classification
    
    def forward(self, features, task_types, labels=None):
        """
        Args:
            features: Region features [B, N, D]
            task_types: Task type indices [B]
            labels: Ground truth labels [B]
        """
        outputs = {}
        losses = {}
        
        # Process each sample with its corresponding head
        for i, task_type in enumerate(task_types):
            task_name = self.get_task_name(task_type)
            if task_name in self.heads:
                # Forward pass qua head tương ứng
                head_output = self.heads[task_name](features[i:i+1])
                outputs[task_name] = head_output
                
                # Tính loss nếu có labels
                if labels is not None:
                    loss = getattr(self, f'{task_name}_loss')(
                        head_output, 
                        labels[i:i+1]
                    )
                    losses[task_name] = loss
        
        return outputs, losses
```

### 3. Cập nhật Training Arguments
File: `llava/train/args.py`

```python
@dataclass
class TrainingArguments(transformers.TrainingArguments):
    # ... existing arguments ...
    
    # Region heads arguments
    enable_region_heads: bool = field(
        default=False,
        metadata={"help": "Whether to enable region heads"}
    )
    tune_region_heads: bool = field(
        default=False,
        metadata={"help": "Whether to tune region heads"}
    )
```

### 4. Cập nhật Model Configuration
File: `llava/model/language_model/llava_llama.py`

```python
class LlavaLlamaConfig(PretrainedConfig):
    def __init__(
        self,
        # ... existing arguments ...
        enable_region_heads: bool = False,
        region_heads_cfg: dict = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        # ... existing initialization ...
        
        self.enable_region_heads = enable_region_heads
        self.region_heads_cfg = region_heads_cfg or {
            'distance': {'enabled': True, 'dropout': 0.1},
            'left_right': {'enabled': True, 'dropout': 0.1},
            'multiple_choice': {'enabled': True, 'dropout': 0.1},
            'counting': {'enabled': True, 'dropout': 0.1}
        }
```

### 5. Cập nhật Model Forward Pass
File: `llava/model/llava_arch.py`

```python
class LlavaMetaForCausalLM(PreTrainedModel):
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        images: Optional[torch.FloatTensor] = None,
        task_types: Optional[torch.LongTensor] = None,
        **kwargs
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        # ... existing forward pass code ...
        
        # Add region heads processing if enabled
        if self.config.enable_region_heads and task_types is not None:
            region_outputs, region_losses = self.region_heads(
                features=hidden_states,
                task_types=task_types,
                labels=labels
            )
            
            # Add region losses to total loss
            if labels is not None:
                for loss_name, loss_value in region_losses.items():
                    loss = loss + loss_value
        
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
```

## Lưu ý quan trọng

1. **Dataset và Dataloader**:
   - Kế thừa từ dataset và collator hiện có
   - Chỉ bổ sung thêm task_type và label
   - Giữ nguyên các chức năng cơ bản của dataset gốc

2. **Loss Functions**:
   - Distance: MSELoss cho regression
   - LeftRight: CrossEntropyLoss cho binary classification
   - MultipleChoice: CrossEntropyLoss cho multi-class classification
   - Counting: CrossEntropyLoss cho classification

3. **Evaluation**:
   - Đánh giá riêng cho từng task type
   - Metrics phù hợp cho từng loại task:
     - Distance: MSE, MAE
     - LeftRight: Accuracy
     - MultipleChoice: Accuracy
     - Counting: Accuracy

4. **Data Pipeline**:
   - Cần cân bằng số lượng samples cho mỗi task type
   - Xử lý các trường hợp đặc biệt
   - Logging chi tiết cho từng task type

## Các bước tiếp theo

1. Implement các region heads
2. Thêm logic xác định task type
3. Thêm các test cases
4. Cập nhật documentation
5. Thêm các metric đánh giá
6. Tối ưu hóa hiệu suất
7. Thêm visualization cho kết quả của từng head

## Tài liệu tham khảo

1. [SpatialRGPT Documentation](link_to_docs)
2. [Region Heads Design](link_to_heads_design)
3. [Dynamic Routing Design](link_to_routing_design)
4. [Training Guide](link_to_training_guide) 