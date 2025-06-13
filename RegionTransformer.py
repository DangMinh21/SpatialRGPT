import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer

class RegionTransformer(nn.Module):
    def __init__(self, d_model=1152, nhead=8, num_self_layers=6, num_cross_layers=4, max_regions=15, max_tokens_img=800):
        super().__init__()
        self.d_model = d_model
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # Positional embeddings
        self.pos_embed_region = nn.Parameter(torch.randn(1, 2 * max_regions + 1, d_model))
        self.pos_embed_image = nn.Parameter(torch.randn(1, max_tokens_img, d_model))

        # Transformer encoders
        self.region_encoder = TransformerEncoder(
            TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True),
            num_layers=num_self_layers
        )
        self.cross_decoder = TransformerDecoder(
            TransformerDecoderLayer(d_model=d_model, nhead=nhead, batch_first=True),
            num_layers=num_cross_layers
        )

    def forward(self, Freg_rgb, Freg_depth, Fimg_rgb):
        B, Nr, F = Freg_rgb.shape
        _, Nt, _ = Fimg_rgb.shape

        # Concatenate region features + CLS token
        Freg = torch.cat([Freg_rgb, Freg_depth], dim=1)       # (B, 2*Nr, F)
        cls = self.cls_token.expand(B, 1, F)                  # (B, 1, F)
        Freg = torch.cat([cls, Freg], dim=1)                  # (B, 2*Nr+1, F)
        Freg += self.pos_embed_region[:, :Freg.size(1), :]    # Add pos emb; Freg.size(1) maybe < max-ntokens

        # Region encoder
        Freg_encoded = self.region_encoder(Freg)              # (B, 2*Nr+1, F)

        # Add position embedding for image features
        Fimg_rgb_pos = Fimg_rgb + self.pos_embed_image[:, :Nt, :]  # (B, Nt, F)

        # Cross attention
        X = self.cross_decoder(Freg_encoded, Fimg_rgb_pos)    # (B, 2*Nr+1, F)

        CLS = X[:, 0, :]                                      # (B, F)
        region_tokens = X[:, 1:, :].chunk(2, dim=1)           # [(B, Nr, F), (B, Nr, F)]
        Freg_rgb_trans, Freg_depth_trans = region_tokens

        return CLS, Freg_rgb_trans, Freg_depth_trans

Freg_rgb = torch.randn(1, 5, 1152)
Freg_depth = torch.randn(1, 5, 1152)
Fimg_rgb = torch.randn(1, 792, 1152)

model = RegionTransformer()
CLS, Freg_rgb_trans, Freg_depth_trans = model(Freg_rgb, Freg_depth, Fimg_rgb)
"""
CLS: cho bài toán phân loại/hồi quy từ đặc trưng gobal của regions
Freg_rgb_trans, Freg_depth_trans: cho qua RGBProjector và DepthProjector
"""
print(CLS.shape)
print(Freg_rgb_trans.shape)
print(Freg_depth_trans.shape)


# class CNNSpaceTransformer(nn.Module):
#     def __init__(self, num_classes=10, in_channels=1, d_model=64, nhead=4, num_layers=4, dim_feedforward=256):
#         super(CNNSpaceTransformer, self).__init__()

#         # === 1. CNN Module ===
#         self.cnn_backbone = nn.Sequential(
#             nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1),  # (B, 16, 28, 28)
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2),                            # (B, 16, 14, 14)
#             nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),           # (B, 32, 14, 14)
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2)                             # (B, 32, 7, 7)
#         )

#         self.token_h = 7
#         self.token_w = 7
#         self.num_tokens = self.token_h * self.token_w  # 49
#         self.channel_dim = 32  # Số kênh đặc trưng sau CNN (B, 4, 1152) -> line

#         # === 2. Tokenization: mỗi vị trí không gian là một token ===
#         self.fc_token = nn.Linear(self.channel_dim, d_model)  # Biến đặc trưng C → d_modeldne (B, 4, 1152)

#         # === 3. CLS token và Positional Embedding ===
#         # self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))  # (1, 1, d_model)
#         self.pos_embed = nn.Parameter(torch.zeros(1, self.num_tokens + 1, d_model))  # (1, 4, 1152)

#         # === 4. Transformer Encoder ===
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True # nhead=8, d_model=1152
#         )
#         self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers) #(B, 4, 1152)

#         # === 5. Classification head ===
#         # self.cls_head = nn.Linear(d_model, num_classes)

#         # === 6. Init weights ===
#         self._init_weights()

#     def _init_weights(self):
#         nn.init.normal_(self.cls_token, std=1e-6)
#         nn.init.normal_(self.pos_embed, std=1e-6)
#         nn.init.xavier_uniform_(self.cls_head.weight)
#         nn.init.constant_(self.cls_head.bias, 0)

#     def forward(self, x):
#         """
#         x: Tensor (B, 1, 28, 28)
#         """
#         B = x.size(0)
#         feat = self.cnn_backbone(x)              # (B, 32, 7, 7)
#         feat = feat.permute(0, 2, 3, 1).reshape(B, -1, self.channel_dim)  # (B, 49, 32)
#         tokens = self.fc_token(feat)             # (B, 49, d_model)

#         cls = self.cls_token.expand(B, -1, -1)   # (B, 1, d_model)
#         x = torch.cat([cls, tokens], dim=1)      # (B, 50, d_model)

#         x = x + self.pos_embed[:, :x.size(1), :] # Positional embedding

#         x = self.transformer(x)                  # (B, 50, d_model)
#         cls_out = x[:, 0, :]                     # (B, d_model)
#         logits = self.cls_head(cls_out)          # (B, num_classes)
#         return logits
    
