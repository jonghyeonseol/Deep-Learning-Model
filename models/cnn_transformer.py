"""
Hybrid CNN-Transformer architecture combining convolutional and attention mechanisms.

This implementation demonstrates:
- CNN backbone for local feature extraction
- Multi-head self-attention for global context
- Vision Transformer (ViT) style patches
- Positional encodings
- Layer normalization and feed-forward networks
"""

import torch
import torch.nn as nn
import math
from .activations import get_activation


class PatchEmbedding(nn.Module):
    """
    Convert image into patches and embed them.

    Similar to ViT (Vision Transformer) approach.

    Args:
        img_size: Input image size (assumed square)
        patch_size: Size of each patch
        in_channels: Number of input channels
        embed_dim: Embedding dimension
    """

    def __init__(self, img_size=32, patch_size=4, in_channels=3, embed_dim=256):
        super(PatchEmbedding, self).__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        # Use convolution to extract patches and embed them
        self.proj = nn.Conv2d(in_channels, embed_dim,
                             kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: [B, C, H, W]
        x = self.proj(x)  # [B, embed_dim, H/patch_size, W/patch_size]
        x = x.flatten(2)  # [B, embed_dim, num_patches]
        x = x.transpose(1, 2)  # [B, num_patches, embed_dim]
        return x


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention mechanism.

    This allows the model to attend to different representation subspaces
    at different positions.

    Architecture:
        Input -> Q, K, V projections -> Split into heads -> Scaled Dot-Product Attention
              -> Concat heads -> Output projection

    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        dropout: Dropout rate
    """

    def __init__(self, embed_dim, num_heads=8, dropout=0.1):
        super(MultiHeadSelfAttention, self).__init__()

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Query, Key, Value projections
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)

        # Output projection
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, num_tokens, embed_dim = x.shape

        # Generate Q, K, V
        qkv = self.qkv(x)  # [B, N, 3*embed_dim]
        qkv = qkv.reshape(batch_size, num_tokens, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, num_heads, N, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, num_heads, N, N]
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        x = (attn @ v)  # [B, num_heads, N, head_dim]
        x = x.transpose(1, 2)  # [B, N, num_heads, head_dim]
        x = x.reshape(batch_size, num_tokens, embed_dim)

        # Output projection
        x = self.proj(x)
        x = self.dropout(x)

        return x


class TransformerEncoderBlock(nn.Module):
    """
    Transformer Encoder Block.

    Architecture:
        Input -> LayerNorm -> Multi-Head Attention -> (+) -> LayerNorm -> FFN -> (+) -> Output
        |                                              ^                            ^
        |______________________________________________|____________________________|
                    (residual connection)                    (residual connection)

    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        mlp_ratio: Ratio of MLP hidden dim to embedding dim
        dropout: Dropout rate
        activation: Activation function for MLP
    """

    def __init__(self, embed_dim, num_heads=8, mlp_ratio=4.0, dropout=0.1, activation='gelu'):
        super(TransformerEncoderBlock, self).__init__()

        # Layer normalization
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        # Multi-head self-attention
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)

        # Feed-forward network (MLP)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            get_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # Attention with residual connection
        x = x + self.attn(self.norm1(x))

        # MLP with residual connection
        x = x + self.mlp(self.norm2(x))

        return x


class CNNBackbone(nn.Module):
    """
    CNN backbone for local feature extraction.

    Extracts local features before feeding to transformer.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        activation: Activation function
    """

    def __init__(self, in_channels=3, out_channels=256, activation='relu'):
        super(CNNBackbone, self).__init__()

        self.features = nn.Sequential(
            # Conv block 1
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            get_activation(activation),

            # Conv block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            get_activation(activation),
            nn.MaxPool2d(2, 2),  # 32 -> 16

            # Conv block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            get_activation(activation),

            # Conv block 4
            nn.Conv2d(256, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            get_activation(activation),
            nn.MaxPool2d(2, 2),  # 16 -> 8
        )

    def forward(self, x):
        return self.features(x)


class CNNTransformer(nn.Module):
    """
    Hybrid CNN-Transformer architecture.

    Combines:
    1. CNN backbone for local feature extraction
    2. Transformer encoder for global context modeling
    3. Classification head

    This hybrid approach:
    - Uses CNNs to extract local patterns and reduce spatial dimensions
    - Uses transformers to model long-range dependencies
    - Gets benefits of both inductive bias (CNN) and flexibility (Transformer)

    Args:
        img_size: Input image size
        num_classes: Number of output classes
        embed_dim: Transformer embedding dimension
        depth: Number of transformer blocks
        num_heads: Number of attention heads
        mlp_ratio: MLP expansion ratio
        dropout: Dropout rate
        activation: Activation function
        use_cnn_backbone: Whether to use CNN backbone (True) or direct patch embedding (False)
    """

    def __init__(self, img_size=32, num_classes=10, embed_dim=256, depth=6, num_heads=8,
                 mlp_ratio=4.0, dropout=0.1, activation='gelu', use_cnn_backbone=True):
        super(CNNTransformer, self).__init__()

        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.activation_name = activation
        self.use_cnn_backbone = use_cnn_backbone

        if use_cnn_backbone:
            # CNN backbone for feature extraction
            self.cnn_backbone = CNNBackbone(in_channels=3, out_channels=embed_dim,
                                           activation=activation)

            # After CNN backbone: 32x32 -> 8x8
            reduced_size = img_size // 4
            self.patch_embed = PatchEmbedding(img_size=reduced_size, patch_size=2,
                                             in_channels=embed_dim, embed_dim=embed_dim)
            num_patches = (reduced_size // 2) ** 2  # 4x4 = 16 patches
        else:
            # Direct patch embedding (ViT-style)
            self.cnn_backbone = None
            self.patch_embed = PatchEmbedding(img_size=img_size, patch_size=4,
                                             in_channels=3, embed_dim=embed_dim)
            num_patches = (img_size // 4) ** 2  # 8x8 = 64 patches

        # Class token (learnable)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Positional embeddings (learnable)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        self.dropout = nn.Dropout(dropout)

        # Transformer encoder blocks
        self.transformer = nn.Sequential(*[
            TransformerEncoderBlock(embed_dim, num_heads, mlp_ratio, dropout, activation)
            for _ in range(depth)
        ])

        # Classification head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights."""
        # Initialize patch embedding
        if hasattr(self.patch_embed, 'proj'):
            nn.init.kaiming_normal_(self.patch_embed.proj.weight, mode='fan_out', nonlinearity='relu')

        # Initialize positional embeddings
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=0.02)

        # Initialize linear layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        batch_size = x.shape[0]

        # CNN backbone (if used)
        if self.use_cnn_backbone:
            x = self.cnn_backbone(x)

        # Patch embedding
        x = self.patch_embed(x)  # [B, num_patches, embed_dim]

        # Prepend class token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # [B, 1, embed_dim]
        x = torch.cat([cls_tokens, x], dim=1)  # [B, num_patches+1, embed_dim]

        # Add positional embeddings
        x = x + self.pos_embed
        x = self.dropout(x)

        # Transformer encoder
        x = self.transformer(x)

        # Classification (use class token)
        x = self.norm(x[:, 0])  # Take the class token
        x = self.head(x)

        return x

    def get_num_parameters(self):
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def summary(self):
        """Print a summary of the network architecture."""
        print(f"CNN-Transformer Hybrid Summary:")
        print(f"Number of classes: {self.num_classes}")
        print(f"Embedding dimension: {self.embed_dim}")
        print(f"Activation: {self.activation_name}")
        print(f"Use CNN backbone: {self.use_cnn_backbone}")
        print(f"Total parameters: {self.get_num_parameters():,}")


# Predefined variants
def CNNTransformer_Small(num_classes=10, activation='gelu'):
    """Small CNN-Transformer (faster training)"""
    return CNNTransformer(num_classes=num_classes, embed_dim=128, depth=4,
                         num_heads=4, activation=activation, use_cnn_backbone=True)


def CNNTransformer_Base(num_classes=10, activation='gelu'):
    """Base CNN-Transformer"""
    return CNNTransformer(num_classes=num_classes, embed_dim=256, depth=6,
                         num_heads=8, activation=activation, use_cnn_backbone=True)


def VisionTransformer_Tiny(num_classes=10, activation='gelu'):
    """Pure Vision Transformer (no CNN backbone)"""
    return CNNTransformer(num_classes=num_classes, embed_dim=192, depth=12,
                         num_heads=3, activation=activation, use_cnn_backbone=False)
