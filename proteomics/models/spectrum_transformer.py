"""
Spectrum Transformer Model

Transformer-based architecture for mass spectrum analysis.
Uses self-attention to capture relationships between m/z peaks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class PositionalEncoding(nn.Module):
    """
    Positional encoding for m/z values.

    Adds positional information to peak embeddings.
    """

    def __init__(self, d_model: int, max_len: int = 20000):
        """
        Initialize positional encoding.

        Args:
            d_model: Embedding dimension
            max_len: Maximum sequence length
        """
        super(PositionalEncoding, self).__init__()

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Add positional encoding to input.

        Args:
            x: Input tensor (batch_size, seq_len, d_model)

        Returns:
            Positionally encoded tensor
        """
        x = x + self.pe[:, :x.size(1), :]
        return x


class SpectrumTransformer(nn.Module):
    """
    Transformer model for mass spectrum classification/regression.

    Processes binned spectra using self-attention mechanism.
    """

    def __init__(
        self,
        input_dim: int = 19500,  # Number of m/z bins
        num_classes: int = 10,
        task: str = "classification",
        d_model: int = 256,  # Embedding dimension
        nhead: int = 8,  # Number of attention heads
        num_encoder_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_seq_len: int = 20000,
    ):
        """
        Initialize SpectrumTransformer.

        Args:
            input_dim: Input spectrum dimension
            num_classes: Number of output classes
            task: Task type ('classification' or 'regression')
            d_model: Embedding dimension
            nhead: Number of attention heads
            num_encoder_layers: Number of transformer encoder layers
            dim_feedforward: Dimension of feedforward network
            dropout: Dropout probability
            max_seq_len: Maximum sequence length for positional encoding
        """
        super(SpectrumTransformer, self).__init__()

        self.input_dim = input_dim
        self.num_classes = num_classes
        self.task = task
        self.d_model = d_model

        # Input embedding (project input_dim to d_model)
        self.input_projection = nn.Linear(input_dim, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # Classification/regression head
        self.fc1 = nn.Linear(d_model, 512)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(512, 256)

        if task == "classification":
            self.output = nn.Linear(256, num_classes)
        else:  # regression
            self.output = nn.Linear(256, 1)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input spectrum tensor (batch_size, input_dim)

        Returns:
            Output tensor (batch_size, num_classes) or (batch_size, 1)
        """
        # Project input to d_model dimension
        # (batch_size, input_dim) -> (batch_size, 1, d_model)
        x = self.input_projection(x).unsqueeze(1)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Transformer encoder
        x = self.transformer_encoder(x)

        # Global pooling over sequence dimension
        x = x.mean(dim=1)  # (batch_size, d_model)

        # Classification/regression head
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.output(x)

        return x

    def get_feature_embeddings(self, x):
        """
        Extract feature embeddings before output layer.

        Args:
            x: Input spectrum tensor

        Returns:
            Feature embeddings (batch_size, 256)
        """
        x = self.input_projection(x).unsqueeze(1)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))

        return x


class PeakTransformer(nn.Module):
    """
    Transformer model that operates on peak lists (sparse representation).

    Instead of binned spectra, processes (m/z, intensity) pairs directly.
    More efficient for sparse spectra.
    """

    def __init__(
        self,
        num_classes: int = 10,
        task: str = "classification",
        d_model: int = 128,
        nhead: int = 4,
        num_encoder_layers: int = 3,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        max_peaks: int = 500,
    ):
        """
        Initialize PeakTransformer.

        Args:
            num_classes: Number of output classes
            task: Task type
            d_model: Embedding dimension
            nhead: Number of attention heads
            num_encoder_layers: Number of encoder layers
            dim_feedforward: Feedforward dimension
            dropout: Dropout probability
            max_peaks: Maximum number of peaks to process
        """
        super(PeakTransformer, self).__init__()

        self.task = task
        self.d_model = d_model
        self.max_peaks = max_peaks

        # Peak embedding (m/z and intensity -> d_model)
        self.peak_embedding = nn.Linear(2, d_model)  # 2 features: m/z, intensity

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_peaks)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # Output head
        self.fc1 = nn.Linear(d_model, 256)
        self.dropout = nn.Dropout(dropout)

        if task == "classification":
            self.output = nn.Linear(256, num_classes)
        else:
            self.output = nn.Linear(256, 1)

    def forward(self, peaks, peak_mask=None):
        """
        Forward pass.

        Args:
            peaks: Peak tensor (batch_size, max_peaks, 2) with [m/z, intensity]
            peak_mask: Mask for padding (batch_size, max_peaks), True for valid peaks

        Returns:
            Output tensor
        """
        # Embed peaks
        x = self.peak_embedding(peaks)  # (batch_size, max_peaks, d_model)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Create attention mask if peak_mask is provided
        # Transformer expects False for valid positions, True for masked
        if peak_mask is not None:
            attn_mask = ~peak_mask
        else:
            attn_mask = None

        # Transformer encoder
        x = self.transformer_encoder(x, src_key_padding_mask=attn_mask)

        # Global pooling (mean over valid peaks)
        if peak_mask is not None:
            # Masked mean
            mask_expanded = peak_mask.unsqueeze(-1).float()
            x = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        else:
            x = x.mean(dim=1)

        # Output head
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.output(x)

        return x


# Example usage
if __name__ == "__main__":
    # Test SpectrumTransformer
    print("Testing SpectrumTransformer...")
    model = SpectrumTransformer(
        input_dim=19500,
        num_classes=10,
        task="classification",
        d_model=256,
        nhead=8,
        num_encoder_layers=4
    )

    batch_size = 4
    x = torch.randn(batch_size, 19500)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {num_params:,}")

    # Test PeakTransformer
    print("\nTesting PeakTransformer...")
    peak_model = PeakTransformer(
        num_classes=10,
        task="classification",
        d_model=128,
        max_peaks=500
    )

    # Create dummy peak data
    max_peaks = 500
    peaks = torch.randn(batch_size, max_peaks, 2)  # (m/z, intensity) pairs
    peak_mask = torch.ones(batch_size, max_peaks, dtype=torch.bool)
    peak_mask[:, 400:] = False  # Mask last 100 peaks (padding)

    output = peak_model(peaks, peak_mask)
    print(f"Peak input shape: {peaks.shape}")
    print(f"Output shape: {output.shape}")

    num_params = sum(p.numel() for p in peak_model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {num_params:,}")
