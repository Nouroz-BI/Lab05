
"""
AIGC 5500 - Lab 05
Attention Against Recurrence

This file contains:
1. Scaled dot-product attention
2. Multi-head attention
3. Basic tests
"""

import math
import torch
import torch.nn as nn

from lab05_starter import reference_attention, ReferenceMHA


def scaled_dot_product_attention(q, k, v, mask=None):
    """Calculate attention scores and use them to combine values."""

    # Compare each query with every key.
    scores = torch.matmul(q, k.transpose(-2, -1))

    # Divide by the square root of the key size.
    scores = scores / math.sqrt(q.shape[-1])

    # A True mask value means that position can be used.
    if mask is not None:
        scores = scores.masked_fill(~mask.bool(), float("-inf"))

    # Turn the scores into probabilities.
    weights = torch.softmax(scores, dim=-1)

    # Use the probabilities to combine the values.
    output = torch.matmul(weights, v)

    return output, weights


class MultiHeadAttention(nn.Module):
    """A simple multi-head attention layer."""

    def __init__(self, d_model, n_heads):
        super().__init__()

        # Each head needs the same number of features.
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # These layers learn how to prepare queries, keys and values.
        self.query_layer = nn.Linear(d_model, d_model)
        self.key_layer = nn.Linear(d_model, d_model)
        self.value_layer = nn.Linear(d_model, d_model)
        self.output_layer = nn.Linear(d_model, d_model)

    def split_heads(self, x):
        """Separate the features into multiple heads."""

        batch_size, seq_len, _ = x.shape

        x = x.reshape(
            batch_size, seq_len, self.n_heads, self.head_dim
        )

        # Put the heads before the sequence dimension.
        return x.transpose(1, 2)

    def forward(self, x, context=None, mask=None):
        # If context is missing, use x for queries, keys and values.
        if context is None:
            context = x

        q = self.query_layer(x)
        k = self.key_layer(context)
        v = self.value_layer(context)

        # Separate the features into heads.
        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)

        # Calculate attention for every head.
        output, weights = scaled_dot_product_attention(q, k, v, mask)

        # Join the heads back together.
        output = output.transpose(1, 2).contiguous()
        batch_size, seq_len, _, _ = output.shape

        output = output.reshape(batch_size, seq_len, self.d_model)

        # Combine the information from all heads.
        return self.output_layer(output)


def run_tests():
    """Check that the attention code produces the expected shapes."""

    torch.manual_seed(0)

    batch_size = 2
    seq_len = 5
    d_model = 32

    x = torch.randn(batch_size, seq_len, d_model)

    # Test multi-head attention.
    model = MultiHeadAttention(d_model=32, n_heads=4)
    output = model(x)

    print("Input shape:", x.shape)
    print("Output shape:", output.shape)

    assert output.shape == x.shape

    # Test attention against PyTorch's reference function.
    q = torch.randn(2, 4, 5, 8)
    k = torch.randn(2, 4, 5, 8)
    v = torch.randn(2, 4, 5, 8)

    our_output, our_weights = scaled_dot_product_attention(q, k, v)
    ref_output, _ = reference_attention(q, k, v)

    print("Our attention matches PyTorch:",
          torch.allclose(our_output, ref_output, atol=1e-5))

    assert torch.allclose(our_output, ref_output, atol=1e-5)

    # Check that masked positions receive zero attention.
    mask = torch.tril(torch.ones(5, 5, dtype=torch.bool))

    _, weights = scaled_dot_product_attention(q, k, v, mask)

    assert torch.all(weights.masked_select(~mask) == 0)

    print("Mask test passed.")
    print("All basic tests passed.")


if __name__ == "__main__":
    run_tests()
