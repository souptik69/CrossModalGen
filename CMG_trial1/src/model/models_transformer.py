import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from typing import Optional


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def Linear(in_features, out_features, bias=True):
    """Initialize a linear layer with Xavier uniform initialization."""
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.)
    return m


def LayerNorm(embedding_dim):
    """Create a LayerNorm module."""
    m = nn.LayerNorm(embedding_dim)
    return m


def fill_with_neg_inf(t):
    """FP16-compatible function that fills a tensor with -inf."""
    return t.float().fill_(float('-inf')).type_as(t)


def buffered_future_mask(tensor, tensor2=None):
    """
    Create a causal mask for attention (prevents attending to future positions).
    
    Args:
        tensor: Query tensor
        tensor2: Key tensor (optional, for cross-attention)
    
    Returns:
        Causal mask with -inf above diagonal
    """
    dim1 = dim2 = tensor.size(0)
    if tensor2 is not None:
        dim2 = tensor2.size(0)
    future_mask = torch.triu(fill_with_neg_inf(torch.ones(dim1, dim2)), 1 + abs(dim2 - dim1))
    if tensor.is_cuda:
        future_mask = future_mask.cuda()
    return future_mask[:dim1, :dim2]


# ============================================================================
# MULTIHEAD ATTENTION WITH PADDING MASK SUPPORT
# ============================================================================

class MultiheadAttention(nn.Module):
    """
    Multi-headed attention with explicit key_padding_mask support.
    
    Uses PyTorch-style additive masking: boolean mask is converted to float mask
    where padding positions = -inf, which becomes 0 after softmax.
    
    Args:
        embed_dim: Total dimension of the model
        num_heads: Number of parallel attention heads
        attn_dropout: Dropout probability on attention weights
        bias: Whether to use bias in projections
        add_bias_kv: Whether to add bias to key and value
        add_zero_attn: Whether to add zero attention
    """

    def __init__(self, embed_dim, num_heads, attn_dropout=0.,
                 bias=True, add_bias_kv=False, add_zero_attn=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.attn_dropout = attn_dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.in_proj_weight = Parameter(torch.Tensor(3 * embed_dim, embed_dim))
        self.register_parameter('in_proj_bias', None)
        if bias:
            self.in_proj_bias = Parameter(torch.Tensor(3 * embed_dim))
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        if add_bias_kv:
            self.bias_k = Parameter(torch.Tensor(1, 1, embed_dim))
            self.bias_v = Parameter(torch.Tensor(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

    def forward(self, query, key, value, key_padding_mask=None, attn_mask=None):
        """
        Forward pass with padding mask support.
        
        Args:
            query: [tgt_len, batch, embed_dim]
            key: [src_len, batch, embed_dim]
            value: [src_len, batch, embed_dim]
            key_padding_mask: [batch, src_len] - Boolean mask where True = padding
            attn_mask: [tgt_len, src_len] - Additive attention mask (e.g., causal mask)
        
        Returns:
            attn: [tgt_len, batch, embed_dim]
            attn_weights: [batch, tgt_len, src_len]
        """
        qkv_same = query.data_ptr() == key.data_ptr() == value.data_ptr()
        kv_same = key.data_ptr() == value.data_ptr()

        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        assert key.size() == value.size()

        # Compute Q, K, V projections
        if qkv_same:
            q, k, v = self.in_proj_qkv(query)
        elif kv_same:
            q = self.in_proj_q(query)
            if key is None:
                assert value is None
                k = v = None
            else:
                k, v = self.in_proj_kv(key)
        else:
            q = self.in_proj_q(query)
            k = self.in_proj_k(key)
            v = self.in_proj_v(value)
        
        q = q * self.scaling

        # Add bias to key and value if specified
        if self.bias_k is not None:
            assert self.bias_v is not None
            k = torch.cat([k, self.bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [key_padding_mask, key_padding_mask.new_zeros(bsz, 1)], dim=1
                )

        # Reshape for multi-head attention: [batch*num_heads, seq_len, head_dim]
        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if k is not None:
            k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if v is not None:
            v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        src_len = k.size(1)

        # Add zero attention if specified
        if self.add_zero_attn:
            src_len += 1
            k = torch.cat([k, k.new_zeros((k.size(0), 1) + k.size()[2:])], dim=1)
            v = torch.cat([v, v.new_zeros((v.size(0), 1) + v.size()[2:])], dim=1)
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [key_padding_mask, key_padding_mask.new_zeros(bsz, 1)], dim=1
                )

        # Compute attention weights: [batch*num_heads, tgt_len, src_len]
        attn_weights = torch.bmm(q, k.transpose(1, 2))
        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        # Apply causal attention mask if provided (additive)
        if attn_mask is not None:
            attn_weights += attn_mask.unsqueeze(0)

        # Apply key padding mask - CRITICAL PART
        if key_padding_mask is not None:
            float_mask = torch.zeros_like(
                key_padding_mask, dtype=attn_weights.dtype
            ).masked_fill_(key_padding_mask, float('-inf'))
            
            # Expand to [batch*num_heads, 1, src_len]
            float_mask = float_mask.unsqueeze(1)  # [batch, 1, src_len]
            float_mask = float_mask.repeat_interleave(self.num_heads, dim=0)  # [batch*num_heads, 1, src_len]
            
            attn_weights = attn_weights + float_mask

        # Apply softmax - padding positions with -inf become 0
        attn_weights = F.softmax(attn_weights.float(), dim=-1).type_as(attn_weights)
        attn_weights = F.dropout(attn_weights, p=self.attn_dropout, training=self.training)

        # Apply attention to values
        attn = torch.bmm(attn_weights, v)
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]

        # Reshape back to [tgt_len, batch, embed_dim]
        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)

        # Average attention weights over heads for output
        attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
        attn_weights = attn_weights.sum(dim=1) / self.num_heads
        
        return attn, attn_weights

    def in_proj_qkv(self, query):
        return self._in_proj(query).chunk(3, dim=-1)

    def in_proj_kv(self, key):
        return self._in_proj(key, start=self.embed_dim).chunk(2, dim=-1)

    def in_proj_q(self, query, **kwargs):
        return self._in_proj(query, end=self.embed_dim, **kwargs)

    def in_proj_k(self, key):
        return self._in_proj(key, start=self.embed_dim, end=2 * self.embed_dim)

    def in_proj_v(self, value):
        return self._in_proj(value, start=2 * self.embed_dim)

    def _in_proj(self, input, start=0, end=None, **kwargs):
        weight = kwargs.get('weight', self.in_proj_weight)
        bias = kwargs.get('bias', self.in_proj_bias)
        weight = weight[start:end, :]
        if bias is not None:
            bias = bias[start:end]
        return F.linear(input, weight, bias)


# ============================================================================
# SINUSOIDAL POSITIONAL EMBEDDINGS (NO IMPLICIT PADDING DETECTION)
# ============================================================================

class SinusoidalPositionalEmbedding(nn.Module):
    """
    Sinusoidal positional embeddings without implicit padding detection.
    
    Generates positional embeddings using sin/cos functions based purely on
    position indices. Does not attempt to detect padding from input features.
    Padding should be handled via explicit attention masks.
    
    Args:
        embedding_dim: Dimension of positional embeddings
        init_size: Initial size of embedding table (will expand as needed)
    """

    def __init__(self, embedding_dim, init_size=1024):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.weights = dict()  # device --> embedding table
        self.register_buffer('_float_tensor', torch.FloatTensor(1))

    @staticmethod
    def get_embedding(num_embeddings, embedding_dim):
        """
        Build sinusoidal embeddings using pure mathematics (no data dependency).
        
        PE(pos, 2i) = sin(pos / 10000^(2i/d))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
        
        Args:
            num_embeddings: Number of positions
            embedding_dim: Dimension of embeddings
        
        Returns:
            Tensor of shape [num_embeddings, embedding_dim]
        """
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).reshape(num_embeddings, -1)
        if embedding_dim % 2 == 1:
            # Zero pad for odd dimensions
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        return emb

    def forward(self, input_shape, device):
        """
        Get positional embeddings for all positions in the sequence.
        
        Args:
            input_shape: Tuple of (batch_size, seq_len)
            device: Torch device
        
        Returns:
            Positional embeddings of shape [seq_len, batch, embed_dim]
        """
        bsz, seq_len = input_shape
        max_pos = seq_len
        
        # Create or expand embedding table if needed
        if device not in self.weights or max_pos > self.weights[device].size(0):
            self.weights[device] = SinusoidalPositionalEmbedding.get_embedding(
                max_pos,
                self.embedding_dim,
            )
        self.weights[device] = self.weights[device].type_as(self._float_tensor)
        
        # Create position indices [0, 1, 2, ..., seq_len-1] for all positions
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(bsz, -1)
        
        # Get embeddings and reshape to [seq_len, batch, embed_dim]
        pos_emb = self.weights[device].index_select(0, positions.reshape(-1))
        pos_emb = pos_emb.reshape(bsz, seq_len, -1).transpose(0, 1)
        
        return pos_emb.detach()

    def max_positions(self):
        """Maximum number of supported positions."""
        return int(1e5)


# ============================================================================
# TRANSFORMER ENCODER LAYER
# ============================================================================

class TransformerEncoderLayer(nn.Module):
    """
    Transformer encoder layer with explicit padding mask support.
    
    Architecture:
        1. Multi-head self-attention with padding mask
        2. Add & Norm
        3. Feedforward network
        4. Add & Norm
        5. Zero out padding positions in output
    
    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        attn_dropout: Dropout for attention weights
        relu_dropout: Dropout for feedforward ReLU
        res_dropout: Dropout for residual connections
        attn_mask: Whether to use causal attention mask
    """

    def __init__(self, embed_dim, num_heads=4, attn_dropout=0.1, relu_dropout=0.1, 
                 res_dropout=0.1, attn_mask=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.self_attn = MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            attn_dropout=attn_dropout
        )
        self.attn_mask = attn_mask

        self.relu_dropout = relu_dropout
        self.res_dropout = res_dropout
        self.normalize_before = True

        self.fc1 = Linear(self.embed_dim, 4 * self.embed_dim)
        self.fc2 = Linear(4 * self.embed_dim, self.embed_dim)
        self.layer_norms = nn.ModuleList([LayerNorm(self.embed_dim) for _ in range(2)])

    def forward(self, x, x_k=None, x_v=None, key_padding_mask=None):
        """
        Forward pass with attention and feedforward layers.
        
        Args:
            x: [seq_len, batch, embed_dim] - Input sequence
            x_k: [seq_len, batch, embed_dim] - Keys for cross-attention (optional)
            x_v: [seq_len, batch, embed_dim] - Values for cross-attention (optional)
            key_padding_mask: [batch, seq_len] - Boolean mask where True = padding
        
        Returns:
            [seq_len, batch, embed_dim] - Output with padding positions zeroed
        """
        # Self-attention block
        residual = x
        x = self.maybe_layer_norm(0, x, before=True)
        
        # Create causal mask if specified
        causal_mask = buffered_future_mask(x, x_k) if self.attn_mask else None
        
        # Apply attention (self-attention or cross-attention)
        if x_k is None and x_v is None:
            x, _ = self.self_attn(
                query=x, 
                key=x, 
                value=x, 
                key_padding_mask=key_padding_mask,
                attn_mask=causal_mask
            )
        else:
            x_k = self.maybe_layer_norm(0, x_k, before=True)
            x_v = self.maybe_layer_norm(0, x_v, before=True)
            x, _ = self.self_attn(
                query=x, 
                key=x_k, 
                value=x_v, 
                key_padding_mask=key_padding_mask,
                attn_mask=causal_mask
            )
        
        x = F.dropout(x, p=self.res_dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(0, x, after=True)

        # Feedforward block
        residual = x
        x = self.maybe_layer_norm(1, x, before=True)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.relu_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.res_dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(1, x, after=True)
        
        # CRITICAL: Zero out padding positions in output
        if key_padding_mask is not None:
            # Convert to [seq_len, batch, 1] for broadcasting
            padding_mask_expanded = (~key_padding_mask).transpose(0, 1).unsqueeze(-1)
            x = x * padding_mask_expanded
        
        return x

    def maybe_layer_norm(self, i, x, before=False, after=False):
        """Apply layer normalization conditionally."""
        assert before ^ after
        if after ^ self.normalize_before:
            return self.layer_norms[i](x)
        else:
            return x


# ============================================================================
# TRANSFORMER ENCODER
# ============================================================================

class TransformerEncoder(nn.Module):
    """
    Transformer encoder with sinusoidal positional embeddings and padding support.
    
    Key features:
    - Adds sinusoidal positional embeddings ONLY to valid (non-padding) positions
    - Propagates key_padding_mask through all layers
    - Zeros out padding positions at multiple stages to prevent information leakage
    
    Args:
        embed_dim: Model dimension
        num_heads: Number of attention heads
        layers: Number of encoder layers
        attn_dropout: Attention dropout rate
        relu_dropout: FFN ReLU dropout rate
        res_dropout: Residual connection dropout rate
        embed_dropout: Embedding dropout rate
        attn_mask: Whether to use causal attention masking
    """

    def __init__(self, embed_dim, num_heads, layers, attn_dropout=0.1, relu_dropout=0.1, 
                 res_dropout=0.1, embed_dropout=0.25, attn_mask=False):
        super().__init__()
        self.dropout = embed_dropout
        self.attn_dropout = attn_dropout
        self.embed_dim = embed_dim
        self.embed_scale = math.sqrt(embed_dim)
        
        self.embed_positions = SinusoidalPositionalEmbedding(embed_dim)
        self.attn_mask = attn_mask

        self.layers = nn.ModuleList([])
        for layer in range(layers):
            new_layer = TransformerEncoderLayer(
                embed_dim,
                num_heads=num_heads,
                attn_dropout=attn_dropout,
                relu_dropout=relu_dropout,
                res_dropout=res_dropout,
                attn_mask=attn_mask
            )
            self.layers.append(new_layer)

        self.register_buffer('version', torch.Tensor([2]))
        self.normalize = True
        if self.normalize:
            self.layer_norm = LayerNorm(embed_dim)

    def forward(self, x_in, x_in_k=None, x_in_v=None, key_padding_mask=None):
        """
        Forward pass through the transformer encoder.
        
        Args:
            x_in: [seq_len, batch, embed_dim] - Input embeddings
            x_in_k: [seq_len, batch, embed_dim] - Keys for cross-attention (optional)
            x_in_v: [seq_len, batch, embed_dim] - Values for cross-attention (optional)
            key_padding_mask: [batch, seq_len] - Boolean mask where True = padding
        
        Returns:
            [seq_len, batch, embed_dim] - Encoded sequence with padding zeroed
        """
        seq_len, bsz, _ = x_in.size()
        
        # Scale input embeddings
        x = self.embed_scale * x_in
        
        
        # Add positional embeddings ONLY to valid (non-padding) positions

        # if self.embed_positions is not None:
        pos_emb = self.embed_positions((bsz, seq_len), x_in.device)
        
        if key_padding_mask is not None:
            # Create mask: 1 for valid positions, 0 for padding
            valid_mask = (~key_padding_mask).transpose(0, 1).unsqueeze(-1)
            # CRITICAL: Zero out positional embeddings for padding positions
            pos_emb = pos_emb * valid_mask
        
        x = x + pos_emb
        
        # Apply embedding dropout
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Zero out padding after dropout
        if key_padding_mask is not None:
            padding_mask_expanded = (~key_padding_mask).transpose(0, 1).unsqueeze(-1)
            x = x * padding_mask_expanded

        # Handle cross-attention inputs if provided
        if x_in_k is not None and x_in_v is not None:
            x_k = self.embed_scale * x_in_k
            x_v = self.embed_scale * x_in_v
            
            if self.embed_positions is not None:
                pos_emb_k = self.embed_positions((bsz, x_in_k.size(0)), x_in_k.device)
                pos_emb_v = self.embed_positions((bsz, x_in_v.size(0)), x_in_v.device)
                
                if key_padding_mask is not None:
                    valid_mask = (~key_padding_mask).transpose(0, 1).unsqueeze(-1)
                    pos_emb_k = pos_emb_k * valid_mask
                    pos_emb_v = pos_emb_v * valid_mask
                
                x_k = x_k + pos_emb_k
                x_v = x_v + pos_emb_v
            
            x_k = F.dropout(x_k, p=self.dropout, training=self.training)
            x_v = F.dropout(x_v, p=self.dropout, training=self.training)
            
            if key_padding_mask is not None:
                padding_mask_expanded = (~key_padding_mask).transpose(0, 1).unsqueeze(-1)
                x_k = x_k * padding_mask_expanded
                x_v = x_v * padding_mask_expanded

        # Pass through encoder layers
        for layer in self.layers:
            if x_in_k is not None and x_in_v is not None:
                x = layer(x, x_k, x_v, key_padding_mask=key_padding_mask)
            else:
                x = layer(x, key_padding_mask=key_padding_mask)

        # Final layer normalization
        if self.normalize:
            x = self.layer_norm(x)
        
        # Final masking - ensure padding stays zero
        if key_padding_mask is not None:
            padding_mask_expanded = (~key_padding_mask).transpose(0, 1).unsqueeze(-1)
            x = x * padding_mask_expanded

        return x

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return self.embed_positions.max_positions()


# ============================================================================
# INTERNAL TEMPORAL RELATION MODULE
# ============================================================================


class InternalTemporalRelationModule(nn.Module):
    def __init__(self, input_dim, d_model, num_heads=5, num_layers=5, 
                 dropout=0.1):
        super(InternalTemporalRelationModule, self).__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.encoder = TransformerEncoder(
            embed_dim=d_model,
            num_heads=num_heads,
            layers=num_layers,
            attn_dropout=dropout,
            relu_dropout=dropout,
            res_dropout=dropout,
            embed_dropout=0.25,
            attn_mask=False
        )
        self.proj = nn.Conv1d(input_dim, d_model, kernel_size=1, padding=0, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, feature, attention_mask=None):
        batch_size, seq_len, input_dim = feature.size()
        feature = feature.transpose(1, 2).contiguous()  # [batch, input_dim, seq_len]
        feature = self.proj(feature)  # [batch, d_model, seq_len]
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(1).float()  # [batch, 1, seq_len]
            feature = feature * mask_expanded  # Zero out padding
        feature = feature.permute(2, 0, 1)  # [seq_len, batch, d_model]
        if attention_mask is not None:
            key_padding_mask = ~attention_mask
        else:
            key_padding_mask = None
        feature = self.encoder(feature, key_padding_mask=key_padding_mask)
        feature = feature.transpose(0, 1).contiguous()  # [batch, seq_len, d_model]
        return feature

# ============================================================================
# USAGE EXAMPLES AND TESTS
# ============================================================================

def test_padding_behavior():
    """
    Test that demonstrates correct padding behavior throughout the architecture.
    """
    print("="*80)
    print("TESTING PADDING BEHAVIOR IN TRANSFORMER ENCODER")
    print("="*80)
    
    batch_size = 2
    seq_len = 6
    input_dim = 10
    d_model = 16
    
    # Create module
    module = InternalTemporalRelationModule(
        input_dim=input_dim, 
        d_model=d_model,
        num_heads=2,
        num_layers=2
    )
    module.eval()
    
    # Create sample input
    features = torch.randn(batch_size, seq_len, input_dim)
    
    # Attention mask: True = valid, False = padding
    # Sequence 1: 4 valid positions, 2 padding
    # Sequence 2: 5 valid positions, 1 padding
    attention_mask = torch.tensor([
        [True, True, True, True, False, False],
        [True, True, True, True, True,  False],
    ])
    
    print(f"\nInput shape: {features.shape}")
    print(f"Attention mask:\n{attention_mask}")
    
    # Forward pass
    with torch.no_grad():
        output = module(features, attention_mask=attention_mask)
    
    print(f"\nOutput shape: {output.shape}")
    
    # Verify padding positions are zero
    print("\n" + "-"*80)
    print("VERIFICATION: Checking that padding positions are exactly zero")
    print("-"*80)
    
    for i in range(batch_size):
        valid_length = attention_mask[i].sum().item()
        print(f"\nSequence {i+1}: {valid_length} valid positions, {seq_len - valid_length} padding")
        
        for t in range(seq_len):
            is_valid = attention_mask[i, t].item()
            output_norm = output[i, t].norm().item()
            
            if is_valid:
                print(f"  Position {t} (valid):   ||output|| = {output_norm:.6f} (should be > 0)")
                assert output_norm > 1e-6, f"Valid position {t} has near-zero output!"
            else:
                print(f"  Position {t} (padding): ||output|| = {output_norm:.6f} (should be = 0)")
                assert output_norm < 1e-6, f"Padding position {t} has non-zero output!"
    
    print("\n" + "="*80)
    print("✅ ALL TESTS PASSED!")
    print("="*80)
    print("✓ Positional embeddings applied only to valid positions")
    print("✓ Attention properly masked padding positions")
    print("✓ Output padding positions are exactly zero")
    print("✓ Valid positions have meaningful non-zero outputs")


def test_attention_weights():
    """
    Test that attention weights sum to 1 and don't attend to padding.
    """
    print("\n" + "="*80)
    print("TESTING ATTENTION WEIGHT DISTRIBUTION")
    print("="*80)
    
    # Simple test with custom attention layer
    embed_dim = 8
    num_heads = 2
    seq_len = 5
    batch_size = 1
    
    attn = MultiheadAttention(embed_dim, num_heads, attn_dropout=0.0)
    attn.eval()
    
    # Create input
    x = torch.randn(seq_len, batch_size, embed_dim)
    
    # Mask: last 2 positions are padding
    key_padding_mask = torch.tensor([[False, False, False, True, True]])
    
    print(f"\nSequence length: {seq_len}")
    print(f"Padding mask (True=padding): {key_padding_mask[0]}")
    print(f"Valid positions: 0, 1, 2")
    print(f"Padding positions: 3, 4")
    
    # Forward pass
    with torch.no_grad():
        output, attn_weights = attn(x, x, x, key_padding_mask=key_padding_mask)
    
    print(f"\nAttention weights shape: {attn_weights.shape}")
    print(f"Attention weights:\n{attn_weights[0]}")
    
    # Verify
    print("\n" + "-"*80)
    print("VERIFICATION:")
    print("-"*80)
    
    for query_pos in range(seq_len):
        weights = attn_weights[0, query_pos]
        weight_sum = weights.sum().item()
        padding_weight = weights[3:].sum().item()
        
        print(f"Query position {query_pos}:")
        print(f"  Weights: {weights.numpy()}")
        print(f"  Sum: {weight_sum:.6f} (should be ≈ 1.0)")
        print(f"  Attention to padding: {padding_weight:.6f} (should be ≈ 0.0)")
        
        assert abs(weight_sum - 1.0) < 1e-5, "Weights don't sum to 1!"
        assert padding_weight < 1e-5, "Attending to padding positions!"
    
    print("\n✅ Attention weights correctly normalized and padding ignored!")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("TRANSFORMER ENCODER WITH EXPLICIT PADDING MASK SUPPORT")
    print("="*80)
    
    # Run tests
    test_padding_behavior()
    test_attention_weights()
    
    print("\n" + "="*80)
    print("INTEGRATION EXAMPLE WITH MOSEI-STYLE DATA")
    print("="*80)
    
    # Simulate MOSEI data format
    batch_size = 4
    max_seq_len = 50
    
    # Audio: 74 dims, Video: 35 dims, Text: 300 dims
    audio_features = torch.randn(batch_size, max_seq_len, 74)
    video_features = torch.randn(batch_size, max_seq_len, 35)
    text_features = torch.randn(batch_size, max_seq_len, 300)
    
    # Simulate variable lengths (from your collate function)
    attention_mask = torch.ones(batch_size, max_seq_len, dtype=torch.bool)
    actual_lengths = [30, 45, 25, 50]
    for i, length in enumerate(actual_lengths):
        attention_mask[i, length:] = False
    
    print(f"\nProcessing batch of {batch_size} sequences")
    print(f"Max sequence length: {max_seq_len}")
    print(f"Actual lengths: {actual_lengths}")
    
    # Create modules for each modality
    audio_module = InternalTemporalRelationModule(input_dim=74, d_model=500)
    video_module = InternalTemporalRelationModule(input_dim=35, d_model=500)
    text_module = InternalTemporalRelationModule(input_dim=300, d_model=500)
    
    # Process each modality
    with torch.no_grad():
        audio_encoded = audio_module(audio_features, attention_mask=attention_mask)
        video_encoded = video_module(video_features, attention_mask=attention_mask)
        text_encoded = text_module(text_features, attention_mask=attention_mask)
    
    print(f"\nEncoded shapes:")
    print(f"  Audio: {audio_encoded.shape}")
    print(f"  Video: {video_encoded.shape}")
    print(f"  Text: {text_encoded.shape}")
    
    # Verify padding is zero
    for i, length in enumerate(actual_lengths):
        if length < max_seq_len:
            padding_norm = audio_encoded[i, length:].norm().item()
            print(f"  Sequence {i}: padding positions [{length}:{max_seq_len}] norm = {padding_norm:.2e}")
            assert padding_norm < 1e-5, f"Sequence {i} has non-zero padding!"
    
    print("\n✅ Successfully processed multimodal data with correct padding handling!")
    print("\n" + "="*80)