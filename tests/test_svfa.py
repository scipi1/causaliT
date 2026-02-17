"""
Test suite for Structure-Value Factorized Attention (SVFA) implementation.

Run with: pytest tests/test_svfa.py -v

This suite validates:
1. Embedding layer returns correct tuple format in SVFA mode
2. Encoder/Decoder handle SVFA tuples correctly
3. Forward pass shapes are correct
4. Backward compatibility with standard mode
"""

import pytest
import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from causaliT.core.modules.embedding import ModularEmbedding
from causaliT.core.modules.encoder import EncoderLayer, Encoder
from causaliT.core.modules.decoder import DecoderLayer, Decoder
from causaliT.core.modules.attention import AttentionLayer, ScaledDotAttention
from causaliT.core.modules.extra_layers import Normalization, UniformAttentionMask


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def d_model():
    return 32


@pytest.fixture
def vocab_size():
    return 10


@pytest.fixture
def batch_size():
    return 4


@pytest.fixture
def seq_len():
    return 8


@pytest.fixture
def d_ff():
    return 64


@pytest.fixture
def d_qk():
    return 16


@pytest.fixture
def svfa_embed_config(vocab_size, d_model):
    """SVFA embedding configuration with role assignments"""
    return {
        "setting": {"d_model": d_model},
        "modules": [
            {
                "idx": 0,
                "embed": "nn_embedding",
                "label": "variable",
                "role": "structure",  # For Q, K
                "kwargs": {"num_embeddings": vocab_size, "embedding_dim": d_model}
            },
            {
                "idx": 1,
                "embed": "linear",
                "label": "value",
                "role": "value",  # For V
                "kwargs": {"input_dim": 1, "embedding_dim": d_model}
            }
        ]
    }


@pytest.fixture
def standard_embed_config(vocab_size, d_model):
    """Standard embedding configuration (no roles)"""
    return {
        "setting": {"d_model": d_model},
        "modules": [
            {
                "idx": 0,
                "embed": "nn_embedding",
                "label": "variable",
                "kwargs": {"num_embeddings": vocab_size, "embedding_dim": d_model}
            },
            {
                "idx": 1,
                "embed": "linear",
                "label": "value",
                "kwargs": {"input_dim": 1, "embedding_dim": d_model}
            }
        ]
    }


@pytest.fixture
def sample_input(batch_size, seq_len, vocab_size):
    """Create sample input tensor with [variable_id, value] columns"""
    X = torch.zeros(batch_size, seq_len, 2)
    X[:, :, 0] = torch.randint(0, vocab_size, (batch_size, seq_len)).float()  # variable IDs
    X[:, :, 1] = torch.randn(batch_size, seq_len)  # values
    return X


@pytest.fixture
def attention_layer(d_model, d_qk):
    """Create a standard attention layer"""
    return AttentionLayer(
        attention=ScaledDotAttention,
        d_model_queries=d_model,
        d_model_keys=d_model,
        d_model_values=d_model,
        d_queries_keys=d_qk,
        n_heads=1,
        mask_layer=UniformAttentionMask(),
        attention_dropout=0.0,
        dropout_qkv=0.0
    )


# ============================================================================
# Embedding Tests
# ============================================================================

class TestModularEmbeddingSVFA:
    """Tests for ModularEmbedding in SVFA mode"""
    
    def test_svfa_returns_tuple(self, svfa_embed_config, sample_input):
        """SVFA embedding should return a tuple"""
        emb = ModularEmbedding(ds_embed=svfa_embed_config, comps="svfa", device="cpu")
        output = emb(sample_input)
        
        assert isinstance(output, tuple), "SVFA embedding should return a tuple"
        assert len(output) == 2, "SVFA embedding should return (emb_struct, emb_val)"
    
    def test_svfa_output_shapes(self, svfa_embed_config, sample_input, batch_size, seq_len, d_model):
        """SVFA embedding outputs should have correct shapes"""
        emb = ModularEmbedding(ds_embed=svfa_embed_config, comps="svfa", device="cpu")
        emb_struct, emb_val = emb(sample_input)
        
        assert emb_struct.shape == (batch_size, seq_len, d_model), \
            f"Structure embedding shape mismatch: {emb_struct.shape}"
        assert emb_val.shape == (batch_size, seq_len, d_model), \
            f"Value embedding shape mismatch: {emb_val.shape}"
    
    def test_svfa_requires_structure_role(self, d_model, vocab_size):
        """SVFA should raise error if no structure role is defined"""
        config = {
            "setting": {"d_model": d_model},
            "modules": [
                {
                    "idx": 0,
                    "embed": "linear",
                    "label": "value",
                    "role": "value",
                    "kwargs": {"input_dim": 1, "embedding_dim": d_model}
                }
            ]
        }
        
        with pytest.raises(ValueError, match="structure"):
            ModularEmbedding(ds_embed=config, comps="svfa", device="cpu")
    
    def test_svfa_requires_value_role(self, d_model, vocab_size):
        """SVFA should raise error if no value role is defined"""
        config = {
            "setting": {"d_model": d_model},
            "modules": [
                {
                    "idx": 0,
                    "embed": "nn_embedding",
                    "label": "variable",
                    "role": "structure",
                    "kwargs": {"num_embeddings": vocab_size, "embedding_dim": d_model}
                }
            ]
        }
        
        with pytest.raises(ValueError, match="value"):
            ModularEmbedding(ds_embed=config, comps="svfa", device="cpu")


class TestModularEmbeddingStandard:
    """Tests for ModularEmbedding in standard mode (backward compatibility)"""
    
    def test_standard_returns_tensor(self, standard_embed_config, sample_input):
        """Standard embedding should return a tensor"""
        emb = ModularEmbedding(ds_embed=standard_embed_config, comps="summation", device="cpu")
        output = emb(sample_input)
        
        assert isinstance(output, torch.Tensor), "Standard embedding should return a tensor"
    
    def test_standard_output_shape(self, standard_embed_config, sample_input, batch_size, seq_len, d_model):
        """Standard embedding should have correct shape"""
        emb = ModularEmbedding(ds_embed=standard_embed_config, comps="summation", device="cpu")
        output = emb(sample_input)
        
        assert output.shape == (batch_size, seq_len, d_model), f"Shape mismatch: {output.shape}"


# ============================================================================
# Encoder Tests
# ============================================================================

class TestEncoderSVFA:
    """Tests for Encoder with SVFA factorization"""
    
    def test_encoder_layer_returns_tuple(self, attention_layer, d_model, d_ff, batch_size, seq_len):
        """SVFA encoder layer should return tuple"""
        enc_layer = EncoderLayer(
            global_attention=attention_layer,
            d_model_enc=d_model,
            activation="gelu",
            norm="layer",
            d_ff=d_ff,
            dropout_ff=0.0,
            dropout_attn_out=0.0,
            factorization="svfa"
        )
        
        emb_struct = torch.randn(batch_size, seq_len, d_model)
        emb_val = torch.randn(batch_size, seq_len, d_model)
        X = (emb_struct, emb_val)
        
        output, attn, ent = enc_layer(X=X, mask_miss_k=None, mask_miss_q=None, 
                                       enc_input_pos=None, causal_mask=False)
        
        assert isinstance(output, tuple), "SVFA encoder should return tuple"
    
    def test_encoder_layer_structure_unchanged(self, attention_layer, d_model, d_ff, batch_size, seq_len):
        """Structure embedding should pass through encoder unchanged"""
        enc_layer = EncoderLayer(
            global_attention=attention_layer,
            d_model_enc=d_model,
            activation="gelu",
            norm="layer",
            d_ff=d_ff,
            dropout_ff=0.0,
            dropout_attn_out=0.0,
            factorization="svfa"
        )
        
        emb_struct = torch.randn(batch_size, seq_len, d_model)
        emb_val = torch.randn(batch_size, seq_len, d_model)
        X = (emb_struct, emb_val)
        
        output, _, _ = enc_layer(X=X, mask_miss_k=None, mask_miss_q=None,
                                  enc_input_pos=None, causal_mask=False)
        out_struct, out_val = output
        
        assert torch.allclose(out_struct, emb_struct), \
            "Structure embedding should pass through unchanged"
    
    def test_encoder_layer_value_updated(self, attention_layer, d_model, d_ff, batch_size, seq_len):
        """Value embedding should be updated through encoder"""
        enc_layer = EncoderLayer(
            global_attention=attention_layer,
            d_model_enc=d_model,
            activation="gelu",
            norm="layer",
            d_ff=d_ff,
            dropout_ff=0.0,
            dropout_attn_out=0.0,
            factorization="svfa"
        )
        
        emb_struct = torch.randn(batch_size, seq_len, d_model)
        emb_val = torch.randn(batch_size, seq_len, d_model)
        X = (emb_struct, emb_val)
        
        output, _, _ = enc_layer(X=X, mask_miss_k=None, mask_miss_q=None,
                                  enc_input_pos=None, causal_mask=False)
        _, out_val = output
        
        # Value should be different due to residual connections
        assert not torch.allclose(out_val, emb_val), \
            "Value embedding should be updated"
        assert out_val.shape == (batch_size, seq_len, d_model)


class TestEncoderStandard:
    """Tests for Encoder in standard mode (backward compatibility)"""
    
    def test_encoder_layer_returns_tensor(self, attention_layer, d_model, d_ff, batch_size, seq_len):
        """Standard encoder layer should return tensor"""
        enc_layer = EncoderLayer(
            global_attention=attention_layer,
            d_model_enc=d_model,
            activation="gelu",
            norm="layer",
            d_ff=d_ff,
            dropout_ff=0.0,
            dropout_attn_out=0.0,
            factorization="standard"
        )
        
        X = torch.randn(batch_size, seq_len, d_model)
        
        output, _, _ = enc_layer(X=X, mask_miss_k=None, mask_miss_q=None,
                                  enc_input_pos=None, causal_mask=False)
        
        assert isinstance(output, torch.Tensor), "Standard encoder should return tensor"
        assert output.shape == (batch_size, seq_len, d_model)


# ============================================================================
# Decoder Tests
# ============================================================================

class TestDecoderSVFA:
    """Tests for Decoder with SVFA factorization"""
    
    def test_decoder_layer_returns_tuple(self, d_model, d_ff, d_qk, batch_size):
        """SVFA decoder layer should return tuple"""
        enc_seq_len = 8
        dec_seq_len = 4
        
        self_attention = AttentionLayer(
            attention=ScaledDotAttention,
            d_model_queries=d_model, d_model_keys=d_model, d_model_values=d_model,
            d_queries_keys=d_qk, n_heads=1, mask_layer=UniformAttentionMask(),
            attention_dropout=0.0, dropout_qkv=0.0
        )
        cross_attention = AttentionLayer(
            attention=ScaledDotAttention,
            d_model_queries=d_model, d_model_keys=d_model, d_model_values=d_model,
            d_queries_keys=d_qk, n_heads=1, mask_layer=UniformAttentionMask(),
            attention_dropout=0.0, dropout_qkv=0.0
        )
        
        dec_layer = DecoderLayer(
            global_self_attention=self_attention,
            global_cross_attention=cross_attention,
            d_model_dec=d_model,
            activation="gelu",
            norm="layer",
            d_ff=d_ff,
            dropout_ff=0.0,
            dropout_attn_out=0.0,
            factorization="svfa"
        )
        
        # Decoder input (tuple)
        dec_struct = torch.randn(batch_size, dec_seq_len, d_model)
        dec_val = torch.randn(batch_size, dec_seq_len, d_model)
        X = (dec_struct, dec_val)
        
        # Encoder output (tuple)
        enc_struct = torch.randn(batch_size, enc_seq_len, d_model)
        enc_val = torch.randn(batch_size, enc_seq_len, d_model)
        enc_out = (enc_struct, enc_val)
        
        output, _, _, _, _ = dec_layer(
            X=X, enc_out=enc_out,
            self_mask_miss_k=None, self_mask_miss_q=None,
            cross_mask_miss_k=None, cross_mask_miss_q=None,
            dec_input_pos=None, causal_mask=False
        )
        
        assert isinstance(output, tuple), "SVFA decoder should return tuple"
    
    def test_decoder_layer_structure_unchanged(self, d_model, d_ff, d_qk, batch_size):
        """Structure embedding should pass through decoder unchanged"""
        enc_seq_len = 8
        dec_seq_len = 4
        
        self_attention = AttentionLayer(
            attention=ScaledDotAttention,
            d_model_queries=d_model, d_model_keys=d_model, d_model_values=d_model,
            d_queries_keys=d_qk, n_heads=1, mask_layer=UniformAttentionMask(),
            attention_dropout=0.0, dropout_qkv=0.0
        )
        cross_attention = AttentionLayer(
            attention=ScaledDotAttention,
            d_model_queries=d_model, d_model_keys=d_model, d_model_values=d_model,
            d_queries_keys=d_qk, n_heads=1, mask_layer=UniformAttentionMask(),
            attention_dropout=0.0, dropout_qkv=0.0
        )
        
        dec_layer = DecoderLayer(
            global_self_attention=self_attention,
            global_cross_attention=cross_attention,
            d_model_dec=d_model,
            activation="gelu",
            norm="layer",
            d_ff=d_ff,
            dropout_ff=0.0,
            dropout_attn_out=0.0,
            factorization="svfa"
        )
        
        dec_struct = torch.randn(batch_size, dec_seq_len, d_model)
        dec_val = torch.randn(batch_size, dec_seq_len, d_model)
        X = (dec_struct, dec_val)
        
        enc_struct = torch.randn(batch_size, enc_seq_len, d_model)
        enc_val = torch.randn(batch_size, enc_seq_len, d_model)
        enc_out = (enc_struct, enc_val)
        
        output, _, _, _, _ = dec_layer(
            X=X, enc_out=enc_out,
            self_mask_miss_k=None, self_mask_miss_q=None,
            cross_mask_miss_k=None, cross_mask_miss_q=None,
            dec_input_pos=None, causal_mask=False
        )
        out_struct, _ = output
        
        assert torch.allclose(out_struct, dec_struct), \
            "Decoder structure should pass through unchanged"


# ============================================================================
# ReversedDecoder Tests (SingleCausalLayer)
# ============================================================================

class TestReversedDecoderSVFA:
    """Tests for ReversedDecoder with SVFA factorization (SingleCausalLayer)"""
    
    def test_reversed_decoder_layer_returns_tuple(self, d_model, d_ff, d_qk, batch_size):
        """SVFA ReversedDecoderLayer should return tuple"""
        from causaliT.core.architectures.stage_causal.decoder import ReversedDecoderLayer
        
        dec_seq_len = 4
        ext_seq_len = 8
        
        cross_attention = AttentionLayer(
            attention=ScaledDotAttention,
            d_model_queries=d_model, d_model_keys=d_model, d_model_values=d_model,
            d_queries_keys=d_qk, n_heads=1, mask_layer=UniformAttentionMask(),
            attention_dropout=0.0, dropout_qkv=0.0
        )
        self_attention = AttentionLayer(
            attention=ScaledDotAttention,
            d_model_queries=d_model, d_model_keys=d_model, d_model_values=d_model,
            d_queries_keys=d_qk, n_heads=1, mask_layer=UniformAttentionMask(),
            attention_dropout=0.0, dropout_qkv=0.0
        )
        
        layer = ReversedDecoderLayer(
            global_cross_attention=cross_attention,
            global_self_attention=self_attention,
            d_model_dec=d_model,
            activation="gelu",
            norm="layer",
            d_ff=d_ff,
            dropout_ff=0.0,
            dropout_attn_out=0.0,
            factorization="svfa"
        )
        
        # X as tuple (structure, value)
        X_struct = torch.randn(batch_size, dec_seq_len, d_model)
        X_val = torch.randn(batch_size, dec_seq_len, d_model)
        X = (X_struct, X_val)
        
        # External context as single tensor (like OrthogonalMaskEmbedding)
        external_context = torch.randn(batch_size, ext_seq_len, d_model)
        
        output, cross_att, self_att, cross_ent, self_ent = layer(
            X=X,
            external_context=external_context,
            self_mask_miss_k=None, self_mask_miss_q=None,
            cross_mask_miss_k=None, cross_mask_miss_q=None,
            dec_input_pos=None, causal_mask=False
        )
        
        assert isinstance(output, tuple), "SVFA ReversedDecoderLayer should return tuple"
    
    def test_reversed_decoder_layer_structure_unchanged(self, d_model, d_ff, d_qk, batch_size):
        """Structure embedding should pass through ReversedDecoderLayer unchanged"""
        from causaliT.core.architectures.stage_causal.decoder import ReversedDecoderLayer
        
        dec_seq_len = 4
        ext_seq_len = 8
        
        cross_attention = AttentionLayer(
            attention=ScaledDotAttention,
            d_model_queries=d_model, d_model_keys=d_model, d_model_values=d_model,
            d_queries_keys=d_qk, n_heads=1, mask_layer=UniformAttentionMask(),
            attention_dropout=0.0, dropout_qkv=0.0
        )
        self_attention = AttentionLayer(
            attention=ScaledDotAttention,
            d_model_queries=d_model, d_model_keys=d_model, d_model_values=d_model,
            d_queries_keys=d_qk, n_heads=1, mask_layer=UniformAttentionMask(),
            attention_dropout=0.0, dropout_qkv=0.0
        )
        
        layer = ReversedDecoderLayer(
            global_cross_attention=cross_attention,
            global_self_attention=self_attention,
            d_model_dec=d_model,
            activation="gelu",
            norm="layer",
            d_ff=d_ff,
            dropout_ff=0.0,
            dropout_attn_out=0.0,
            factorization="svfa"
        )
        
        X_struct = torch.randn(batch_size, dec_seq_len, d_model)
        X_val = torch.randn(batch_size, dec_seq_len, d_model)
        X = (X_struct, X_val)
        external_context = torch.randn(batch_size, ext_seq_len, d_model)
        
        output, _, _, _, _ = layer(
            X=X,
            external_context=external_context,
            self_mask_miss_k=None, self_mask_miss_q=None,
            cross_mask_miss_k=None, cross_mask_miss_q=None,
            dec_input_pos=None, causal_mask=False
        )
        out_struct, _ = output
        
        assert torch.allclose(out_struct, X_struct), \
            "Structure should pass through ReversedDecoderLayer unchanged"
    
    def test_reversed_decoder_with_tuple_external_context(self, d_model, d_ff, d_qk, batch_size):
        """ReversedDecoderLayer should handle tuple external_context in SVFA mode"""
        from causaliT.core.architectures.stage_causal.decoder import ReversedDecoderLayer
        
        dec_seq_len = 4
        ext_seq_len = 8
        
        cross_attention = AttentionLayer(
            attention=ScaledDotAttention,
            d_model_queries=d_model, d_model_keys=d_model, d_model_values=d_model,
            d_queries_keys=d_qk, n_heads=1, mask_layer=UniformAttentionMask(),
            attention_dropout=0.0, dropout_qkv=0.0
        )
        self_attention = AttentionLayer(
            attention=ScaledDotAttention,
            d_model_queries=d_model, d_model_keys=d_model, d_model_values=d_model,
            d_queries_keys=d_qk, n_heads=1, mask_layer=UniformAttentionMask(),
            attention_dropout=0.0, dropout_qkv=0.0
        )
        
        layer = ReversedDecoderLayer(
            global_cross_attention=cross_attention,
            global_self_attention=self_attention,
            d_model_dec=d_model,
            activation="gelu",
            norm="layer",
            d_ff=d_ff,
            dropout_ff=0.0,
            dropout_attn_out=0.0,
            factorization="svfa"
        )
        
        X_struct = torch.randn(batch_size, dec_seq_len, d_model)
        X_val = torch.randn(batch_size, dec_seq_len, d_model)
        X = (X_struct, X_val)
        
        # External context as tuple (for when source also uses SVFA)
        ext_struct = torch.randn(batch_size, ext_seq_len, d_model)
        ext_val = torch.randn(batch_size, ext_seq_len, d_model)
        external_context = (ext_struct, ext_val)
        
        output, _, _, _, _ = layer(
            X=X,
            external_context=external_context,
            self_mask_miss_k=None, self_mask_miss_q=None,
            cross_mask_miss_k=None, cross_mask_miss_q=None,
            dec_input_pos=None, causal_mask=False
        )
        
        assert isinstance(output, tuple), "Should handle tuple external_context"


# ============================================================================
# Gradient Flow Tests
# ============================================================================

class TestGradientFlow:
    """Tests for gradient flow through SVFA"""
    
    def test_gradients_flow_through_encoder(self, attention_layer, d_model, d_ff, batch_size, seq_len):
        """Gradients should flow through SVFA encoder"""
        enc_layer = EncoderLayer(
            global_attention=attention_layer,
            d_model_enc=d_model,
            activation="gelu",
            norm="layer",
            d_ff=d_ff,
            dropout_ff=0.0,
            dropout_attn_out=0.0,
            factorization="svfa"
        )
        
        emb_struct = torch.randn(batch_size, seq_len, d_model, requires_grad=True)
        emb_val = torch.randn(batch_size, seq_len, d_model, requires_grad=True)
        X = (emb_struct, emb_val)
        
        output, _, _ = enc_layer(X=X, mask_miss_k=None, mask_miss_q=None,
                                  enc_input_pos=None, causal_mask=False)
        _, out_val = output
        
        loss = out_val.sum()
        loss.backward()
        
        assert emb_struct.grad is not None, "Structure embedding should have gradients"
        assert emb_val.grad is not None, "Value embedding should have gradients"
        assert emb_val.grad.abs().sum() > 0, "Value embedding gradients should be non-zero"
    
    def test_structure_grad_from_attention(self, attention_layer, d_model, d_ff, batch_size, seq_len):
        """Structure embedding should receive gradients from attention computation"""
        enc_layer = EncoderLayer(
            global_attention=attention_layer,
            d_model_enc=d_model,
            activation="gelu",
            norm="layer",
            d_ff=d_ff,
            dropout_ff=0.0,
            dropout_attn_out=0.0,
            factorization="svfa"
        )
        
        emb_struct = torch.randn(batch_size, seq_len, d_model, requires_grad=True)
        emb_val = torch.randn(batch_size, seq_len, d_model, requires_grad=True)
        X = (emb_struct, emb_val)
        
        output, _, _ = enc_layer(X=X, mask_miss_k=None, mask_miss_q=None,
                                  enc_input_pos=None, causal_mask=False)
        _, out_val = output
        
        loss = out_val.sum()
        loss.backward()
        
        # Structure gradients come from attention Q, K computation
        assert emb_struct.grad.abs().sum() > 0, \
            "Structure embedding should receive gradients from attention"


# ============================================================================
# Run tests if executed directly
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
