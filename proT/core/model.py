# Standard library imports
import sys
import warnings
from functools import partial
from os.path import dirname, abspath

# Third-party imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as pyd

# Local imports
# ROOT_DIR = dirname(abspath(__file__))
# sys.path.append(ROOT_DIR)
from proT.core.modules import (
    LieAttention, ScaledDotAttention, AttentionLayer,
    Decoder, DecoderLayer,
    ModularEmbedding,
    Encoder, EncoderLayer,
    Normalization, UniformAttentionMask
)


class ProT(nn.Module):
    """
    ProT based on Spacetimeformer (https://github.com/QData/spacetimeformer)
    Required data shapes: (BATCH_SIZE, sequence_length, variables)
    """
    def __init__(
        self,
        model: str,
        
        # embeddings
        ds_embed_enc,
        comps_embed_enc,
        ds_embed_dec,
        comps_embed_dec,
        
        # attention
        enc_attention_type,
        enc_mask_type,
        dec_self_attention_type,
        dec_self_mask_type,
        dec_cross_attention_type,
        dec_cross_mask_type,
        n_heads: int,
        #causal_mask: bool,
        enc_causal_mask: bool,
        dec_causal_mask: bool,
        #attn_factor: int = 5, #TODO understand, DO NOT DEL for now!
        
        
        # dropout
        dropout_emb: float,
        dropout_data: float, # from old embeddings
        dropout_attn_out: float,
        dropout_ff: float ,
        enc_dropout_qkv: float,
        enc_attention_dropout: float,
        dec_self_dropout_qkv: float,
        dec_self_attention_dropout: float,
        dec_cross_dropout_qkv: float,
        dec_cross_attention_dropout: float,
        
        # options
        e_layers: int,
        d_layers: int,
        activation: str,
        norm: str,
        use_final_norm: bool,
        device,
        out_dim: int,
        d_ff: int,
        d_model_enc: int,
        d_model_dec: int,
        d_qk: int,
        given_target_max_pos: int = None,
        ):
        super().__init__()
        
        # embeddings. separate enc/dec in case the variable indices are not aligned
        self.enc_embedding = ModularEmbedding(
            ds_embed=ds_embed_enc,
            comps=comps_embed_enc,
            device=device)
        
        self.dec_embedding = ModularEmbedding(
            ds_embed=ds_embed_dec,
            comps=comps_embed_dec,
            device=device)
        
        # Select Attention Options
        attn_shared_kwargs = {
            "n_heads"           : n_heads,
            "d_queries_keys"    : d_qk,
        }

        attn_enc_kwargs = {
            "d_model_queries"   : d_model_enc,
            "d_model_keys"      : d_model_enc,
            "d_model_values"    : d_model_enc,
            "attention_type"    : enc_attention_type,
            "mask_type"         : enc_mask_type,
            "dropout_qkv"       : enc_dropout_qkv,
            "attention_dropout" : enc_attention_dropout,
            "register_entropy"  : True,                     # Enable entropy registration for encoder self-attention
            "layer_name"        : "enc_self_att"            # Name for entropy registration
        }
        
        attn_dec_self_kwargs = {
            "d_model_queries"   : d_model_dec,
            "d_model_keys"      : d_model_dec,
            "d_model_values"    : d_model_dec,
            "attention_type"    : dec_self_attention_type,
            "mask_type"         : dec_self_mask_type,
            "dropout_qkv"       : dec_self_dropout_qkv,
            "attention_dropout" : dec_self_attention_dropout,
            "register_entropy"  : True,                     # Enable entropy registration for decoder self-attention
            "layer_name"        : "enc_self_att"            # Name for entropy registration
        }
        
        attn_dec_cross_kwargs = {
            "d_model_queries"   : d_model_dec,
            "d_model_keys"      : d_model_enc,
            "d_model_values"    : d_model_dec,
            "attention_type"    : dec_cross_attention_type,
            "mask_type"         : dec_cross_mask_type,
            "dropout_qkv"       : dec_cross_dropout_qkv,
            "attention_dropout" : dec_cross_attention_dropout,
            "register_entropy" : True,                      # Enable entropy registration for cross-attention
            "layer_name": "cross_att"                       # Name for entropy registration
        }
        
        #self.causal_mask = causal_mask
        self.enc_causal_mask = enc_causal_mask
        self.dec_causal_mask = dec_causal_mask
        
        self.encoder = Encoder(
            encoder_layers=[
                EncoderLayer(
                    global_attention=self._attn(**(attn_shared_kwargs | attn_enc_kwargs)),
                    d_model_enc=d_model_enc,
                    d_ff=d_ff,
                    dropout_ff=dropout_ff,
                    dropout_attn_out=dropout_attn_out,
                    activation=activation,
                    norm=norm
                    ) for _ in range(e_layers)],
            
            #conv_layers = [ConvBlock(split_length_into=split_length_into, d_model=d_model) for l in range(intermediate_downsample_convs)],
            norm_layer = Normalization(norm, d_model=d_model_enc) if use_final_norm else None,
            emb_dropout = dropout_emb
            )


        self.decoder = Decoder(
            decoder_layers=[
                DecoderLayer(
                    global_self_attention=self._attn(**(attn_shared_kwargs | attn_dec_self_kwargs)),
                    global_cross_attention=self._attn(**(attn_shared_kwargs | attn_dec_cross_kwargs)),
                    d_model_dec=d_model_dec,
                    d_ff=d_ff,
                    dropout_ff=dropout_ff,
                    dropout_attn_out=dropout_attn_out,
                    activation=activation,
                    norm=norm,
                    ) for _ in range(d_layers)],

            norm_layer=Normalization(norm, d_model=d_model_dec) if use_final_norm else None,
            emb_dropout=dropout_emb
            )
        
        # final linear layer turns Transformer output into predictions
        self.forecaster = nn.Linear(d_model_dec, out_dim, bias=False)
        
        
    def forward(
        self,
        input_tensor,
        target_tensor,
        trg_pos_mask=None
        ):
        
        # embed input and get mask for missing values
        enc_input = self.enc_embedding(X=input_tensor)
        enc_input_pos = self.enc_embedding.pass_var(X=input_tensor) #TODO still relevant?
        enc_mask = self.enc_embedding.get_mask(X=input_tensor)
        
        if enc_input_pos is None and self.enc_causal_mask == True:
            self.enc_causal_mask = False
            warnings.warn(f"encoder causal_mask required {self.enc_causal_mask} but encoder got null input positions, set to False.")
        
        # pass embedded input to encoder
        enc_out, enc_self_att, enc_self_ent = self.encoder(
            X=enc_input,
            mask_miss_k=enc_mask,
            mask_miss_q=enc_mask,
            enc_input_pos=enc_input_pos,
            causal_mask=self.enc_causal_mask
            )
        
        # embed target
        dec_input = self.dec_embedding(X=target_tensor, mask_given=trg_pos_mask)
        dec_input_pos = self.dec_embedding.pass_var(X=target_tensor)
        dec_self_mask=self.dec_embedding.get_mask(X=target_tensor)
        
        # get mask for given targets, they don't participate in the cross attention
        if trg_pos_mask is not None:
            dec_cross_mask = torch.logical_or(dec_self_mask, torch.logical_not(trg_pos_mask))
        else:
            dec_cross_mask = dec_self_mask
        
        if dec_input_pos is None and self.dec_causal_mask == True:
            warnings.warn(f"decoder causal_mask required {self.dec_causal_mask} but encoder got null input positions, set to False.")
        
        # pass embedded target and encoder output to decoder
        dec_out, dec_self_att, dec_cross_att, dec_self_ent, dec_cross_ent = self.decoder(
            X=dec_input,
            enc_out=enc_out,
            self_mask_miss_k=dec_self_mask, 
            self_mask_miss_q=dec_self_mask,
            cross_mask_miss_k=enc_mask, 
            cross_mask_miss_q=dec_cross_mask,
            dec_input_pos = dec_input_pos,
            causal_mask=self.dec_causal_mask
            )
        
        # forecasting predictions
        forecast_out = self.forecaster(dec_out)
        
        return forecast_out, (enc_self_att, dec_self_att, dec_cross_att), enc_mask, (enc_self_ent, dec_self_ent, dec_cross_ent)
    
    
    def _attn(
        self,
        d_model_queries: int,
        d_model_keys: int,
        d_model_values: int,
        n_heads: int,
        d_queries_keys: int,
        attention_type: str,
        mask_type: str,
        dropout_qkv: float,
        attention_dropout: float,
        register_entropy: bool, 
        layer_name: str
        ):

        # choose attention type
        assert attention_type in ["ScaledDotProduct", "LieAttention"]
            
        if attention_type == "ScaledDotProduct":
            attention_module = ScaledDotAttention
            
        if attention_type == "LieAttention":
            attention_module = LieAttention
            
        # choose mask type (currently only uniform)
        mask_layer = None # init
        
        if mask_type is not None:
            if mask_type == "Uniform":
                mask_layer = UniformAttentionMask()

            
        att = AttentionLayer(
            attention=attention_module,
            d_model_queries=d_model_queries,
            d_model_keys=d_model_keys,
            d_model_values=d_model_values,
            d_queries_keys=d_queries_keys,
            n_heads=n_heads,
            mask_layer=mask_layer,
            attention_dropout=attention_dropout,
            dropout_qkv=dropout_qkv,
            register_entropy=register_entropy,
            layer_name=layer_name
            )
            
        return att
