import torch
import torch.nn as nn
import torch.nn.functional as F
# from einops import rearrange, repeat

from os.path import dirname, abspath
import sys
# root_path = dirname(dirname(dirname(abspath(__file__))))
# sys.path.append(root_path)
from proT.core.modules.embedding_layers import Time2Vec, SinusoidalPosition, identity_emb, nn_embedding, linear_emb


class EmbeddingMap(nn.Module):
    """Mapping between dataset variable index and embedding
    """
    def __init__(self,var_idx:int ,embedding:nn.Module, kwargs:dict, device):
        super().__init__()
        self.var_idx = var_idx
        self.embedding = embedding(**kwargs,device=device) 
        
        
        
    def __call__(self,X: torch.Tensor):
        
        # before every embedding, nan values are replaced by 0
        X_ = torch.nan_to_num(X[:,:,self.var_idx])
        # out = self.embedding(X_)
        
        try:
            out = self.embedding(X_)
        except:
            # in case the embedding is a lookup table, the input must be int
            X_ = X_.to(torch.long)
            out = self.embedding(X_)
            
        return out




class ModularEmbedding(nn.Module):
    """
    Compatible with configuration files version 4.1.0
    It builds the embedding specific for the dataset 
    though the instructions in "ds_embed" 'dictionary argument.
    
    Dictionary example:
    ds_embed = [
        {
            "idx":0,
            "embed":"nn_embedding",
            "kwargs": {"num_embeddings":vocab_size, "embedding_dim":d_model}
            },
        {
            "idx": 1,
            "embed":"nn_embedding",
            "kwargs": {"num_embeddings":vocab_size, "embedding_dim":d_model}
            },
    ]

    Options
    - embed: "mask", "nn_embedding", "time2vec", "identity", "pass"
    
    """
    def __init__(
        self,
        ds_embed:dict,
        comps:str,
        device,
        
    ):
        super().__init__()
        
        assert comps in ["concat","summation","spatiotemporal","sum_val_to_concat"]
        
        self.comps = comps
        
        # assemble list of embeddings according to "ds_embed"
        self.embed_list = []
        self.embed_label_list = []
        self.pass_idx_list = None
        self.mask_idx = None
        self.mask_given_idx = None
        self.val_idx = None
        
        # unpack settings for spatiotemporal
        d_model = ds_embed["setting"]["d_model"] if comps == "spatiotemporal" else None
        d_time = ds_embed["setting"]["d_time"]  if comps == "spatiotemporal" else None
        d_val = ds_embed["setting"]["d_value"]  if comps == "spatiotemporal" else None
        
        if comps == "spatiotemporal":
            self.W_time_val = nn.Linear(d_time+d_val, d_model, bias=True)
        
        for var in ds_embed["modules"]:
            
            idx_, embed_, label_ ,kwargs = var["idx"], var["embed"], var["label"], var["kwargs"]
            
            # assign embedding layers
            assert embed_ in ["mask", "mask_given", "nn_embedding", "sinusoidal","time2vec","identity","linear","pass","value"], AssertionError("Invalid embedding selected!")
            
            if embed_ == "mask":
                self.mask_idx = idx_
                emb_module = None
                
            if embed_ == "mask_given":
                self.mask_given_idx = idx_
                emb_module = None
                
            if embed_ == "nn_embedding":
                emb_module = nn_embedding
                
            if embed_ == "sinusoidal":
                emb_module = SinusoidalPosition
                
            if embed_ == "time2vec":
                emb_module = Time2Vec
                
            if embed_ == "identity":
                emb_module = identity_emb
                
            if embed_ == "linear":
                emb_module = linear_emb
            
            # store value index
            if embed_ == "value":
                emb_module = None             # do not append in the Embedding List
                
                if self.val_idx is None:
                    self.val_idx = idx_
            
            
            # store index to pass
            if embed_ == "pass":
                emb_module = None             # do not append in the Embedding List
                
                if self.pass_idx_list is None:
                    self.pass_idx_list = []
                
                self.pass_idx_list.append(idx_)
                
                
            if emb_module is not None:
                self.embed_list.append(EmbeddingMap(var_idx=idx_ ,embedding=emb_module, kwargs=kwargs, device=device))
                
                if label_ is not None:
                    self.embed_label_list.append(label_)
                else:
                    self.embed_label_list.append("empty_label")
                    
        
        # save the list into a ModuleList to train/save them
        self.embed_modules_list = nn.ModuleList(self.embed_list)


    def __call__(
        self, 
        X: torch.Tensor, 
        mask_given: torch.Tensor=None # position mask
        )->torch.Tensor:
        """
        Choose the embedding strategy and apply it to X
        Args:
            X (torch.Tensor): data to be embedded
        Returns:
            torch.Tensor: embedded data
        """
        
        if mask_given is not None:
            mask = mask_given.long()+1 # zero reserved for missing values
            X_ = torch.cat([X, mask], dim=-1)
        else:
            X_ = X
        
        if self.comps == "concat":
            return self.concat(X_)
        
        elif self.comps == "summation_value":
            return self.summation_value(X_)
        
        elif self.comps == "summation":
            return self.summation(X_)
        
        elif self.comps == "spatiotemporal":
            return self.spatiotemporal(X_)
        
        
    def concat(self, X: torch.Tensor)->torch.Tensor:
        """
        Returns all concat embeddings on the last dimension
        out tensor shape (BS, seq_length, sum(d_embed))   
        """     
        return torch.cat([embed(X) for embed in self.embed_list],dim=-1)
    
    
    def summation(self, X: torch.Tensor):
        """
        Returns all summed embeddings on the last dimension
        all embedding dimensions must be = d_model
        out tensor shape (BS, seq_length, d_model)   
        """ 
        
        return torch.sum(torch.concat([embed(X).unsqueeze(-1) for embed in self.embed_list if embed(X).shape[-1]!=0],dim=-1),dim=-1)
    
    
    
    def summation_value(self, X: torch.Tensor):
        # ! Currently not working --> returns nan because val can have missing points
        assert self.val_idx is not None, AssertionError("No value index defined!")
        val = X[:,:,self.val_idx]
        embed = self.concat(X)
        return val.unsqueeze(-1) + embed
    
    
    def spatiotemporal(self, X: torch.Tensor)->torch.Tensor:
        """
        Spatiotemporal composition from Spacetimeformer.
        To use this composition, the following instructions must be followed:
        - a model dimension must be chosen (d_model)
        - values must be embedded using the "identity" embedding
        - time must be embedded using the "time2vec" embedding
        - position must be embedded using "nn_embedding" embedding with d_model as embedding dimension
        - variable must be embedded using "nn_embedding" embedding with d_model as embedding dimension
        Args:
            X (torch.Tensor): data to be embedded 
        """     
        
        X_emb = [embed(X) for embed in self.embed_list]
        
        var_idx = self.embed_label_list.index("variable")
        pos_idx = self.embed_label_list.index("position")
        time_idx = self.embed_label_list.index("time")
        val_idx = self.embed_label_list.index("value")
        
        pos = torch.nan_to_num(X_emb[pos_idx])
        time = torch.nan_to_num(X_emb[time_idx])
        val = torch.nan_to_num(X_emb[val_idx])
        var = torch.nan_to_num(X_emb[var_idx])
        time_val = torch.cat([time, val], dim=-1)
        time_val_emb = self.W_time_val(time_val)
        
        # for ablation
        res = time_val_emb + pos if pos.shape[-1]!=0 else time_val_emb
        res = time_val_emb + var if var.shape[-1]!=0 else res
        
        return res
    
    
    def get_mask_tensor(self, X: torch.Tensor)->torch.Tensor:
        return X[:,:,self.mask_idx] if self.mask_idx is not None else None
    
    
    def get_mask(self, X: torch.Tensor)->torch.Tensor:
        """
        Negative mask for nan in the value dimension
        True means missing value, False means present value

        Args:
            X (torch.Tensor): input tensor BxLxD

        Returns:
            torch.Tensor: _description_ BxLx1
        """
        X_mask = self.get_mask_tensor(X)
        
        if X_mask is None:
            return None
        else: 
            return X_mask.isnan().unsqueeze(-1)
        
        
        
        
    
    
    def pass_var(self, X: torch.Tensor)->torch.Tensor:
        return X[:,:,self.pass_idx_list] if self.pass_idx_list is not None else None




def main():
    # quick test
    
    X = torch.rand(5,10,3)
    X[:,:,0] = torch.arange(10)
    X[:,:,1] = torch.zeros(10)
    X[:,:,2] = 100*torch.arange(10)
    d_model = 4
    vocab_size = 15

    ds_embed = [
        {
            "idx":0,
            "embed":"nn_embedding",
            "kwargs": {"num_embeddings":vocab_size, "embedding_dim":d_model}
            },
        {
            "idx": 1,
            "embed":"nn_embedding",
            "kwargs": {"num_embeddings":vocab_size, "embedding_dim":d_model}
            },
        {
            "idx": 2,
            "embed": "value",
            "kwargs": None
            },
    ]
    
    embed = ModularEmbedding(ds_embed,"sum_val_to_concat",device="cpu")
    X_emb = embed(X)
    print(X_emb.shape)
    print(embed.pass_idx_list)
    print(embed.pass_var(X))
    
if __name__ == "__main__":
    main()
