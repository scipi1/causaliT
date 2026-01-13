import torch
from torch import nn

class Time2Vec(nn.Module):
    """
    Time2Vec embeddings (https://arxiv.org/abs/1907.05321) from Borealis AI
    implementation from Spacetimeformer 
    """
    def __init__(self, input_dim:int, embed_dim:int, device):
        super(Time2Vec, self).__init__()
        activation = torch.sin
        assert embed_dim % input_dim == 0
        
        self.embed_dim = embed_dim // input_dim # so that the final dimension is embed_dim
        self.input_dim = input_dim
        self.activation = activation
        
        # initialize learnable weights and biases
        self.embed_weight = nn.parameter.Parameter(torch.rand(self.input_dim,self.embed_dim,device=device))
        self.embed_bias = nn.parameter.Parameter(torch.rand(self.input_dim,self.embed_dim,device=device))

    def forward(self, x: torch.Tensor):
        if self.embed_dim == 0:
            # for ablation study
            return torch.empty((x.shape[0], x.shape[1],0), device=x.get_device())
        
        else:
            x = torch.nan_to_num(x) # shape: (B, L, input_dim)

            x_diag = torch.diag_embed(x).clone().detach()
            x_affine = torch.matmul(x_diag, self.embed_weight) + self.embed_bias # shape: (B, L, input_dim, embed_dim)
            
            # separate the first dimension (no activation applied)
            x_affine_0, x_affine_remain = torch.split(x_affine, [1, self.embed_dim - 1], dim=-1) # shapes: (B, L, 1) and (B, L, emb_dim-1)
            
            # apply activation on the remaining dimensions
            x_affine_remain = self.activation(x_affine_remain)
            
            # join again the zero and activated dimensions
            x_out = torch.cat([x_affine_0, x_affine_remain], dim=-1)
            
            # different time components are concatenated
            x_out = x_out.view(x_out.size(0), x_out.size(1), -1)
            return x_out



class SinusoidalPosition(nn.Module):
    """
    Sinusoidal positional embedding 
    used in "Attention is all you need" (https://arxiv.org/abs/1706.03762)
    Embedding type: absolute & fixed
    """
    
    
    def __init__(self, max_pos:int, embed_dim:int, device):
        super().__init__()
        
        n = 10000.0 # internal variable, this n=10000 in the original paper
        
        assert embed_dim % 2 == 0, AssertionError("Sinusoidal positional embedding cannot apply to odd token embedding dim (got dim={:d})".format(embed_dim))
        
        positions = torch.arange(0, max_pos).unsqueeze_(1)
        denominators = torch.pow(n, 2*torch.arange(0, embed_dim//2)/embed_dim) # 10000^(2i/d_model), i is the index of embedding
        
        self.embeddings = torch.zeros(max_pos, embed_dim, device=device)
        self.embeddings[:, 0::2] = torch.sin(positions/denominators) # sin(pos/10000^(2i/d_model))
        self.embeddings[:, 1::2] = torch.cos(positions/denominators) # cos(pos/10000^(2i/d_model))
        
    def forward(self, x: torch.Tensor):
        return self.embeddings[x]



class identity_emb(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
    
    def forward(self, X: torch.Tensor):
        indentity_fun = nn.Identity(device=self.device, dtype=torch.float32)
        return indentity_fun(X.unsqueeze(-1))




class nn_embedding(nn.Module):
    def __init__(self, num_embeddings,embedding_dim, device, *args, **kwargs):
        super().__init__()
        self.embed_dim = embedding_dim
        self.embedding = nn.Embedding(num_embeddings, embedding_dim, device=device, dtype=torch.float32, *args, **kwargs)
    
    def forward(self, X: torch.Tensor):
        if self.embed_dim == 0:
            return torch.empty((X.shape[0], X.shape[1],0), device=X.get_device())
        else:
            X = X.to(torch.long)
            return self.embedding(X)



class linear_emb(nn.Module):
    def __init__(self, input_dim, embedding_dim, device):
        super().__init__()
        self.embedding = nn.Linear(in_features=input_dim, out_features=embedding_dim, device=device, dtype=torch.float32)
        
    def forward(self, X: torch.Tensor):
        return self.embedding(X.unsqueeze(-1))
        



def main():
    # quick Time2Vec test
    
    BATCH_SIZE = 1
    seq_len = 5
    time_dim = 5
    embed_dim = 10
    
    x_test = torch.rand(BATCH_SIZE,seq_len,time_dim)
    time_embed = Time2Vec(input_dim=time_dim, embed_dim=embed_dim)
    
    x_out = time_embed.forward(x_test)
    print(f"Actual latent dimension: {embed_dim} --> {time_embed.embed_dim}")
    print(f"X input shape check: {x_test.shape} <--> {BATCH_SIZE},{seq_len},{time_dim}")
    print(f"X output shape: {x_out.shape} <--> {BATCH_SIZE},{seq_len},{embed_dim}")
    
if __name__ == "__main__":
    main()

