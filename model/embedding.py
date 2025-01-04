import math
import torch
import torch.nn as nn
from model.gnn import GnnEmbedding


class UserEmbedding(nn.Module):
    def __init__(self, user_size, embed_size):
        super().__init__()
        self.user_emb = nn.Embedding(num_embeddings=user_size + 1, embedding_dim=embed_size, padding_idx=0)

    def forward(self, x):
        return self.user_emb(x)


class ContextEmbedding(nn.Module):
    def __init__(self, context_size, embed_size):
        super().__init__()
        self.cte_emb = nn.Embedding(num_embeddings=context_size + 1, embedding_dim=embed_size, padding_idx=0)

    def forward(self, x):
        return self.cte_emb(x)


class SpatialEmbedding(nn.Module):
    def __init__(self, fea_size, g_dim_per_layer, g_heads_per_layer, num_layers, dropout):
        super().__init__()

        g_dim_per_layer = eval(g_dim_per_layer)
        g_heads_per_layer = eval(g_heads_per_layer)
        assert num_layers == len(g_dim_per_layer) == len(g_heads_per_layer)

        self.padding = nn.Parameter(data=torch.zeros(g_dim_per_layer[-1]), requires_grad=False)
        self.mask = nn.Parameter(data=torch.randn(g_dim_per_layer[-1]), requires_grad=True)
        self.spe = GnnEmbedding(fea_size=fea_size,
                                g_dim_per_layer=g_dim_per_layer,
                                g_heads_per_layer=g_heads_per_layer,
                                num_layers=num_layers,
                                dropout=dropout)
        torch.nn.init.normal_(self.mask, std=.02)

    def forward(self, x, edge_index):
        return torch.vstack([self.padding, self.mask, self.spe(x, edge_index)])


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=256):
        super().__init__()
        self.d_model = d_model
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)  # (max_len, d_model/2)
        pe[:, 1::2] = torch.cos(position * div_term)  # (max_len, d_model/2)

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)].detach()
