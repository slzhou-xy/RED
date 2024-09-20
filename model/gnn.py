import torch
import torch.nn as nn
from torch_geometric.nn import GATConv


class ShortCut(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.connection = nn.Linear(in_dim, out_dim)

        nn.init.xavier_uniform_(self.connection.weight)
        if self.connection.bias is not None:
            torch.nn.init.zeros_(self.connection.bias)

    def forward(self, x):
        return self.connection(x)


class GnnLayer(nn.Module):
    def __init__(self, in_channels, out_channels, heads, act, short_cut=True, dropout=0):
        super().__init__()
        self.layer = GATConv(
            in_channels=in_channels,
            out_channels=out_channels,
            heads=heads,
            dropout=dropout,
            add_self_loops=True,
        )
        self.short_cut = ShortCut(in_channels, out_channels * heads)
        if short_cut is False:
            self.short_cut = None
        self.act = act

    def forward(self, data):
        x, edge_index = data
        gx = self.layer(x, edge_index)
        if self.short_cut is not None:
            gx += self.short_cut(x)
        gx = gx if self.act is None else self.act(gx)
        return (gx, edge_index)


class GnnEmbedding(nn.Module):
    def __init__(self, fea_size, g_dim_per_layer, g_heads_per_layer, num_layers, dropout=0):
        super().__init__()
        g_dim_per_layer = [fea_size] + g_dim_per_layer
        g_heads_per_layer = [1] + g_heads_per_layer
        gat_layers = []
        for i in range(num_layers):
            layer = GnnLayer(
                in_channels=g_dim_per_layer[i] * g_heads_per_layer[i],
                out_channels=g_dim_per_layer[i + 1],
                heads=g_heads_per_layer[i + 1],
                act=nn.GELU() if i != num_layers - 1 else None,
                short_cut=True if i != 0 else False,
                dropout=dropout,
            )
            gat_layers.append(layer)

        self.gat_net = nn.Sequential(
            *gat_layers,
        )

    def forward(self, x, edge_index):
        data = (x, edge_index)
        x, edge_index = self.gat_net(data)
        return x
