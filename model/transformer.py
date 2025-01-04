import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.mat_dim = 64
        self.temporal_mat_bias = nn.Linear(1, self.mat_dim, bias=True)
        self.dis_mat_bias = nn.Linear(1, self.mat_dim, bias=True)
        self.td_mat_bias = nn.Linear(self.mat_dim, 1, bias=True)

    def scaled_dot_product_attention(self, Q, K, V, temporal_mat, dis_mat, mask=None, lambda2=0.5):
        # Calculate attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        temporal_mat = 1.0 / torch.log(torch.exp(torch.tensor(1.0).to(attn_scores.device)) + temporal_mat)
        dis_mat = 1.0 / torch.log(torch.exp(torch.tensor(1.0).to(attn_scores.device)) + dis_mat)

        temporal_mat = F.relu(self.temporal_mat_bias(temporal_mat.unsqueeze(-1)))
        dis_mat = F.relu(self.dis_mat_bias(dis_mat.unsqueeze(-1)))
        td_mat = lambda2 * dis_mat + (1 - lambda2) * temporal_mat
        td_mat = self.td_mat_bias(td_mat).squeeze(-1).unsqueeze(1)
        attn_scores = attn_scores + td_mat

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 1, -1e9)

        attn_probs = torch.softmax(attn_scores, dim=-1)

        output = torch.matmul(attn_probs, V)
        return output

    def split_heads(self, x):
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)

    def forward(self, Q, K, V, temporal_mat, dis_mat, mask=None, lambda2=0.5):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))

        attn_output = self.scaled_dot_product_attention(Q, K, V, temporal_mat, dis_mat, mask, lambda2)

        output = self.W_o(self.combine_heads(attn_output))
        return output


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(TransformerBlock, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask, temporal_mat, dis_mat, lambda2=0.5):
        x = self.norm1(x)
        attn_output = self.self_attn(x, x, x, temporal_mat, dis_mat, mask, lambda2)
        x = x + self.dropout(attn_output)
        x = self.norm2(x)
        ff_output = self.feed_forward(x)
        x = x + self.dropout(ff_output)
        return x
