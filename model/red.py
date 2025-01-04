import torch
import torch.nn as nn
import torch.nn.functional as F
from .transformer import TransformerBlock
from .embedding import SpatialEmbedding, UserEmbedding, PositionalEmbedding, ContextEmbedding


class Encoder(nn.Module):
    def __init__(self,
                 embed_dim=128,
                 depth=6,
                 num_heads=8,
                 ffn_size=128 * 4,
                 tfm_dropout=0.1,
                 ):
        super().__init__()

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ffn_size, tfm_dropout) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, mask, temporal_mat, dis_mat, lambda2=0.5):
        for blk in self.blocks:
            x = blk(x, mask, temporal_mat, dis_mat, lambda2=0.5)
        x = self.norm(x)

        return x


class Decoder(nn.Module):
    def __init__(self,
                 embed_dim=128,
                 depth=6,
                 num_heads=8,
                 ffn_size=128 * 4,
                 tfm_dropout=0.1
                 ):
        super().__init__()

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ffn_size, tfm_dropout) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, mask, temporal_mat, dis_mat, lambda2=0.5):
        for blk in self.blocks:
            x = blk(x, mask, temporal_mat, dis_mat, lambda2=0.5)
        x = self.norm(x)
        return x


class RED(nn.Module):
    def __init__(self,
                 fea_size=13,
                 g_heads_per_layer='[8, 16, 1]',
                 g_dim_per_layer='[16, 16, 128]',
                 g_depths=3,
                 g_dropout=0.1,
                 enc_embed_dim=128,
                 enc_ffn_dim=128 * 4,
                 enc_depths=6,
                 enc_num_heads=8,
                 enc_emb_dropout=0.1,
                 enc_tfm_dropout=0.1,
                 dec_embed_dim=128,
                 dec_ffn_dim=128 * 4,
                 dec_depths=6,
                 dec_num_heads=8,
                 dec_emb_dropout=0.1,
                 dec_tfm_dropout=0.1,
                 vocab_size=30000,
                 user_size=0,
                 context_size=0,
                 ):
        super().__init__()

        # embedding
        self.spe = SpatialEmbedding(fea_size, g_dim_per_layer, g_heads_per_layer, g_depths, g_dropout)
        self.ce = ContextEmbedding(context_size, enc_embed_dim // 2)
        self.ue = UserEmbedding(user_size, enc_embed_dim)
        self.pe = PositionalEmbedding(enc_embed_dim)

        # encoder
        self.enc_dropout = nn.Dropout(enc_emb_dropout)
        self.encoder = Encoder(enc_embed_dim,
                               enc_depths,
                               enc_num_heads,
                               enc_ffn_dim,
                               enc_tfm_dropout
                               )
        self.f_linear1 = nn.Linear(enc_embed_dim, enc_embed_dim)
        self.task1 = MaskReconstructionTask(enc_embed_dim, vocab_size)

        # decoder
        self.dec_dropout = nn.Dropout(dec_emb_dropout)
        self.decoder = Decoder(dec_embed_dim,
                               dec_depths,
                               dec_num_heads,
                               dec_ffn_dim,
                               dec_tfm_dropout
                               )
        self.f_linear2 = nn.Linear(dec_embed_dim, dec_embed_dim)
        self.task2 = MaskReconstructionTask(dec_embed_dim, vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, node_fea, edge_index, enc_data, dec_data, lambda2=0.5):
        key_traj_x, key_temporal_x, key_user_id_x, key_highway_x, key_temporal_mat_x, key_dis_mat_x, key_mask, key_idx_x, end_idx, key_idx_without_end = enc_data
        mask_traj_x, full_temporal_x, full_user_id_x, full_highway_x, full_temporal_mat_x, full_dis_mat_x, full_padding_mask, mask_idx_x = dec_data

        spe = self.spe(node_fea, edge_index)
        ce = self.ce(key_highway_x)
        key_fusion = F.relu(self.f_linear1(torch.cat([key_temporal_x, ce], dim=-1)))
        ue = self.ue(key_user_id_x)
        x = spe[key_traj_x] + ue + key_fusion
        x += self.pe(x)

        x = self.enc_dropout(x)
        x = self.encoder(
            x=x,
            mask=key_mask,
            temporal_mat=key_temporal_mat_x,
            dis_mat=key_dis_mat_x,
            lambda2=lambda2
        )

        x = torch.gather(x, dim=1, index=key_idx_without_end.unsqueeze(-1).repeat(1, 1, x.shape[-1]))

        pred1 = self.task1(x)

        full_idx = torch.cat([key_idx_x, mask_idx_x], dim=1)
        idx_restore = torch.argsort(full_idx, dim=1)

        special_emb = spe[:2]
        mask_emb = special_emb[mask_traj_x]
        x = torch.cat([x, mask_emb], dim=1)
        x = torch.gather(x, dim=1, index=idx_restore.unsqueeze(-1).repeat(1, 1, x.shape[-1]))
        full_fusion = F.relu(self.f_linear2(torch.cat([full_temporal_x, self.ce(full_highway_x)], dim=-1)))

        x = x + self.ue(full_user_id_x) + full_fusion
        x += self.pe(x)
        x = self.dec_dropout(x)

        x = self.decoder(
            x=x,
            mask=full_padding_mask,
            temporal_mat=full_temporal_mat_x,
            dis_mat=full_dis_mat_x,
            lambda2=lambda2
        )

        pred2 = self.task2(x)

        return pred1, pred2


class MaskReconstructionTask(nn.Module):
    def __init__(self, emb_size, vocab_size):
        super().__init__()
        self.linear = nn.Linear(emb_size, vocab_size, bias=True)
        self.logsoftmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.logsoftmax(self.linear(x))
