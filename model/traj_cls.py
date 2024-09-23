import torch
import torch.nn as nn
import torch.nn.functional as F
from model.red import RED


class TrajCls(nn.Module):
    def __init__(self, config, path):
        super().__init__()
        self.pretraining_model = RED(
            fea_size=config['fea_size'],
            g_heads_per_layer=config['g_heads_per_layer'],
            g_dim_per_layer=config['g_dim_per_layer'],
            g_depths=config['g_depths'],
            g_dropout=config['g_dropout'],
            enc_embed_dim=config['enc_embed_dim'],
            enc_ffn_dim=config['enc_ffn_dim'],
            enc_depths=config['enc_depths'],
            enc_num_heads=config['enc_num_heads'],
            enc_emb_dropout=config['enc_emb_dropout'],
            enc_tfm_dropout=config['enc_tfm_dropout'],
            dec_embed_dim=config['dec_embed_dim'],
            dec_ffn_dim=config['dec_ffn_dim'],
            dec_depths=config['dec_depths'],
            dec_num_heads=config['dec_num_heads'],
            dec_emb_dropout=config['dec_emb_dropout'],
            dec_tfm_dropout=config['dec_tfm_dropout'],
            vocab_size=config['vocab_size'],
            user_size=config['user_size']
        )
        self.pretraining_model.load_state_dict(torch.load(path))
        if config['dataset'] == 'cd' or config['dataset'] == 'big_cd':
            self.pre_linear = nn.Linear(config['dec_embed_dim'], 2)
        else:
            self.pre_linear = nn.Linear(config['dec_embed_dim'], config['user_size'])
        # torch.nn.init.xavier_uniform_(self.pre_linear.weight)
        # torch.nn.init.constant_(self.pre_linear.bias, 0)

    def forward(self, node_feature, edge_index, enc_data, lambda2):
        traj_x, temporal_x, user_id_x, mask, end_idx, idx_without_end, temporal_mat_x, dis_mat_x, highway_x = enc_data
        spe = self.pretraining_model.spe(node_feature, edge_index)
        ce = self.pretraining_model.ce(highway_x)
        ue = self.pretraining_model.ue(user_id_x)

        x = spe[traj_x] + ue + F.relu(self.pretraining_model.f_linear1(torch.cat([temporal_x, ce], dim=-1)))

        x += self.pretraining_model.pe(x)

        x = self.pretraining_model.enc_dropout(x)

        x = self.pretraining_model.encoder(
            x=x,
            mask=mask,
            temporal_mat=temporal_mat_x,
            dis_mat=dis_mat_x,
            lambda2=lambda2
        )

        traj_emb = torch.gather(x, dim=1, index=end_idx.view(-1, 1, 1).repeat(1, 1, x.shape[-1])).squeeze(1)

        return self.pre_linear(traj_emb)
