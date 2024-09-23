import argparse


def get_config():
    parser = argparse.ArgumentParser(description='Model HyperParams')
    parser.add_argument('--seed', default=2023, type=int)
    parser.add_argument('--exp_id', default='test', type=str)
    parser.add_argument('--model', default='mae', type=str)
    parser.add_argument('--early_stop', default=5, type=int)

    # dataset
    parser.add_argument('--dataset', default='porto', type=str)
    parser.add_argument('--num_workers', default=8, type=int)

    parser.add_argument('--max_len', default=256, type=int)
    parser.add_argument('--use_mask', default=True, type=bool)
    parser.add_argument('--use_start', default=True, type=bool)
    parser.add_argument('--use_extract', default=True, type=bool)
    parser.add_argument('--use_sep', default=True, type=bool)
    parser.add_argument('--pre_len', default=0, type=int)

    # optimize
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--clip', default=1.0, type=float)
    parser.add_argument('--warmup_steps', default=5, type=int)
    parser.add_argument('--betas', default='(0.9, 0.95)', type=str)
    parser.add_argument('--weight_decay', default=0.05, type=float)

    # graph
    parser.add_argument('--g_depths', default=3, type=int)
    parser.add_argument('--g_heads_per_layer', default='[8, 16, 1]', type=str)
    parser.add_argument('--g_dim_per_layer', default='[16, 16, 128]', type=str)
    parser.add_argument('--g_dropout', default=0.1, type=float)

    # encoder
    parser.add_argument('--enc_embed_dim', default=128, type=int)
    parser.add_argument('--enc_ffn_dim', default=128 * 4, type=int)
    parser.add_argument('--enc_depths', default=6, type=int)
    parser.add_argument('--enc_num_heads', default=8, type=int)
    parser.add_argument('--enc_emb_dropout', default=0.1, type=float)
    parser.add_argument('--enc_tfm_dropout', default=0.1, type=float)

    # decoder
    parser.add_argument('--dec_embed_dim', default=128, type=int)
    parser.add_argument('--dec_ffn_dim', default=128 * 4, type=int)
    parser.add_argument('--dec_depths', default=6, type=int)
    parser.add_argument('--dec_num_heads', default=8, type=int)
    parser.add_argument('--dec_emb_dropout', default=0.1, type=float)
    parser.add_argument('--dec_tfm_dropout', default=0.1, type=float)

    parser.add_argument('--lambda1', default=0.1, type=float)
    parser.add_argument('--lambda2', default=0.5, type=float)

    # device
    parser.add_argument('--device', default='cuda:2', type=str)
    args = parser.parse_args()
    config = vars(args)
    return config
