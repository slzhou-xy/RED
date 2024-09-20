import os
import warnings
import random
import torch
import numpy as np
from trainer.trainer import REDTrainer
from model.red import RED
from dataset.vocab import WordVocab
from preprocess import TrajPreprocess
from dataset.dataloader import TrajDataLoader
from config.config import get_config

warnings.filterwarnings("ignore")

config = get_config()

data_path = '/home/zhousilin/Code/zhousilin/RED-vldb/TrajModel_final/data'
config['data_path'] = data_path

data_name = config['dataset']

roadnetwork_path = f'{data_path}/{data_name}/rn/edge.csv'
traj_path = f'{data_path}/{data_name}/traj/traj.csv'
vocab_path = f'{data_path}/{data_name}/vocab.pkl'

# fix seed
seed = config['seed']
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

# construct vocab
if not os.path.exists(vocab_path):
    vocab = WordVocab(traj_path=traj_path,
                      roadnetwork_path=roadnetwork_path,
                      use_mask=config['use_mask'],
                      use_sep=config['use_sep'],
                      use_start=config['use_start'],
                      use_extract=config['use_extract']
                      )
    vocab.save_vocab(vocab_path)
else:
    vocab = WordVocab.load_vocab(vocab_path)

traj_preprocess = TrajPreprocess(config=config, vocab=vocab)
train_data, eval_data, test_data = traj_preprocess.data_split()
node_feature = traj_preprocess.get_initial_feature()
edge_index = traj_preprocess.get_graph()

config['vocab_size'] = vocab.vocab_size
config['user_size'] = vocab.user_num
config['highway_size'] = traj_preprocess.edge['highway_type'].nunique() + 1
config['fea_size'] = node_feature.shape[1]

traj_dataloader = TrajDataLoader(config)
train_dataloader = traj_dataloader.get_dataloader(train_data, vocab)
eval_dataloader = traj_dataloader.get_dataloader(eval_data, vocab)
test_dataloader = traj_dataloader.get_dataloader(test_data, vocab)

model = RED(
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
trainer = REDTrainer(
    config=config,
    model=model,
    node_feature=node_feature,
    edge_index=edge_index,
    train_dataloader=train_dataloader
)

print("Total number of paramerters in networks is {}  ".format(sum(x.numel() for x in model.parameters())))

trainer.train()
