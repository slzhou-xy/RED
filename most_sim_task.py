import os
import random
import torch
import numpy as np
from model.traj_sim import TrajSim
from dataset.vocab import WordVocab
from preprocess import TrajPreprocess
from dataset.most_sim_dataloader import MostSimDataLoader
from trainer.most_sim_trainer import MostSimTrainer
from config.config import get_config

import pickle

config = get_config()

data_name = config['dataset']
roadnetwork_path = 'data/{}/rn/edge.csv'.format(data_name)
traj_path = 'data/{}/traj/traj.csv'.format(data_name)
vocab_path = 'data/{}/vocab.pkl'.format(data_name)

# fix seed
seed = config['seed']
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

vocab = WordVocab.load_vocab(vocab_path)
traj_preprocess = TrajPreprocess(config=config, vocab=vocab)
node_feature = traj_preprocess.get_initial_feature()
edge_index = traj_preprocess.get_graph()

config['epochs'] = 30
config['device'] = 'cuda:0'
config['vocab_size'] = vocab.vocab_size
config['user_size'] = vocab.user_num
config['highway_size'] = traj_preprocess.edge['highway_type'].nunique() + 1
config['fea_size'] = node_feature.shape[1]


traj_dataloader = MostSimDataLoader(config)
query = pickle.load(open(f'data/{data_name}/traj/query.pkl', 'rb'))
db = pickle.load(open(f'data/{data_name}/traj/database.pkl', 'rb'))
q_dataloader = traj_dataloader.get_dataloader(query, vocab)
db_dataloader = traj_dataloader.get_dataloader(db, vocab)


if config['dataset'] == 'rome':
    pretraining_model_path = os.path.join('checkpoints', config['exp_id'], 'pretraining', 'pretraining_30.pt')
elif config['dataset'] == 'cd':
    pretraining_model_path = os.path.join('checkpoints', config['exp_id'], 'pretraining', 'pretraining_20.pt')
elif config['dataset'] == 'big_cd':
    pretraining_model_path = os.path.join('checkpoints', config['exp_id'], 'pretraining', 'pretraining_5.pt')
elif config['dataset'] == 'porto':
    pretraining_model_path = os.path.join('checkpoints', config['exp_id'], 'pretraining', 'pretraining_10.pt')
else:
    raise NotImplementedError

model = TrajSim(config, pretraining_model_path).to(config['device'])
trainer = MostSimTrainer(
    config=config,
    model=model,
    node_feature=node_feature,
    edge_index=edge_index,
    q_dataloader=q_dataloader,
    db_dataloader=db_dataloader,
)

trainer.test()
