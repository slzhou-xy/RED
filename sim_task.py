import os
import random
import warnings
import torch
import numpy as np
from model.traj_sim import TrajSim
from dataset.vocab import WordVocab
from preprocess import TrajPreprocess
from dataset.sim_dataloader import SimTrajDataLoader
from trainer.sim_trainer import SimTrainer
from config.config import get_config

warnings.filterwarnings("ignore")

config = get_config()

data_path = './data'
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

vocab = WordVocab.load_vocab(vocab_path)

traj_preprocess = TrajPreprocess(config=config, vocab=vocab)
train_data, eval_data, test_data = traj_preprocess.data_split()
node_feature = traj_preprocess.get_initial_feature()
edge_index = traj_preprocess.get_graph()


config['epochs'] = 30
config['device'] = 'cuda:0'
config['vocab_size'] = vocab.vocab_size
config['user_size'] = vocab.user_num
config['fea_size'] = node_feature.shape[1]
config['highway_size'] = traj_preprocess.edge['highway_type'].nunique()


traj_dataloader = SimTrajDataLoader(config)
train_dataloader = traj_dataloader.get_dataloader(train_data, vocab)
eval_dataloader = traj_dataloader.get_dataloader(eval_data, vocab)
test_dataloader = traj_dataloader.get_dataloader(test_data, vocab)

if config['dataset'] == 'rome':
    pretraining_model_path = os.path.join('checkpoints', data_name, config['exp_id'], 'pretraining', 'pretraining_30.pt')
elif config['dataset'] == 'cd':
    pretraining_model_path = os.path.join('checkpoints', data_name, config['exp_id'], 'pretraining', 'pretraining_20.pt')
elif config['dataset'] == 'big_cd':
    pretraining_model_path = os.path.join('checkpoints', data_name, config['exp_id'], 'pretraining', 'pretraining_5.pt')
elif config['dataset'] == 'porto':
    pretraining_model_path = os.path.join('checkpoints', data_name, config['exp_id'], 'pretraining', 'pretraining_10.pt')
else:
    raise NotImplementedError

model = TrajSim(config, pretraining_model_path).to(config['device'])
trainer = SimTrainer(
    config=config,
    model=model,
    node_feature=node_feature,
    edge_index=edge_index,
    train_dataloader=train_dataloader,
    eval_dataloader=eval_dataloader,
    test_dataloader=test_dataloader,
)

trainer.test()
