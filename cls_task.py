import os
import warnings
import pandas as pd
import random
import torch
import numpy as np
from model.traj_cls import TrajCls
from dataset.vocab import WordVocab
from preprocess import TrajPreprocess
from dataset.cls_dataloader import ClsTrajDataLoader
from dataset.binary_cls_dataloader import BinaryClsTrajDataLoader
from trainer.cls_trainer import ClsTrainer
from trainer.binary_cls_trainer import BinaryClsTrainer
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
config['clip'] = 1.0
config['device'] = 'cuda:1'
config['lr'] = 1e-4
config['vocab_size'] = vocab.vocab_size
config['user_size'] = vocab.user_num
config['highway_size'] = traj_preprocess.edge['highway_type'].nunique()
config['fea_size'] = node_feature.shape[1]
config['batch_size'] = 64


if data_name == 'cd' or data_name == 'big_cd':
    dataset = pd.read_csv(f'data/{data_name}/traj/traj.csv')
    train_index = np.load(f'data/{data_name}/train_index.npy')
    eval_index = np.load(f'data/{data_name}/eval_index.npy')
    test_index = np.load(f'data/{data_name}/test_index.npy')
    train_label = list(dataset.iloc[train_index].flag)
    eval_label = list(dataset.iloc[eval_index].flag)
    test_label = list(dataset.iloc[test_index].flag)
    traj_dataloader = BinaryClsTrajDataLoader(config)
    train_dataloader = traj_dataloader.get_dataloader(train_data, train_label, vocab, 'train')
    eval_dataloader = traj_dataloader.get_dataloader(eval_data, eval_label, vocab, 'eval')
    test_dataloader = traj_dataloader.get_dataloader(test_data, test_label, vocab, 'test')
else:
    traj_dataloader = ClsTrajDataLoader(config)
    train_dataloader = traj_dataloader.get_dataloader(train_data, vocab, 'train')
    eval_dataloader = traj_dataloader.get_dataloader(eval_data, vocab, 'eval')
    test_dataloader = traj_dataloader.get_dataloader(test_data, vocab, 'test')

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

model = TrajCls(config, pretraining_model_path).to(config['device'])

if config['dataset'] == 'cd' or config['dataset'] == 'big_cd':
    trainer = BinaryClsTrainer(
        config=config,
        model=model,
        node_feature=node_feature,
        edge_index=edge_index,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        test_dataloader=test_dataloader,
    )
else:
    trainer = ClsTrainer(
        config=config,
        model=model,
        node_feature=node_feature,
        edge_index=edge_index,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        test_dataloader=test_dataloader,
    )

trainer.train()
