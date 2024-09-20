import argparse
import os
from dataset.vocab import WordVocab

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='porto', help='the name of dataset')
parser.add_argument('--use_mask', type=bool, default=True, help='Whether to use mask or not in vocab')
args = parser.parse_args()

use_mask = args.use_mask
data_name = args.dataset

roadnetwork_path = 'data/{}/rn/edge.csv'.format(data_name)
traj_path = 'data/{}/traj/traj.csv'.format(data_name, data_name)
vocab_path = 'vocab.pkl'.format(data_name)


if not os.path.exists(vocab_path):
    vocab = WordVocab(traj_path=traj_path, roadnetwork_path=roadnetwork_path, use_mask=use_mask)
    vocab.save_vocab(vocab_path)
    print("VOCAB SIZE ", len(vocab))
else:
    vocab = WordVocab.load_vocab(vocab_path)
print('user num ', vocab.user_num)
print("vocab size ", vocab.vocab_size)
print(vocab.to_seq([1, 2, 3, 4, 5, 6]))
print(vocab.from_seq(vocab.to_seq([1, 2, 3, 4, 5, 6])))
