import os
import pickle
import numpy as np
import pandas as pd
import scipy.sparse as sp
from tqdm import tqdm
from date2vec import Date2vec
from utils.utils_fun import add_dis


class TrajPreprocess:
    def __init__(self, config, vocab):
        self.vocab = vocab

        self.dataset = config['dataset']

        self.path = os.path.join(config['data_path'], config['dataset'])

        self.edge_rn_path = os.path.join(self.path, 'rn/edge_rn.csv')
        self.edge_path = os.path.join(self.path, 'rn/edge.csv')
        self.node_path = os.path.join(self.path, 'rn/node.csv')
        self.traj_path = os.path.join(self.path, 'traj/traj.csv')

        print('Load dataset...')
        self.edge_rn = pd.read_csv(self.edge_rn_path)
        self.edge = pd.read_csv(self.edge_path)
        self.node = pd.read_csv(self.node_path)
        self.d2v = Date2vec()
        print('Load dataset finish!')

    def _normalization(self, data):
        return (data - np.min(data)) / (np.max(data) - np.min(data))

    def _cal_mat(self, tim_list):
        seq_len = len(tim_list)
        mat = np.zeros((seq_len, seq_len))
        for i in range(seq_len):
            for j in range(seq_len):
                off = abs(tim_list[i] - tim_list[j])
                mat[i][j] = off / 60
        return mat

    def _cal_dis_mat(self, segment_list):
        seq_len = len(segment_list)
        mat = np.zeros((seq_len, seq_len))
        for i in range(seq_len):
            for j in range(seq_len):
                off = abs(segment_list[i] - segment_list[j])
                mat[i][j] = off / 1000
        return mat

    def get_initial_feature(self):
        print('Load initial feature...')
        feature_path = os.path.join(self.path, 'node_feature.npy')
        if os.path.exists(feature_path):
            return np.load(file=feature_path)

        speed = self.edge['length'].fillna(0)
        speed = self._normalization(speed.to_numpy())

        traval_time = self.edge['traval_time'].fillna(0)
        traval_time = self._normalization(traval_time.to_numpy())

        bearing = self.edge['bearing'].fillna(0)
        bearing = self._normalization(bearing.to_numpy())

        out_degree = self.edge['out_degree'].to_numpy()
        in_degree = self.edge['in_degree'].to_numpy()
        highway_type = pd.get_dummies(self.edge['highway_type']).to_numpy()
        feature = np.concatenate([speed[:, np.newaxis], traval_time[:, np.newaxis],
                                  bearing[:, np.newaxis], out_degree[:, np.newaxis],
                                  in_degree[:, np.newaxis], highway_type], axis=1)
        # 3 for sep, start, and end, but sep is not used in the model
        special = np.zeros((3, feature.shape[1]))
        feature = np.vstack([special, feature])
        np.save(feature_path, feature)
        print('Load initial feature finish!')
        return feature

    def data_split(self):
        print('Trajectory dataset split...')
        train_traj_path = os.path.join(self.path, 'traj/traj_train.pkl')
        eval_traj_path = os.path.join(self.path, 'traj/traj_eval.pkl')
        test_traj_path = os.path.join(self.path, 'traj/traj_test.pkl')
        if os.path.exists(eval_traj_path):
            train_traj = pickle.load(open(train_traj_path, 'rb'))
            eval_traj = pickle.load(open(eval_traj_path, 'rb'))
            test_traj = pickle.load(open(test_traj_path, 'rb'))
            return train_traj, eval_traj, test_traj

        if os.path.exists(f'{self.path}/train_index.npy'):
            rand_index = np.arange(self.traj.shape[0])
            rand_index = np.random.permutation(rand_index)
            if self.dataset == 'rome':
                train_index = rand_index[:int(rand_index.shape[0] * 0.8)]
                eval_index = rand_index[int(rand_index.shape[0] * 0.8): int(rand_index.shape[0] * 0.9)]
                test_index = rand_index[int(rand_index.shape[0] * 0.9):]
            else:
                train_index = rand_index[:int(rand_index.shape[0] * 0.6)]
                eval_index = rand_index[int(rand_index.shape[0] * 0.6): int(rand_index.shape[0] * 0.8)]
                test_index = rand_index[int(rand_index.shape[0] * 0.8):]

            np.save(f'{self.path}/train_index.npy', train_index)
            np.save(f'{self.path}/eval_index.npy', eval_index)
            np.save(f'{self.path}/test_index.npy', test_index)
        else:
            train_index = np.load(f'{self.path}/train_index.npy')
            eval_index = np.load(f'{self.path}/eval_index.npy')
            test_index = np.load(f'{self.path}/test_index.npy')

        traj = pd.read_csv(self.traj_path)
        if 'full_dis' not in self.traj.columns:
            traj = add_dis(traj, self.edge)
            traj.to_csv(self.traj_path, index=False)
        train_traj = traj.loc[train_index]
        eval_traj = traj.loc[eval_index]
        test_traj = traj.loc[test_index]

        train_traj = self._to_pkl(train_traj, train_traj_path, data_type='train')
        eval_traj = self._to_pkl(eval_traj, eval_traj_path, data_type='eval')
        test_traj = self._to_pkl(test_traj, test_traj_path, data_type='test')
        print('Trajectory dataset split finish!')

        return train_traj, eval_traj, test_traj

    def _to_pkl(self, data, save_path, data_type=None):
        save_data = []
        taxi_ids = data['taxi_id'].to_numpy()

        for i in tqdm(range(data.shape[0]), desc=data_type):
            row = data.iloc[i]
            full_traj = eval(row['cpath'])
            full_taxi_id = [taxi_ids[i] + 1] * len(full_traj)
            full_highway = [self.edge.loc[e, 'highway_type'] + 1 for e in full_traj]

            full_temporal = eval(row['align_time'])
            full_temporal_vec = self.d2v(full_temporal)
            full_temporal_mat = self._cal_mat(full_temporal)

            full_dis = eval(row['full_dis'])
            full_dis_mat = self._cal_dis_mat(full_dis)

            full_traj = [self.vocab.loc2index.get(e) for e in full_traj]

            # full mask
            full_mask_traj = eval(row['mask_traj'])
            full_mask_traj = [self.vocab.loc2index.get(e, -1) for e in full_mask_traj]

            # key
            key_traj = eval(row['key_cpath'])
            key_taxi_id = [taxi_ids[i] + 1] * len(key_traj)
            key_highway = [self.edge.loc[e, 'highway_type'] + 1 for e in key_traj]

            key_dis = eval(row['key_dis'])
            key_dis_mat = self._cal_dis_mat(key_dis)

            key_temporal = eval(row['key_time'])
            key_temporal_vec = self.d2v(key_temporal)
            key_temporal_mat = self._cal_mat(key_temporal)

            key_traj = [self.vocab.loc2index.get(e) for e in key_traj]

            save_data.append([[full_taxi_id, full_mask_traj, full_traj, full_temporal, full_temporal_vec, full_temporal_mat, full_dis_mat, full_highway],
                              [key_taxi_id, key_traj, key_temporal, key_temporal_vec, key_temporal_mat, key_dis_mat, key_highway]])

        pickle.dump(save_data, open(save_path, 'wb'))
        return save_data

    def get_graph(self):
        G_path = os.path.join(self.path, 'G.npy')
        if os.path.exists(G_path):
            return np.load(file=G_path)

        # 2 for mask and padding token
        size = self.vocab.vocab_size - 2
        G = sp.dok_matrix((size, size), dtype=int)
        start_id = self.edge_rn['from_edge_id'].to_numpy()
        end_id = self.edge_rn['to_edge_id'].to_numpy()

        start_id += 3
        end_id += 3

        for i in range(3, size):
            G[1, i] = 1

        for s, e in zip(start_id, end_id):
            G[s, e] = 1

        G = G.tocoo()
        start_id = G.row
        end_id = G.col
        G = np.vstack([start_id, end_id])
        np.save(G_path, G)
        return G
