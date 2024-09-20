import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
from datetime import datetime


class TrajDataSet(Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


def padding_mask_fn(lengths, max_len=None):
    batch_size = lengths.numel()
    max_len = max_len or lengths.max_val()
    return (torch.arange(0, max_len)
            .type_as(lengths)
            .repeat(batch_size, 1)
            .lt(lengths.unsqueeze(1)))


class ETADataLoader:
    def __init__(self, config):
        self.pre_len = config['pre_len']
        self.model = config['model']
        self.bz = config['batch_size']
        self.max_len = config['max_len']
        self.num_workers = config['num_workers']

    def _collate_fn(self, batch_data, vocab, type, pre_len=0):
        bz = len(batch_data)
        full_data = [data[0] for data in batch_data]

        user_id, mask_traj, traj, temporal, temporal_vec, temporal_mat, dis_mat, highway = zip(*full_data)

        traj_len = [len(t) for t in traj]
        max_traj_len = max(traj_len) + 2

        traj_x = torch.zeros(size=(bz, max_traj_len), dtype=torch.long)
        highway_x = torch.zeros_like(traj_x, dtype=torch.long)
        user_id_x = torch.zeros_like(traj_x, dtype=torch.long)

        y = torch.zeros(size=(bz, max_traj_len - 1), dtype=torch.long)

        temporal_x = torch.zeros(size=(bz, max_traj_len, 64))
        temporal_mat_x = torch.zeros(size=(bz, max_traj_len, max_traj_len))
        dis_mat_x = torch.zeros(size=(bz, max_traj_len, max_traj_len))

        eta = [(datetime.fromtimestamp(tlist[-1]) - datetime.fromtimestamp(tlist[0])).seconds / 60
               for tlist in temporal]

        for i in range(bz):
            # encoder
            end = traj_len[i]
            traj_x[i, 1:end + 1] = torch.tensor(traj[i], dtype=torch.long)
            traj_x[i, 0] = vocab.start_index
            traj_x[i, end + 1] = vocab.extract_index

            user_id_x[i, 1:end + 1] = torch.tensor(user_id[i], dtype=torch.long)
            temporal_x[i, 1] = torch.tensor(temporal_vec[i][0])
            highway_x[i, 1:end + 1] = torch.tensor(highway[i], dtype=torch.long)

            y[i, :end] = torch.tensor(traj[i], dtype=torch.long)
            y[i, end] = vocab.sep_index

            dis_mat_x[i, 1:end + 1, 1:end + 1] = torch.tensor(dis_mat[i], dtype=float)

        traj_len = [tl + 2 for tl in traj_len]
        key_padding_mask = padding_mask_fn(torch.tensor(traj_len, dtype=torch.int16), max_len=max_traj_len)
        end_idx = torch.sum(key_padding_mask, dim=1) - 1

        key_idx_without_end = [[k for k in range(max_traj_len) if k != end_idx[i]] for i in range(bz)]
        key_idx_without_end = torch.tensor(key_idx_without_end, dtype=torch.long)

        key_padding_mask = ~key_padding_mask
        key_padding_mask = key_padding_mask.unsqueeze(1).repeat(1, key_padding_mask.size(1), 1).unsqueeze(1)
        key_future_mask = torch.triu(torch.ones((1, max_traj_len, max_traj_len)), diagonal=1).bool()
        key_mask = key_padding_mask[:] | key_future_mask

        eta = torch.tensor(eta, dtype=torch.float32).unsqueeze(-1)

        data = [traj_x, temporal_x, user_id_x, key_mask, end_idx, key_idx_without_end, temporal_mat_x, dis_mat_x, highway_x]

        return data, eta, y

    def get_dataloader(self, data, vocab, type):
        dataset = TrajDataSet(data=data)
        dataloader = DataLoader(dataset, batch_size=self.bz, num_workers=self.num_workers, shuffle=False,
                                collate_fn=lambda x: self._collate_fn(x, vocab, type))
        return dataloader
