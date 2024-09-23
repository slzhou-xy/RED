import torch
from torch.utils.data import DataLoader, Dataset


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


class TrajDataLoader:
    def __init__(self, config):
        self.bz = config['batch_size']
        self.pre_len = config['pre_len']
        self.max_len = config['max_len']
        self.num_workers = config['num_workers']

    def _collate_fn(self, batch_data, vocab, pre_len=0):
        bz = len(batch_data)
        full_data = [data[0] for data in batch_data]
        key_data = [data[1] for data in batch_data]
        full_user_id, full_mask_traj, full_traj, _, full_temporal_vec, full_temporal_mat, full_dis_mat, full_highway = zip(*full_data)
        key_user_id, key_traj, _, key_temporal_vec, key_temporal_mat, key_dis_mat, key_highway = zip(*key_data)

        key_idx = [[k for k, segment in enumerate(full_mask_traj[i]) if segment != -1] for i in range(bz)]
        # 0123456789exxxxxx
        # s0123456789exxxxx
        key_idx = [idx + [idx[-1] + 1] for idx in key_idx]
        mask_idx = [[k for k, segment in enumerate(full_mask_traj[i]) if segment == -1] for i in range(bz)]

        key_len = [len(idx) for idx in key_idx]
        max_key_len = max(key_len)

        mask_len = [len(idx) for idx in mask_idx]
        max_mask_len = max(mask_len)

        start_padding_idx = [len(traj) + 1 for traj in full_traj]
        for i in range(bz):
            key_padding_len = max_key_len - len(key_idx[i])
            key_padding_start_idx = start_padding_idx[i]
            key_idx[i] += [key_padding_start_idx + k for k in range(key_padding_len)]

            mask_padding_len = max_mask_len - len(mask_idx[i])
            mask_padding_start_idx = key_padding_start_idx + key_padding_len
            mask_idx[i] += [mask_padding_start_idx + k for k in range(mask_padding_len)]

        max_len = max_key_len + max_mask_len
        # 012345678e
        # s012345678e
        # encoder input, only key data
        key_traj_x = torch.zeros(size=(bz, max_key_len + 1), dtype=torch.long)
        key_highway_x = torch.zeros_like(key_traj_x, dtype=torch.long)
        key_temporal_x = torch.zeros(size=(bz, max_key_len + 1, 64), dtype=torch.float)
        key_user_id_x = torch.zeros_like(key_traj_x, dtype=torch.long)
        key_idx_x = torch.zeros(size=(bz, max_key_len), dtype=torch.long)
        key_temporal_mat_x = torch.zeros(size=(bz, max_key_len + 1, max_key_len + 1), dtype=torch.float32)
        key_dis_mat_x = torch.zeros(size=(bz, max_key_len + 1, max_key_len + 1), dtype=torch.float32)
        y1 = torch.zeros_like(key_idx_x, dtype=torch.long)

        # decoder input, only mask data
        mask_traj_x = torch.zeros(size=(bz, max_mask_len), dtype=torch.long)

        full_highway_x = torch.zeros(size=(bz, max_len), dtype=torch.long)
        full_temporal_x = torch.zeros(size=(bz, max_len, 64), dtype=torch.float)
        full_temporal_mat_x = torch.zeros(size=(bz, max_len, max_len), dtype=torch.float32)
        full_dis_mat_x = torch.zeros(size=(bz, max_len, max_len), dtype=torch.float32)
        full_user_id_x = torch.zeros(size=(bz, max_len), dtype=torch.long)
        mask_idx_x = torch.zeros_like(mask_traj_x, dtype=torch.long)
        y2 = torch.zeros(size=(bz, max_len), dtype=torch.long)  # decoder label，注意长度

        for i in range(bz):
            # encoder
            cur_key_end = key_len[i]
            key_traj_x[i, 1:cur_key_end] = torch.tensor(key_traj[i], dtype=torch.long)
            key_traj_x[i, 0] = vocab.start_index
            key_traj_x[i, cur_key_end] = vocab.extract_index
            key_temporal_mat_x[i, 1:cur_key_end, 1:cur_key_end] = torch.tensor(key_temporal_mat[i], dtype=float)
            key_dis_mat_x[i, 1:cur_key_end, 1:cur_key_end] = torch.tensor(key_dis_mat[i], dtype=float)

            key_idx_x[i] = torch.tensor(key_idx[i], dtype=torch.long)
            key_highway_x[i, 1:cur_key_end] = torch.tensor(key_highway[i], dtype=torch.long)

            key_temporal_x[i, 1:cur_key_end] = torch.tensor(key_temporal_vec[i])
            key_user_id_x[i, 1:cur_key_end] = torch.tensor(key_user_id[i], dtype=torch.long)

            y1[i, pre_len:cur_key_end - 1] = torch.tensor(key_traj[i][pre_len:], dtype=torch.long)
            y1[i, cur_key_end - 1] = vocab.sep_index 

            # decoder
            cur_mask_end = mask_len[i]
            mask_traj_x[i, :cur_mask_end] = torch.tensor([vocab.mask_index] * cur_mask_end, dtype=torch.long)
            mask_idx_x[i] = torch.tensor(mask_idx[i], dtype=torch.long)

            full_end = len(full_traj[i])

            full_highway_x[i, :full_end] = torch.tensor(full_highway[i], dtype=torch.long)
            full_temporal_x[i, :full_end] = torch.tensor(full_temporal_vec[i])
            full_temporal_mat_x[i, :full_end, :full_end] = torch.tensor(full_temporal_mat[i], dtype=float)
            full_dis_mat_x[i, :full_end, :full_end] = torch.tensor(full_dis_mat[i], dtype=float)

            full_user_id_x[i, :full_end] = torch.tensor(full_user_id[i], dtype=torch.long)

            y2[i, :full_end] = torch.tensor(full_traj[i], dtype=torch.long)

        key_len = [kl + 1 for kl in key_len]
        key_padding_mask = padding_mask_fn(torch.tensor(key_len, dtype=torch.int16), max_len=max_key_len + 1)
        end_idx = torch.sum(key_padding_mask, dim=1) - 1
        key_idx_without_end = [[k for k in range(max_key_len + 1) if k != end_idx[i]] for i in range(bz)]
        key_idx_without_end = torch.tensor(key_idx_without_end, dtype=torch.long)
        key_padding_mask = ~key_padding_mask
        key_padding_mask = key_padding_mask.unsqueeze(1).repeat(1, key_padding_mask.size(1), 1).unsqueeze(1)
        key_future_mask = torch.triu(torch.ones((1, max_key_len + 1, max_key_len + 1)), diagonal=1).bool()
        key_mask = key_padding_mask | key_future_mask

        full_len = [len1 + len2 - 1 for len1, len2 in zip(key_len, mask_len)]
        full_padding_mask = padding_mask_fn(torch.tensor(full_len, dtype=torch.int16), max_len=max_len)
        full_padding_mask = ~full_padding_mask
        full_padding_mask = full_padding_mask.unsqueeze(1).repeat(1, full_padding_mask.size(1), 1).unsqueeze(1)

        enc_data = [key_traj_x, key_temporal_x, key_user_id_x, key_highway_x, key_temporal_mat_x, key_dis_mat_x,
                    key_mask, key_idx_x, end_idx, key_idx_without_end]
        dec_data = [mask_traj_x, full_temporal_x, full_user_id_x, full_highway_x, full_temporal_mat_x, full_dis_mat_x, full_padding_mask, mask_idx_x]

        return enc_data, dec_data, y1, y2

    def get_dataloader(self, data, vocab):
        dataset = TrajDataSet(data=data)
        # shuffle is False, because we have shuffled the data in the preprocessing.
        dataloader = DataLoader(dataset, batch_size=self.bz, shuffle=False,
                                collate_fn=lambda x: self._collate_fn(x, vocab, self.pre_len))
        return dataloader
