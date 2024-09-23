import torch
from torch.utils.data import DataLoader, Dataset


class TrajDataSet(Dataset):
    def __init__(self, data, vflag):
        super().__init__()
        self.data = data
        self.vflag = vflag

    def __getitem__(self, index):
        return self.data[index], self.vflag[index]

    def __len__(self):
        return len(self.data)


def padding_mask_fn(lengths, max_len=None):
    batch_size = lengths.numel()
    max_len = max_len or lengths.max_val()
    return (torch.arange(0, max_len)
            .type_as(lengths)
            .repeat(batch_size, 1)
            .lt(lengths.unsqueeze(1)))


class BinaryClsTrajDataLoader:
    def __init__(self, config):
        self.pre_len = config['pre_len']
        self.model = config['model']
        self.bz = config['batch_size']
        self.max_len = config['max_len']
        self.num_workers = config['num_workers']

    def _collate_fn(self, batch_data, vocab, type, pre_len=0):
        # 和simple是一样，只是输出取的embedding位置不一样
        bz = len(batch_data)
        vflag = [d[1] for d in batch_data]
        batch_data = [d[0] for d in batch_data]
        full_data = [data[0] for data in batch_data]

        user_id, mask_traj, traj, temporal, temporal_vec, temporal_mat, dis_mat, highway = zip(*full_data)

        traj_len = [len(t) for t in traj]
        max_traj_len = max(traj_len) + 2

        # tensor data
        traj_x = torch.zeros(size=(bz, max_traj_len), dtype=torch.long)
        highway_x = torch.zeros_like(traj_x, dtype=torch.long)
        user_id_x = torch.zeros_like(traj_x, dtype=torch.long)

        temporal_x = torch.zeros(size=(bz, max_traj_len, 64))
        temporal_mat_x = torch.zeros(size=(bz, max_traj_len, max_traj_len))
        dis_mat_x = torch.zeros(size=(bz, max_traj_len, max_traj_len))

        for i in range(bz):
            # encoder
            end = traj_len[i]
            traj_x[i, 1:end + 1] = torch.tensor(traj[i], dtype=torch.long)
            traj_x[i, 0] = vocab.start_index
            traj_x[i, end + 1] = vocab.extract_index

            user_id_x[i, 1:end + 1] = torch.tensor(user_id[i], dtype=torch.long)

            highway_x[i, 1:end + 1] = torch.tensor(highway[i], dtype=torch.long)

            temporal_x[i, 1:end + 1] = torch.tensor(temporal_vec[i])

            temporal_mat_x[i, 1:end + 1, 1:end + 1] = torch.tensor(temporal_mat[i], dtype=float)
            dis_mat_x[i, 1:end + 1, 1:end + 1] = torch.tensor(dis_mat[i], dtype=float)

        # padding数据
        traj_len = [tl + 2 for tl in traj_len]
        padding_mask = padding_mask_fn(torch.tensor(traj_len, dtype=torch.int16), max_len=max_traj_len)
        end_idx = torch.sum(padding_mask, dim=1) - 1
        idx_without_end = [[k for k in range(max_traj_len) if k != end_idx[i]] for i in range(bz)]
        idx_without_end = torch.tensor(idx_without_end, dtype=torch.long)

        padding_mask = ~padding_mask
        padding_mask = padding_mask.unsqueeze(1).repeat(1, padding_mask.size(1), 1).unsqueeze(1)
        future_mask = torch.triu(torch.ones((1, max_traj_len, max_traj_len)), diagonal=1).bool()
        mask = padding_mask | future_mask

        id_y = torch.tensor(vflag, dtype=torch.long)

        data = [traj_x, temporal_x, user_id_x, mask, end_idx, idx_without_end, temporal_mat_x, dis_mat_x, highway_x]

        return data, id_y

    def get_dataloader(self, data, vflag, vocab, type):
        dataset = TrajDataSet(data=data, vflag=vflag)
        dataloader = DataLoader(dataset, batch_size=self.bz, num_workers=self.num_workers, shuffle=False,
                                collate_fn=lambda x: self._collate_fn(x, vocab, type))
        return dataloader
