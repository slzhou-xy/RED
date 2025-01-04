import os
import torch
from loguru import logger

from tqdm import tqdm


class MostSimTrainer:
    def __init__(self,
                 config,
                 model,
                 node_feature,
                 edge_index,
                 q_dataloader,
                 db_dataloader
                 ):
        super().__init__()
        self.lr = config['lr']
        self.device = config['device']
        self.epochs = config['epochs']
        self.clip = config['clip']
        self.dataset = config['dataset']
        self.lambda2 = config['lambda2']

        self.node_feature = node_feature
        self.edge_index = edge_index
        self.q_dataloader = q_dataloader
        self.db_dataloader = db_dataloader

        self.model = model.to(self.device)
        self.save_path = os.path.join('checkpoints', config['dataset'], config['exp_id'], 'most_sim')
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        logger.add(f'{self.save_path}/results.log', mode='w')

    def test(self):
        self.model.eval()
        with torch.no_grad():
            pbar = tqdm(self.db_dataloader)
            db_emb = []
            for batch_data in pbar:
                enc_data = batch_data
                enc_data = [data.to(self.device) for data in enc_data]
                node_feature = torch.tensor(self.node_feature, dtype=torch.float32, requires_grad=False, device=self.device)
                edge_index = torch.tensor(self.edge_index, dtype=torch.long, requires_grad=False, device=self.device)

                emb = self.model(node_feature, edge_index, enc_data, self.lambda2)
                db_emb.append(emb)

            pbar = tqdm(self.q_dataloader)
            q_emb = []
            for batch_data in pbar:
                enc_data = batch_data
                enc_data = [data.to(self.device) for data in enc_data]
                node_feature = torch.tensor(self.node_feature, dtype=torch.float32, requires_grad=False, device=self.device)
                edge_index = torch.tensor(self.edge_index, dtype=torch.long, requires_grad=False, device=self.device)

                emb = self.model(node_feature, edge_index, enc_data, self.lambda2)
                q_emb.append(emb)
        db_emb = torch.cat(db_emb, dim=0)
        q_emb = torch.cat(q_emb, dim=0)

        if self.dataset == 'rome':
            test_range = [11000, 21000, 31000, 41000]
        else:
            test_range = [11000, 21000, 51000, 101000]

        for end in test_range:
            dists = q_emb @ db_emb[:end].T
            targets = torch.diag(dists)  # [1000]
            result = torch.sum(torch.ge(dists.T, targets)).item() / q_emb.shape[0]
            logger.info(f'MR@{q_emb.shape[0]}/{end} {result}')
