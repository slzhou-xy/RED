import os
import torch
import pickle
from loguru import logger

from tqdm import tqdm
from utils.metric import HR


class SimTrainer:
    def __init__(self,
                 config,
                 model,
                 node_feature,
                 edge_index,
                 train_dataloader,
                 eval_dataloader,
                 test_dataloader
                 ):
        super().__init__()
        self.lr = config['lr']
        self.device = config['device']
        self.epochs = config['epochs']
        self.clip = config['clip']
        self.dataset = config['dataset']
        self.dim = config['enc_embed_dim']
        self.lambda2 = config['lambda2']
        self.data_path = config['data_path']

        # early stop
        self.early_stop = config['early_stop']
        self.eval_losses = [torch.inf]

        self.node_feature = node_feature
        self.edge_index = edge_index
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.test_dataloader = test_dataloader

        self.model = model.to(self.device)

        self.save_path = os.path.join('checkpoints', config['dataset'], config['exp_id'], 'compute_sim')
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        logger.add(f'{self.save_path}/results.log', mode='w')

    def test(self):
        self.model.eval()
        with torch.no_grad():
            pbar = tqdm(self.test_dataloader, desc='Traj inference')
            hausdorff_truth = pickle.load(open(f'{self.data_path}/{self.dataset}/hausdorff_sim_test_label.pkl', 'rb'))
            frechet_truth = pickle.load(open(f'{self.data_path}/{self.dataset}/frechet_sim_test_label.pkl', 'rb'))
            assert len(hausdorff_truth) == len(frechet_truth)
            traj_emb = torch.zeros(size=(len(frechet_truth), self.dim), device=self.device)

            for batch_data in pbar:
                traj_id, enc_data = batch_data
                enc_data = [data.to(self.device) for data in enc_data]
                node_feature = torch.tensor(self.node_feature, dtype=torch.float32, requires_grad=False, device=self.device)
                edge_index = torch.tensor(self.edge_index, dtype=torch.long, requires_grad=False, device=self.device)

                emb = self.model(node_feature, edge_index, enc_data, self.lambda2)

                traj_emb[traj_id] = emb.to(self.device)
        rank = torch.zeros(size=(len(frechet_truth), 50), device=self.device)

        num = 1000
        bz = len(frechet_truth) // num
        for i in range(num):
            start = i * bz
            end = (i + 1) * bz
            if i + 1 == num:
                end = len(frechet_truth)
            sim = torch.einsum('ab, bc->ac', traj_emb[start:end], traj_emb.T)
            for j in range(sim.shape[0]):
                sim[j, start + j] = 0
            sim = torch.argsort(sim, descending=True)
            rank[start:end] = sim[:, :50]
        rank = rank.cpu().numpy()
        res = HR(hausdorff_truth, rank, [1, 5, 10, 20, 50])
        logger.info(f'Hausdorff distance HR {res}')

        res = HR(frechet_truth, rank, [1, 5, 10, 20, 50])
        logger.info(f'Frechet distance HR {res}')
