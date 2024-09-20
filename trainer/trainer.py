import os
import torch
import logging
import numpy as np
import torch.nn as nn

from tqdm import tqdm


class REDTrainer:
    def __init__(self,
                 config,
                 model,
                 node_feature,
                 edge_index,
                 train_dataloader
                 ):
        super().__init__()
        self.clip = config['clip']
        self.lr = config['lr']
        self.betas = eval(config['betas'])
        self.weight_decay = config['weight_decay']
        self.device = config['device']
        self.epochs = config['epochs']
        self.warmup_steps = config['warmup_steps']
        self.scheduler = config['scheduler']
        self.dim = config['enc_embed_dim']
        self.lambda1 = config['lambda1']
        self.lambda2 = config['lambda2']

        self.node_feature = node_feature
        self.edge_index = edge_index
        self.train_dataloader = train_dataloader

        self.model = model.to(self.device)

        self.optimizer = torch.optim.AdamW(self.model.parameters(),
                                           lr=self.lr,
                                           weight_decay=self.weight_decay,
                                           betas=self.betas)

        self.loss_fn = nn.NLLLoss(ignore_index=0)
        self.save_path = os.path.join('checkpoints', config['data_name'], config['exp_id'], 'pretraining')
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def iteration(self, epoch, dataloader, iteration_type):
        losses = []
        pbar = tqdm(dataloader)
        for batch_data in pbar:
            enc_data, dec_data, y1, y2 = batch_data
            enc_data = [data.to(self.device) for data in enc_data]
            dec_data = [data.to(self.device) for data in dec_data]
            y1 = y1.to(self.device)
            y2 = y2.to(self.device)
            node_feature = torch.tensor(self.node_feature, dtype=torch.float32, requires_grad=False, device=self.device)
            edge_index = torch.tensor(self.edge_index, dtype=torch.long, requires_grad=False, device=self.device)

            self.optimizer.zero_grad()
            pred1, pred2 = self.model(node_feature, edge_index, enc_data, dec_data, self.lambda2)
            loss = self.lambda1 * self.loss_fn(pred1.transpose(1, 2), y1) + (1 - self.lambda1) * self.loss_fn(pred2.transpose(1, 2), y2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            pbar.set_description('[{} Epoch {}/{}: loss: %f]'.format(iteration_type, str(epoch), str(self.epochs)) % loss)
            losses.append(loss.item())
        return np.array(losses).mean()

    def train(self):
        for epoch in range(self.epochs):
            self.model.train()
            train_loss = self.iteration(epoch, self.train_dataloader, 'train')
            logging.info(f'Epoch {epoch}/{self.epochs}, avg train loss: {train_loss}')
            torch.save(self.model.state_dict(), f'{self.save_path}/pretraining_{epoch + 1}.pt')
