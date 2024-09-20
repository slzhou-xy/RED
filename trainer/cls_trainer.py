import os
import torch
from loguru import logger

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from utils.metric import F1, Cls_HR

from timm.scheduler import CosineLRScheduler


class ClsTrainer:
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
        self.num_user = config['user_size']

        # early stop
        self.early_stop = config['early_stop']
        self.eval_losses = [torch.inf]

        self.node_feature = node_feature
        self.edge_index = edge_index
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.test_dataloader = test_dataloader

        self.model = model.to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            betas=(0.9, 0.95),
            weight_decay=0.05
        )

        self.scheduler = CosineLRScheduler(optimizer=self.optimizer, t_initial=self.epochs, warmup_t=5, warmup_lr_init=1e-6)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=0)
        self.save_path = os.path.join('checkpoints', config['dataset'], config['exp_id'], 'multi_cls')
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        logger.add(f'{self.save_path}/results.log')

    def iteration(self, epoch, dataloader, iteration_type='train'):
        losses = []
        pbar = tqdm(dataloader)

        for batch_data in pbar:
            enc_data, id_y, y = batch_data
            node_feature = torch.tensor(self.node_feature, dtype=torch.float32, requires_grad=False, device=self.device)
            edge_index = torch.tensor(self.edge_index, dtype=torch.long, requires_grad=False, device=self.device)

            enc_data = [data.to(self.device) for data in enc_data]
            id_y = id_y.to(self.device)

            if iteration_type == 'train':
                self.optimizer.zero_grad()
                pred, logits = self.model(node_feature, edge_index, enc_data)
                loss = self.loss_fn(pred, id_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
                self.optimizer.step()
            else:
                with torch.no_grad():
                    pred, logits = self.model(node_feature, edge_index, enc_data)
                    loss = self.ce(pred, id_y)
            pbar.set_description('[{} Epoch {}/{}: loss: %f]'.format(iteration_type, str(epoch), str(self.epochs)) % loss)
            losses.append(loss.item())
        return np.array(losses).mean()

    def train(self):
        train_losses = []
        eval_losses = []
        best_epoch = 0
        min_loss = torch.inf
        for epoch in range(self.epochs):
            self.model.train()
            train_loss = self.iteration(epoch, self.train_dataloader, 'train')

            eval_loss = self.eval(epoch)
            logger.info(f'=====> avg train loss: {train_loss}  |  avg eval loss: {eval_loss}')

            if min_loss > eval_loss:
                min_loss = eval_loss
                best_epoch = epoch

            torch.save(self.model.pretraining_model.state_dict(), f'{self.save_path}/cls_{epoch}.pt')
            self.scheduler.step(epoch + 1)
            train_losses.append(train_loss)
            eval_losses.append(eval_loss)

        logger.info(f'=====> best epoch: {best_epoch}')
        self.test(best_epoch)

    def eval(self, epoch):
        self.model.eval()
        eval_loss = self.iteration(epoch, self.eval_dataloader, 'eval ')
        return eval_loss

    def test(self, best_epoch):
        self.model.load_state_dict(torch.load(f'{self.save_path}/cls_{best_epoch}.pt'))
        self.model.eval()
        with torch.no_grad():
            pbar = tqdm(self.test_dataloader, desc='Traj inference:')
            labels = []
            preds = []
            indices = []
            for batch_data in pbar:
                enc_data, id_y, y = batch_data
                node_feature = torch.tensor(self.node_feature, dtype=torch.float32, requires_grad=False, device=self.device)
                edge_index = torch.tensor(self.edge_index, dtype=torch.long, requires_grad=False, device=self.device)

                enc_data = [data.to(self.device) for data in enc_data]
                id_y = id_y.to(self.device)

                pred, logits = self.model(node_feature, edge_index, enc_data)
                indices.append(F.log_softmax(pred, dim=-1).topk(5).indices.cpu().detach().numpy())
                labels.append(id_y.cpu().detach().numpy())
                preds.append(pred.cpu().detach().numpy())

            labels = np.concatenate(labels, axis=0)
            preds = np.argmax(np.concatenate(preds, axis=0), axis=1)
            indices = np.vstack(indices)

            recall_5 = Cls_HR(labels, indices)

            micro_f1, macro_f1 = F1(labels, preds, self.num_user)
            logger.info(f'micro_f1: {micro_f1}, macro_f1: {macro_f1}, recall@5: {recall_5}')
            return micro_f1, macro_f1, recall_5
