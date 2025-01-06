import os
import torch
from loguru import logger
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from timm.scheduler import CosineLRScheduler

from sklearn.metrics import accuracy_score, precision_score, f1_score


class BinaryClsTrainer:
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
        self.lambda2 = config['lambda2']

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

        self.scheduler = CosineLRScheduler(optimizer=self.optimizer, t_initial=self.epochs, warmup_t=10, warmup_lr_init=1e-6)
        self.loss_fn = nn.BCEWithLogitsLoss()

        self.save_path = os.path.join('checkpoints', config['dataset'], config['exp_id'], 'binary_cls')
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        logger.add(f'{self.save_path}/results.log', mode='w')

    def iteration(self, epoch, dataloader, iteration_type='Train'):
        losses = []
        pbar = tqdm(dataloader, ncols=100)

        for batch_data in pbar:
            enc_data, id_y = batch_data
            node_feature = torch.tensor(self.node_feature, dtype=torch.float32, requires_grad=False, device=self.device)
            edge_index = torch.tensor(self.edge_index, dtype=torch.long, requires_grad=False, device=self.device)

            enc_data = [data.to(self.device) for data in enc_data]
            id_y = id_y.to(self.device)

            if iteration_type == 'Train':
                self.optimizer.zero_grad()

                pred = self.model(node_feature, edge_index, enc_data, self.lambda2)
                loss = self.loss_fn(pred.reshape(-1), id_y.float())

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
                self.optimizer.step()

            else:
                pred = self.model(node_feature, edge_index, enc_data, self.lambda2)
                loss = self.ce1(pred.reshape(-1), id_y.float())
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
            self.scheduler.step(epoch)
            train_loss = self.iteration(epoch, self.train_dataloader, 'Train')

            eval_loss = self.eval(epoch)
            logger.info(f'Epoch {epoch} | Train loss: {train_loss:.8f} | Eval loss: {eval_loss:.8f}')
            self.test(epoch)

            if min_loss > eval_loss:
                min_loss = eval_loss
                best_epoch = epoch

            torch.save(self.model.state_dict(), f'{self.save_path}/cls_{epoch}.pt')

            train_losses.append(train_loss)
            eval_losses.append(eval_loss)

        logger.info('=====> best epoch: %d ' % (best_epoch))
        self.test(best_epoch)

    def eval(self, epoch):
        self.model.eval()
        with torch.no_grad():
            eval_loss = self.iteration(epoch, self.eval_dataloader, 'Eval ')
        return eval_loss

    def test(self, best_epoch):
        self.model.load_state_dict(torch.load(f'{self.save_path}/cls_{best_epoch}.pt'))
        self.model.eval()
        with torch.no_grad():
            pbar = tqdm(self.test_dataloader, desc='Traj inference:', ncols=100)
            labels = []
            preds = []
            for batch_data in pbar:
                enc_data, id_y = batch_data
                node_feature = torch.tensor(self.node_feature, dtype=torch.float32, requires_grad=False, device=self.device)
                edge_index = torch.tensor(self.edge_index, dtype=torch.long, requires_grad=False, device=self.device)

                enc_data = [data.to(self.device) for data in enc_data]
                id_y = id_y.to(self.device)

                pred, logits = self.model(node_feature, edge_index, enc_data, self.lambda2)
                labels.append(id_y.cpu().detach().numpy())
                preds.append(F.sigmoid(pred).cpu().detach().numpy())

            labels = np.concatenate(labels, axis=0)
            preds = np.concatenate(preds, axis=0)
            preds = np.where(preds > 0.5, 1, 0)

            preds = list(preds)
            truth = list(labels)

            assert len(preds) == len(truth)
            intermediate_result = {}
            intermediate_result['pred'] = preds
            intermediate_result['truth'] = truth

            accuracy = accuracy_score(intermediate_result['truth'], intermediate_result['pred'])
            precision = precision_score(intermediate_result['truth'], intermediate_result['pred'])
            f1 = f1_score(intermediate_result['truth'], intermediate_result['pred'])
            logger.info(f'Accuracy: {accuracy::.6f}, Precision: {precision::.6f}, F1: {f1::.6f}')
