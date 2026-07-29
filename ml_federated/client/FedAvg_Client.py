# ml_federated/client/FedAvg_Client.py

import torch
import numpy as np
from sklearn.metrics import f1_score
from ml_federated.client.Base_Client import Base_Client



class FedAvg_Client(Base_Client):
    """Client for FedAvg, which uses a shared model across all clients."""

    def __init__(self, client_id, dataset_configs, device, args):
        super().__init__(client_id, dataset_configs, device, args)


    def local_train(self, model):
        self._set_model_mode(model, mode='train')

        optimizer = self._get_optimizer(model)
        epoch_loss_list = []
        for epoch in range(self.local_epochs):
            batch_loss_list = []
            for x, mask, y in self.train_loader:
                x, mask, y = x.to(self.device), mask.to(self.device), y.to(self.device).long()
                optimizer.zero_grad()
                model_outputs, _, valid_samples_mask = model(x, mask)
                y = y[valid_samples_mask]
                loss = self.criterion(model_outputs, y)
                loss.backward()
                optimizer.step()
                batch_loss_list.append(loss.item())
            # print(f'Client {self.client_id} Epoch {epoch} Loss: {np.mean(batch_loss_list)}')
            epoch_loss_list.append(np.mean(batch_loss_list))

        avg_epoch_loss = np.mean(epoch_loss_list)
        return model, avg_epoch_loss


    @torch.no_grad()
    def local_test(self, model, eval_on):
        self._set_model_mode(model, mode='eval')

        eval_loader = self.test_loader if eval_on == 'test' else self.val_loader
        y_true_list, y_pred_list = [], []
        for x, mask, y in eval_loader:
            x, mask, y = x.to(self.device), mask.to(self.device), y.to(self.device).long()
            model_outputs, _, valid_samples_mask = model(x, mask)
            y = y[valid_samples_mask]
            _, predicted = torch.max(model_outputs, 1)
            predicted = predicted.view(-1)

            y_true_list = np.concatenate((y_true_list, y.cpu()), axis=0)
            y_pred_list = np.concatenate((y_pred_list, predicted.cpu()), axis=0)

        f1_macro = f1_score(y_true_list, y_pred_list, average='macro')
        return f1_macro


    def _get_optimizer(self, model):
        opt_name = self.dataset_configs['optimizer']
        lr = self.dataset_configs['local_lr']
        wd = self.dataset_configs['local_wd']
        if opt_name == 'sgd':
            return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=0.9)
        elif opt_name == 'adam':
            return torch.optim.Adam(model.parameters(), lr=lr)
        raise ValueError(f'Unsupported optimizer {opt_name}')


    def _set_model_mode(self, model, mode):
        """Set the model to train or eval mode."""
        model.to(self.device)
        if mode == 'train':
            model.train()
        else:
            model.eval()