# ml_federated/client/FedPer_Client.py

import copy
import torch
import numpy as np
from sklearn.metrics import f1_score
from ml_federated.client.FedAvg_Client import FedAvg_Client



class FedPer_Client(FedAvg_Client):
    """
    FedPer Client. Keeps a local 'personal_state' dict containing:
        - always: the classification head (prefix 'head')
        - plus: the fusion params if --private_fusion is set
    """
    def __init__(self, client_id, dataset_configs, device, args):
        super().__init__(client_id, dataset_configs, device, args)

        # Decide which parameters are private
        self.private_prefixes = {"head"}
        if args.private_fusion: self.private_prefixes.add("fusion")

        self.private_state = None # Latest private params
        self.best_val_f1 = 0.0
        self.best_full_state = None  # Full model state (shared + private) of the best model


    def local_train(self, model):
        # model (already deepcopy applied from the server)
        if self.private_state is not None:
            model.load_state_dict(self.private_state, strict=False)

        # Set model to train
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

        # Stash new private params for the next round
        state_dict = model.state_dict()
        self.private_state = {k: v.cpu() for k, v in state_dict.items()
                             if any(k.startswith(prefix) for prefix in self.private_prefixes)}
        return model, avg_epoch_loss


    @torch.no_grad()
    def local_test(self, model, eval_on):
        """
        - eval_on == 'val' -> evaluate current model, update best snapshot
        - eval_on == 'test' -> evaluate best snapshot
        """

        # Load the private state if it exists
        if self.private_state is not None:
            model.load_state_dict(self.private_state, strict=False)
        if eval_on == 'test':
            if self.best_full_state is not None:
                model.load_state_dict(self.best_full_state)
            else:
                print('Warning: No best model state available for testing. Using current model state instead.')

        # Set model to eval mode
        self._set_model_mode(model, mode='eval')

        eval_loader = self.val_loader if eval_on == 'val' else self.test_loader
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

        # Update best‑validation snapshot
        if eval_on == "val" and f1_macro > self.best_val_f1:
            self.best_val_f1 = f1_macro
            # store *entire* model (backbone + head) so features match
            self.best_full_state = copy.deepcopy(model.state_dict())
            # print(f'--> Client {self.client_id} New best val F1: {self.best_val_f1:.4f}')

        return f1_macro
