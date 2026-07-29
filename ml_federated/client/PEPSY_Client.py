# ml_federated/client/PEPSY_Client.py

import torch
import numpy as np
from sklearn.metrics import f1_score
from ml_federated.client.Base_Client import Base_Client


class PEPSY_Client(Base_Client):
    def __init__(self, client_id, dataset_configs, device, args):
        super().__init__(client_id, dataset_configs, device, args)

        # Adjust contrastive_weight based on availability level (matching original PEPSY)
        # Higher missingness (low availability) -> 0.2, lower missingness -> 0.1
        base_contrastive_weight = getattr(args, 'pepsy_contrastive_weight', None)
        if base_contrastive_weight is None:
            availability = getattr(args, 'availability', 'complete')
            if availability == 'low':
                base_contrastive_weight = 0.2  # Higher missingness -> higher weight
            else:
                base_contrastive_weight = 0.1  # Lower missingness -> lower weight
        self.contrastive_weight = base_contrastive_weight

    def local_train(self, model):
        self._set_model_mode(model, mode='train')

        optimizer = self._get_optimizer(model)
        epoch_loss_list = []
        for epoch in range(self.local_epochs):
            batch_loss_list = []
            for x, mask, y in self.train_loader:
                x, mask, y = x.to(self.device), mask.to(self.device), y.to(self.device).long()
                optimizer.zero_grad()

                # Forward pass with PEPSY model
                model_outputs, loss = model(x, mask, labels=y, contrastive_weight=self.contrastive_weight)
                loss.backward()

                # Gradient clipping to prevent explosion (especially important for prompts)
                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                batch_loss_list.append(loss.item())
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
            model_outputs = model(x, mask, labels=None, contrastive_weight=self.contrastive_weight)

            # Get predictions
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
        raise ValueError(f'Unsupported optimizer {opt_name}')

    def _set_model_mode(self, model, mode):
        """Set the model to train or eval mode."""
        model.to(self.device)
        if mode == 'train':
            model.train()
        else:
            model.eval()
