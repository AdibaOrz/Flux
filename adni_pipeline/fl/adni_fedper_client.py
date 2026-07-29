"""FedPer client for the ADNI multimodal pipeline."""

import copy
import torch
import numpy as np
from sklearn.metrics import f1_score
from adni_pipeline.fl.adni_fedavg_client import ADNI_FedAvg_Client


class ADNI_FedPer_Client(ADNI_FedAvg_Client):
    """FedPer client: keeps a local private classifier."""

    def __init__(self, client_id, cfg, device):
        super().__init__(client_id, cfg, device)

        self.private_prefixes = {"classifier"}
        self.private_state = None
        self.best_val_f1 = 0.0
        self.best_full_state = None

    def local_train(self, model):
        if self.private_state is not None:
            model.load_state_dict(self.private_state, strict=False)

        self._set_model_mode(model, "train")
        optimizer = self._get_optimizer(model)

        epoch_loss_list = []
        for epoch in range(self.local_epochs):
            batch_loss_list = []
            for X, y, observed in self.train_loader:
                X = {k: v.to(self.device) for k, v in X.items()}
                y = y.to(self.device)
                observed = observed.to(self.device)

                optimizer.zero_grad()
                logits, _ = model(X, observed)
                loss = self.criterion(logits, y)
                loss.backward()
                optimizer.step()
                batch_loss_list.append(loss.item())
            epoch_loss_list.append(np.mean(batch_loss_list))

        avg_epoch_loss = np.mean(epoch_loss_list)

        state_dict = model.state_dict()
        self.private_state = {
            k: v.cpu()
            for k, v in state_dict.items()
            if any(k.startswith(prefix) for prefix in self.private_prefixes)
        }

        return model, avg_epoch_loss

    @torch.no_grad()
    def local_test(self, model, eval_on):
        if self.private_state is not None:
            model.load_state_dict(self.private_state, strict=False)
        if eval_on == "test":
            if self.best_full_state is not None:
                model.load_state_dict(self.best_full_state)
            else:
                print(
                    f"Warning: [{self.client_id}] No best model state for test. "
                    "Using current model state instead."
                )

        self._set_model_mode(model, "eval")
        loader = self.test_loader if eval_on == "test" else self.val_loader
        if loader is None:
            return 0.0

        y_true_list, y_pred_list = [], []
        for X, y, observed in loader:
            X = {k: v.to(self.device) for k, v in X.items()}
            y = y.to(self.device)
            observed = observed.to(self.device)

            logits, _ = model(X, observed)
            _, predicted = torch.max(logits, 1)

            y_true_list = np.concatenate((y_true_list, y.cpu().numpy()), axis=0)
            y_pred_list = np.concatenate((y_pred_list, predicted.cpu().numpy()), axis=0)

        f1_macro = f1_score(y_true_list, y_pred_list, average="macro")

        if eval_on == "val" and f1_macro > self.best_val_f1:
            self.best_val_f1 = f1_macro
            self.best_full_state = copy.deepcopy(model.state_dict())

        return f1_macro
