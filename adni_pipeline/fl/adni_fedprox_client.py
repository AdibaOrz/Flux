"""FedProx client for the ADNI multimodal pipeline."""

import copy
import numpy as np
from adni_pipeline.fl.adni_fedavg_client import ADNI_FedAvg_Client


class ADNI_FedProx_Client(ADNI_FedAvg_Client):
    """FedProx client: FedAvg local training with a proximal regularization term."""

    def __init__(self, client_id, cfg, device):
        super().__init__(client_id, cfg, device)
        self.mu = cfg["fedprox_mu"]

    def local_train(self, model):
        global_ref = copy.deepcopy(model)
        for p in global_ref.parameters():
            p.requires_grad = False

        self._set_model_mode(model, "train")
        self._set_model_mode(global_ref, "eval")

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
                cls_loss = self.criterion(logits, y)
                prox_loss = self._calculate_proximal_term(model, global_ref)
                loss = cls_loss + prox_loss
                loss.backward()
                optimizer.step()
                batch_loss_list.append(loss.item())
            epoch_loss_list.append(np.mean(batch_loss_list))

        avg_epoch_loss = np.mean(epoch_loss_list)
        return model, avg_epoch_loss

    def _calculate_proximal_term(self, model, global_ref):
        """FedProx proximal term: (mu/2) * ||w - w_t||^2 over all parameters."""
        proximal_term = 0.0
        for w, w_t in zip(model.parameters(), global_ref.parameters()):
            proximal_term += (w - w_t).pow(2).sum()
        return 0.5 * self.mu * proximal_term
