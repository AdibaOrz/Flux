# ml_federated/client/FedAvg_Client.py

import copy
import numpy as np
from ml_federated.client.FedAvg_Client import FedAvg_Client



class FedProx_Client(FedAvg_Client):
    """Client for FedProx, which uses a shared model across all clients."""

    def __init__(self, client_id, dataset_configs, device, args):
        super().__init__(client_id, dataset_configs, device, args)
        self.mu = dataset_configs['fedprox_mu']


    def local_train(self, model):
        # Create a global ref (freeze params for efficiency)
        global_ref = copy.deepcopy(model)
        for p in global_ref.parameters():
            p.requires_grad = False

        # Set model modes
        self._set_model_mode(model, mode='train')
        self._set_model_mode(global_ref, mode='eval')

        optimizer = self._get_optimizer(model)
        epoch_loss_list = []
        for epoch in range(self.local_epochs):
            batch_loss_list = []
            for x, mask, y in self.train_loader:
                x, mask, y = x.to(self.device), mask.to(self.device), y.to(self.device).long()
                optimizer.zero_grad()
                model_outputs, _, valid_samples_mask = model(x, mask)
                y = y[valid_samples_mask]
                cls_loss = self.criterion(model_outputs, y) # Classification loss
                prox_loss = self._calculate_proximal_term(model, global_ref) # Proximal loss

                # Combine losses and backprop
                loss = cls_loss + prox_loss
                loss.backward()
                optimizer.step()
                batch_loss_list.append(loss.item())
            # print(f'Client {self.client_id} Epoch {epoch} Loss: {np.mean(batch_loss_list)}')
            epoch_loss_list.append(np.mean(batch_loss_list))

        avg_epoch_loss = np.mean(epoch_loss_list)
        return model, avg_epoch_loss


    def _calculate_proximal_term(self, model, global_ref):
        """Calculate the FedProx proximal term: (μ/2) * ||w - w_t||²
           where ||w - w_t||² = Σᵢ(wᵢ - w_t,ᵢ)² for all model parameters wᵢ."""
        proximal_term = 0.0
        for w, w_t in zip(model.parameters(), global_ref.parameters()):
            proximal_term += (w - w_t).pow(2).sum()
        prox_loss = 0.5 * self.mu * proximal_term
        return prox_loss
