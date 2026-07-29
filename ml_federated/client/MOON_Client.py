# ml_federated/client/MOON_Client.py

import copy
import torch
import numpy as np
from ml_federated.client.FedAvg_Client import FedAvg_Client



class MOON_Client(FedAvg_Client):
    """MOON client: trains a local model using contrastive loss
       to align with the global model (positive) and diverge
       from its own previous local model (negative)."""
    def __init__(self, client_id, dataset_configs, device, args):
        super().__init__(client_id, dataset_configs, device, args)
        # MOON-specific params & configs
        self.moon_mu = dataset_configs['moon_mu']
        self.moon_temp = dataset_configs['moon_temp']

        self.cos_sim = torch.nn.CosineSimilarity(dim=-1).to(self.device)
        self.prev_local_model = None # Will be filled after the 1st round


    def local_train(self, model):
        global_model = copy.deepcopy(model) # Positive
        prev_local_model = (self.prev_local_model or global_model) # Negative

        # Set modes
        self._set_model_mode(model, mode='train')
        self._set_model_mode(global_model, mode='eval')
        self._set_model_mode(prev_local_model, mode='eval')

        # Get optimizer
        optimizer = self._get_optimizer(model)
        epoch_loss_list = []

        for epoch in range(self.local_epochs):
            batch_loss_list = []
            for x, mask, y in self.train_loader:
                x, mask, y = x.to(self.device), mask.to(self.device), y.to(self.device).long()
                optimizer.zero_grad()
                model_outputs, feats, _, valid_samples_mask = model(x, mask, return_projection_output=True)
                y = y[valid_samples_mask]
                moon_loss = self._calculate_moon_loss(x, mask, feats, global_model, prev_local_model)
                cls_loss = self.criterion(model_outputs, y)
                loss = cls_loss + self.moon_mu * moon_loss
                loss.backward()
                optimizer.step()
                batch_loss_list.append(loss.item())
            # print(f'Client {self.client_id} Epoch {epoch} Loss: {np.mean(batch_loss_list)}')
            epoch_loss_list.append(np.mean(batch_loss_list))

        # Keep a copy of the new local model for the next round
        self.prev_local_model = copy.deepcopy(model)

        avg_epoch_loss = np.mean(epoch_loss_list)
        return model, avg_epoch_loss


    def _get_features(self, backbone, x, mask):
        _, feats, _, _ = backbone(x, mask, return_projection_output=True)
        return feats


    def _calculate_moon_loss(self, x, mask, feat_local, global_model, prev_local_model):
        # We don't need gradients for the global and previous local models
        with torch.no_grad():
            feat_pos = self._get_features(global_model, x, mask)
            feat_neg = self._get_features(prev_local_model, x, mask)

        pos = self.cos_sim(feat_local, feat_pos)
        neg = self.cos_sim(feat_local, feat_neg)

        logits = torch.cat((pos.reshape(-1, 1), neg.reshape(-1, 1)), dim=1)
        logits /= self.moon_temp

        targets = torch.zeros(x.size(0)).to(self.device).long()
        loss_moon = self.criterion(logits, targets)
        return loss_moon
