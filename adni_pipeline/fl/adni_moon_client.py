"""MOON client for the ADNI multimodal pipeline."""

import copy
import torch
import numpy as np
from adni_pipeline.fl.adni_fedavg_client import ADNI_FedAvg_Client


class ADNI_Moon_Client(ADNI_FedAvg_Client):
    """MOON client: contrastive alignment with global model, divergence from prev local."""

    def __init__(self, client_id, cfg, device):
        super().__init__(client_id, cfg, device)
        self.moon_mu = cfg["moon_mu"]
        self.moon_temp = cfg["moon_temp"]
        self.cos_sim = torch.nn.CosineSimilarity(dim=-1).to(self.device)
        self.prev_local_model = None

    def local_train(self, model):
        global_model = copy.deepcopy(model)
        prev_local_model = self.prev_local_model or global_model

        self._set_model_mode(model, "train")
        self._set_model_mode(global_model, "eval")
        self._set_model_mode(prev_local_model, "eval")

        optimizer = self._get_optimizer(model)
        epoch_loss_list = []

        for epoch in range(self.local_epochs):
            batch_loss_list = []
            for X, y, observed in self.train_loader:
                X = {k: v.to(self.device) for k, v in X.items()}
                y = y.to(self.device)
                observed = observed.to(self.device)

                optimizer.zero_grad()
                logits, feats, _ = model(X, observed, return_projection_output=True)
                moon_loss = self._calculate_moon_loss(
                    feats, global_model, prev_local_model, X, observed
                )
                cls_loss = self.criterion(logits, y)
                loss = cls_loss + self.moon_mu * moon_loss
                loss.backward()
                optimizer.step()
                batch_loss_list.append(loss.item())
            epoch_loss_list.append(np.mean(batch_loss_list))

        self.prev_local_model = copy.deepcopy(model)
        avg_epoch_loss = np.mean(epoch_loss_list)
        return model, avg_epoch_loss

    def _get_features(self, backbone, X, observed):
        _, feats, _ = backbone(X, observed, return_projection_output=True)
        return feats

    def _calculate_moon_loss(self, feat_local, global_model, prev_local_model, X, observed):
        with torch.no_grad():
            feat_pos = self._get_features(global_model, X, observed)
            feat_neg = self._get_features(prev_local_model, X, observed)

        pos = self.cos_sim(feat_local, feat_pos)
        neg = self.cos_sim(feat_local, feat_neg)

        logits = torch.cat((pos.reshape(-1, 1), neg.reshape(-1, 1)), dim=1)
        logits /= self.moon_temp

        batch_size = observed.shape[0]
        targets = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        return self.criterion(logits, targets)
