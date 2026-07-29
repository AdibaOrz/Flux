"""PEPSY client for the ADNI multimodal pipeline."""

import torch
import numpy as np
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from adni_pipeline.configs.adni_config import resolve_pepsy_contrastive_weight
from adni_pipeline.data.loader import ADNINativeDataset
from adni_pipeline.fl.adni_fedavg_client import adni_collate_fn


class ADNI_PEPSY_Client:
    """PEPSY client: local train with model-internal contrastive + CE loss."""

    def __init__(self, client_id, cfg, device):
        self.client_id = client_id
        self.cfg = cfg
        self.device = device
        self.local_epochs = cfg["local_epochs"]
        self.modalities = cfg["modalities"]
        self.contrastive_weight = cfg.get(
            "pepsy_contrastive_weight",
            resolve_pepsy_contrastive_weight(cfg["partition_name"]),
        )

        adni_root = cfg["adni_root"]
        partition = cfg["partition_name"]
        bs = cfg["local_bs"]

        train_ds = ADNINativeDataset(adni_root, client_id, "train", partition)
        val_ds = ADNINativeDataset(adni_root, client_id, "val", partition)
        test_ds = ADNINativeDataset(adni_root, client_id, "test", partition)

        loader_kwargs = {"collate_fn": adni_collate_fn, "num_workers": 4, "pin_memory": True}
        self.train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                                       **loader_kwargs) if len(train_ds) > 0 else None
        self.val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False,
                                     **loader_kwargs) if len(val_ds) > 0 else None
        self.test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False,
                                      **loader_kwargs) if len(test_ds) > 0 else None

        self.data_sizes, self.modality_data_sizes = self._compute_data_sizes(train_ds, val_ds, test_ds)

    def _compute_data_sizes(self, train_ds, val_ds, test_ds):
        data_sizes = {}
        modality_data_sizes = {}
        for key, ds in [("train", train_ds), ("val", val_ds), ("test", test_ds)]:
            data_sizes[key] = len(ds)
            modality_data_sizes[key] = ds.observed.sum(dim=0).tolist()
        return data_sizes, modality_data_sizes

    def get_data_size(self, data_type):
        return self.data_sizes[data_type]

    def get_modality_data_size(self, data_type):
        return self.modality_data_sizes[data_type]

    def local_train(self, model):
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
                _, loss = model(X, observed, labels=y,
                                contrastive_weight=self.contrastive_weight)
                loss.backward()
                optimizer.step()
                batch_loss_list.append(loss.item())
            epoch_loss_list.append(np.mean(batch_loss_list))

        avg_epoch_loss = np.mean(epoch_loss_list)
        return model, avg_epoch_loss

    @torch.no_grad()
    def local_test(self, model, eval_on):
        self._set_model_mode(model, "eval")
        loader = self.test_loader if eval_on == "test" else self.val_loader
        if loader is None:
            return 0.0

        y_true_list, y_pred_list = [], []
        for X, y, observed in loader:
            X = {k: v.to(self.device) for k, v in X.items()}
            y = y.to(self.device)
            observed = observed.to(self.device)

            logits = model(X, observed, labels=None,
                           contrastive_weight=self.contrastive_weight)
            _, predicted = torch.max(logits, 1)

            y_true_list = np.concatenate((y_true_list, y.cpu().numpy()), axis=0)
            y_pred_list = np.concatenate((y_pred_list, predicted.cpu().numpy()), axis=0)

        return f1_score(y_true_list, y_pred_list, average="macro")

    def _get_optimizer(self, model):
        lr = self.cfg["local_lr"]
        return torch.optim.Adam(model.parameters(), lr=lr)

    def _set_model_mode(self, model, mode):
        model.to(self.device)
        if mode == "train":
            model.train()
        else:
            model.eval()
