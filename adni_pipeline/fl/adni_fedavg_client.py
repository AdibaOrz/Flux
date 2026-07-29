"""FedAvg client for the ADNI multimodal pipeline."""

import torch
import numpy as np
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from adni_pipeline.data.loader import ADNINativeDataset


def adni_collate_fn(batch):
    """Collate ADNI samples into batched tensors."""
    data_dicts, labels, observed_vecs = zip(*batch)
    modalities = data_dicts[0].keys()
    batched_data = {m: torch.stack([d[m] for d in data_dicts]) for m in modalities}
    batched_labels = torch.stack(labels)
    batched_observed = torch.stack(observed_vecs)
    return batched_data, batched_labels, batched_observed


class ADNI_FedAvg_Client:
    """FedAvg client that loads ADNI data and performs local train/test."""

    def __init__(self, client_id, cfg, device):
        """
        Args:
            client_id: silo key string (e.g. "silo_0").
            cfg: dict with keys from ADNI_DATASET_CFG plus 'adni_root'.
            device: torch device.
        """
        self.client_id = client_id
        self.cfg = cfg
        self.device = device
        self.local_epochs = cfg["local_epochs"]
        self.modalities = cfg["modalities"]

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
        self.criterion = torch.nn.CrossEntropyLoss().to(device)

    # ------------------------------------------------------------------ #
    # Data size helpers (used by server for weighted aggregation)
    # ------------------------------------------------------------------ #

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

    # ------------------------------------------------------------------ #
    # Local training
    # ------------------------------------------------------------------ #

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
                logits, _ = model(X, observed)
                loss = self.criterion(logits, y)
                loss.backward()
                optimizer.step()
                batch_loss_list.append(loss.item())
            epoch_loss_list.append(np.mean(batch_loss_list))

        avg_epoch_loss = np.mean(epoch_loss_list)
        return model, avg_epoch_loss

    # ------------------------------------------------------------------ #
    # Local evaluation
    # ------------------------------------------------------------------ #

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

            logits, _ = model(X, observed)
            _, predicted = torch.max(logits, 1)

            y_true_list = np.concatenate((y_true_list, y.cpu().numpy()), axis=0)
            y_pred_list = np.concatenate((y_pred_list, predicted.cpu().numpy()), axis=0)

        f1_macro = f1_score(y_true_list, y_pred_list, average="macro")
        return f1_macro

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _get_optimizer(self, model):
        lr = self.cfg["local_lr"]
        return torch.optim.Adam(model.parameters(), lr=lr)

    def _set_model_mode(self, model, mode):
        model.to(self.device)
        if mode == "train":
            model.train()
        else:
            model.eval()
