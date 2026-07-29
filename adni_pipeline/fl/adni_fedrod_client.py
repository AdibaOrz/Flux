"""FedRoD client (without BRM) for the ADNI multimodal pipeline."""

import copy
import torch
import numpy as np
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from adni_pipeline.data.loader import ADNINativeDataset
from adni_pipeline.fl.adni_fedavg_client import adni_collate_fn


class ADNI_FedRoD_Client:
    """FedRoD client: shared global head + private local head (p_head).

    Training has two phases per batch:
      1. Shared pass: CE loss on global logits, backprop through all shared params.
      2. Personal pass: detach fusion output, run through p_head,
         loss on (logits_p + logits_g_detached), backprop only through p_head.

    At inference: prediction = argmax(logits_p + logits_g).
    """

    def __init__(self, client_id, cfg, device):
        self.client_id = client_id
        self.cfg = cfg
        self.device = device
        self.local_epochs = cfg["local_epochs"]
        self.modalities = cfg["modalities"]
        self.num_classes = cfg["num_classes"]

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

        self.p_head = None
        self.best_full_state = None
        self.best_p_head_state = None
        self.best_val_f1 = -1.0

    # ------------------------------------------------------------------ #
    # Data size helpers
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
        self._ensure_p_head(model)
        self._set_model_mode(model, "train")
        opt_shared, opt_personal = self._get_optimizers(model)

        epoch_losses = []
        for _ in range(self.local_epochs):
            shared_losses, private_losses = [], []
            for X, y, observed in self.train_loader:
                X = {k: v.to(self.device) for k, v in X.items()}
                y = y.to(self.device)
                observed = observed.to(self.device)

                # ---- Shared pass ---- #
                opt_shared.zero_grad()
                logits_g, _, fusion_out = model(X, observed, return_fusion_out=True)
                loss_shared = self.criterion(logits_g, y)
                loss_shared.backward()
                opt_shared.step()

                # ---- Personal pass ---- #
                opt_personal.zero_grad()
                logits_g_detached = logits_g.detach()
                fusion_out_detached = fusion_out.detach()
                logits_p = self.p_head(fusion_out_detached)
                loss_p = self.criterion(logits_p + logits_g_detached, y)
                loss_p.backward()
                opt_personal.step()

                shared_losses.append(loss_shared.item())
                private_losses.append(loss_p.item())

            avg = (np.mean(shared_losses) + np.mean(private_losses)) / 2.0
            epoch_losses.append(avg)

        avg_epoch_loss = np.mean(epoch_losses)
        return model, avg_epoch_loss

    # ------------------------------------------------------------------ #
    # Local evaluation
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def local_test(self, model, eval_on):
        self._ensure_p_head(model)

        if eval_on == "test":
            if self.best_full_state is not None:
                model.load_state_dict(self.best_full_state)
            if self.best_p_head_state is not None:
                self.p_head.load_state_dict(self.best_p_head_state)

        self._set_model_mode(model, "eval")
        loader = self.test_loader if eval_on == "test" else self.val_loader
        if loader is None:
            return 0.0

        y_true_list, y_pred_list = [], []
        for X, y, observed in loader:
            X = {k: v.to(self.device) for k, v in X.items()}
            y = y.to(self.device)
            observed = observed.to(self.device)

            logits_g, _, fusion_out = model(X, observed, return_fusion_out=True)
            logits_p = self.p_head(fusion_out)
            logits = logits_p + logits_g
            _, predicted = torch.max(logits, 1)

            y_true_list = np.concatenate((y_true_list, y.cpu().numpy()), axis=0)
            y_pred_list = np.concatenate((y_pred_list, predicted.cpu().numpy()), axis=0)

        f1_macro = f1_score(y_true_list, y_pred_list, average="macro")

        if eval_on == "val" and f1_macro > self.best_val_f1:
            self.best_val_f1 = f1_macro
            self.best_full_state = copy.deepcopy(model.state_dict())
            self.best_p_head_state = copy.deepcopy(self.p_head.state_dict())

        return f1_macro

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _ensure_p_head(self, model):
        if self.p_head is None:
            self.p_head = copy.deepcopy(model.classifier).to(self.device)

    def _get_optimizers(self, model):
        lr = self.cfg["local_lr"]
        opt_shared = torch.optim.Adam(model.parameters(), lr=lr)
        opt_personal = torch.optim.Adam(self.p_head.parameters(), lr=lr)
        return opt_shared, opt_personal

    def _set_model_mode(self, model, mode):
        model.to(self.device)
        self.p_head.to(self.device)
        if mode == "train":
            model.train()
            self.p_head.train()
        else:
            model.eval()
            self.p_head.eval()
