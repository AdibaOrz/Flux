"""PmcmFL client for the ADNI multimodal pipeline."""

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from adni_pipeline.data.loader import ADNINativeDataset
from adni_pipeline.fl.adni_fedavg_client import adni_collate_fn


class ClipLossHalf(nn.Module):
    """Contrastive loss: CE(scale * features @ prototypes.T, arange(B))."""

    def __init__(self, cache_labels=False):
        super().__init__()
        self.cache_labels = cache_labels
        self.prev_num_logits = 0
        self.labels = {}

    def forward(self, image_features, text_features, logit_scale):
        device = image_features.device
        logits_per_image = logit_scale * image_features @ text_features.T

        num_logits = logits_per_image.shape[0]
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]
        return F.cross_entropy(logits_per_image, labels)


class ADNI_PmcmFL_Client:
    """PmcmFL client: prototype-augmented training with CLIP loss."""

    def __init__(self, client_id, cfg, device):
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
        self.criterion = nn.CrossEntropyLoss().to(device)
        self.clip_criterion = ClipLossHalf()
        self.clip_logits_scale = nn.Parameter(
            torch.ones([]) * np.log(1 / 0.07)
        ).to(device)
        self.clip_loss_weight = cfg.get("pmcmfl_clip_loss_weight", 0.1)

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

    def local_train(self, model, prototypes):
        self._set_model_mode(model, "train")
        optimizer = self._get_optimizer(model)

        epoch_loss_list = []
        for epoch in range(self.local_epochs):
            batch_loss_list = []
            for X, y, observed in self.train_loader:
                X = {k: v.to(self.device) for k, v in X.items()}
                y = y.to(self.device).long()
                observed = observed.to(self.device)

                optimizer.zero_grad()
                logits, unimodal_embeds, fusion_out, valid_mask = model(
                    X, observed, labels=y, prototypes=prototypes,
                    return_embeds=True, mode="train",
                )
                y_valid = y[valid_mask]

                ce_loss = self.criterion(logits, y_valid)

                fusion_proto = prototypes["fusion"].to(self.device)
                clip_loss = self.clip_criterion(
                    fusion_out, fusion_proto[y_valid], self.clip_logits_scale.exp(),
                )
                loss = ce_loss + self.clip_loss_weight * clip_loss

                loss.backward()
                optimizer.step()
                batch_loss_list.append(loss.item())
            epoch_loss_list.append(np.mean(batch_loss_list))

        updated_prototypes = self._update_prototypes(model, prototypes)
        avg_epoch_loss = np.mean(epoch_loss_list)
        return model, updated_prototypes, avg_epoch_loss

    # ------------------------------------------------------------------ #
    # Local evaluation
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def local_test(self, model, prototypes, eval_on):
        self._set_model_mode(model, "eval")
        loader = self.test_loader if eval_on == "test" else self.val_loader
        if loader is None:
            return 0.0

        y_true_list, y_pred_list = [], []
        for X, y, observed in loader:
            X = {k: v.to(self.device) for k, v in X.items()}
            y = y.to(self.device).long()
            observed = observed.to(self.device)

            logits, valid_mask = model(
                X, observed, labels=None, prototypes=prototypes, mode="test",
            )
            y_valid = y[valid_mask]
            _, predicted = torch.max(logits, 1)

            y_true_list = np.concatenate((y_true_list, y_valid.cpu().numpy()), axis=0)
            y_pred_list = np.concatenate((y_pred_list, predicted.cpu().numpy()), axis=0)

        f1_macro = f1_score(y_true_list, y_pred_list, average="macro")
        return f1_macro

    # ------------------------------------------------------------------ #
    # Prototype update
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def _update_prototypes(self, model, prototypes):
        """Compute class-mean prototypes from all training data."""
        self._set_model_mode(model, "eval")

        modality_rep_boxes = {mod: None for mod in self.modalities}
        modality_rep_boxes["fusion"] = None
        label_box = None

        for X, y, observed in self.train_loader:
            X = {k: v.to(self.device) for k, v in X.items()}
            y = y.to(self.device).long()
            observed = observed.to(self.device)

            logits, unimodal_embeds, fusion_out, valid_mask = model(
                X, observed, labels=None, prototypes=prototypes,
                return_embeds=True, mode="train",
            )
            y_valid = y[valid_mask]

            for mod_idx, mod_name in enumerate(self.modalities):
                mod_rep = unimodal_embeds[mod_idx].detach()
                if modality_rep_boxes[mod_name] is None:
                    modality_rep_boxes[mod_name] = mod_rep
                else:
                    modality_rep_boxes[mod_name] = torch.cat(
                        [modality_rep_boxes[mod_name], mod_rep], dim=0,
                    )

            fusion_rep = fusion_out.detach()
            if modality_rep_boxes["fusion"] is None:
                modality_rep_boxes["fusion"] = fusion_rep
            else:
                modality_rep_boxes["fusion"] = torch.cat(
                    [modality_rep_boxes["fusion"], fusion_rep], dim=0,
                )

            if label_box is None:
                label_box = y_valid
            else:
                label_box = torch.cat([label_box, y_valid], dim=0)

        num_classes = self.cfg["num_classes"]
        label_one_hot = torch.zeros(label_box.size(0), num_classes, device=self.device)
        label_one_hot.scatter_(1, label_box.unsqueeze(1), 1)

        updated_prototypes = {}
        for mod_name in self.modalities + ["fusion"]:
            proto_sum = torch.matmul(label_one_hot.T, modality_rep_boxes[mod_name])
            count_per_class = label_one_hot.T.sum(dim=1, keepdim=True)
            proto = proto_sum / (count_per_class + 1e-8)
            proto = torch.where(torch.isnan(proto), torch.zeros_like(proto), proto)
            updated_prototypes[mod_name] = proto.detach().cpu()

        return updated_prototypes

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
