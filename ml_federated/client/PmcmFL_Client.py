# ml_federated/client/PmcmFL_Client.py

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from ml_federated.client.FedAvg_Client import FedAvg_Client


class PmcmFL_Client(FedAvg_Client):
    """Client for PmcmFL"""

    def __init__(self, client_id, dataset_configs, device, args):
        super().__init__(client_id, dataset_configs, device, args)
        self.num_modalities = dataset_configs['num_sensors']
        self.sensor_to_feat_cols = dataset_configs['sensor_to_featureColumn']
        self.unimodal_embed_dim = dataset_configs['conv_embed_dim']
        self.fusion_embed_dim = dataset_configs['conv_embed_dim']
        self.sensor_list = dataset_configs['sensor_list']
        self.device = device
        self.clip_criterion = ClipLossHalf()
        self.clip_logits_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07)).to(self.device)
        self.args = args

    def local_train(self, model, prototypes):
        self._set_model_mode(model, mode='train')

        optimizer = self._get_optimizer(model)
        epoch_loss_list = []
        for epoch in range(self.local_epochs):
            batch_loss_list = []
            for x, mask, y in self.train_loader:
                x, mask, y = x.to(self.device), mask.to(self.device), y.to(self.device).long()
                optimizer.zero_grad()

                # Replace unimodal outputs with prototypes for missing modalities
                model_outputs, _, fusion_output, _, _, valid_samples_mask = model(x, y, mask, prototypes,
                                                                                  return_embeds=True)
                y = y[valid_samples_mask]
                ce_loss = self.criterion(model_outputs, y)

                fusion_prototype = prototypes["fusion"].to(self.device)
                # Calculate CLIP loss
                clip_loss = self.clip_criterion(fusion_output, fusion_prototype[y], self.clip_logits_scale.exp())
                loss = ce_loss + clip_loss * self.dataset_configs['pmcmfl_clip_loss_weight']

                loss.backward()
                optimizer.step()
                batch_loss_list.append(loss.item())
            # print(f'Client {self.client_id} Epoch {epoch} Loss: {np.mean(batch_loss_list)}')
            epoch_loss_list.append(np.mean(batch_loss_list))

        updated_prototypes = self._update_prototypes(model, prototypes)
        avg_epoch_loss = np.mean(epoch_loss_list)
        return model, updated_prototypes, avg_epoch_loss

    @torch.no_grad()
    def local_test(self, model, prototypes, eval_on):
        self._set_model_mode(model, mode='eval')

        eval_loader = self.test_loader if eval_on == 'test' else self.val_loader
        y_true_list, y_pred_list = [], []
        for x, mask, y in eval_loader:
            x, mask, y = x.to(self.device), mask.to(self.device), y.to(self.device).long()
            model_outputs, _, valid_samples_mask = model(x, None, mask, prototypes, type='test')
            y = y[valid_samples_mask]
            _, predicted = torch.max(model_outputs, 1)
            predicted = predicted.view(-1)

            y_true_list = np.concatenate((y_true_list, y.cpu()), axis=0)
            y_pred_list = np.concatenate((y_pred_list, predicted.cpu()), axis=0)

        f1_macro = f1_score(y_true_list, y_pred_list, average='macro')
        return f1_macro

    def _update_prototypes(self, model, prototypes):
        self._set_model_mode(model, mode='eval')
        updated_prototypes = {}

        # Initialize containers for each modality
        modality_rep_boxes = {modality: None for modality in self.sensor_list}
        modality_rep_boxes['fusion'] = None
        label_box = None

        with torch.no_grad():
            for x, mask, y in self.train_loader:
                x, mask, y = x.to(self.device), mask.to(self.device), y.to(self.device).long()
                model_outputs, unimodal_outputs, fusion_output, _, _, valid_samples_mask = model(x, None, mask,
                                                                                                 prototypes,
                                                                                                 return_embeds=True)
                y = y[valid_samples_mask]

                # Collect representations for each modality
                for modality_idx, modality_name in enumerate(self.sensor_list):
                    modality_rep = unimodal_outputs[modality_idx].clone().detach()
                    if modality_rep_boxes[modality_name] is None:
                        modality_rep_boxes[modality_name] = modality_rep
                    else:
                        modality_rep_boxes[modality_name] = torch.cat([modality_rep_boxes[modality_name], modality_rep],
                                                                      dim=0)

                # Collect targets and labels
                if label_box is None:
                    label_box = y
                else:
                    label_box = torch.cat([label_box, y], dim=0)

                if modality_rep_boxes['fusion'] is None:
                    modality_rep_boxes['fusion'] = fusion_output.clone().detach()
                else:
                    modality_rep_boxes['fusion'] = torch.cat([modality_rep_boxes['fusion'], fusion_output], dim=0)

        # Create one-hot encoding for labels
        num_classes = len(self.dataset_configs['classes'])
        label_one_hot = torch.zeros(label_box.size(0), num_classes, device=self.device)
        label_one_hot.scatter_(1, label_box.unsqueeze(1), 1)

        unimodal_zero_prototype = torch.zeros([1, self.unimodal_embed_dim], device=self.device, dtype=torch.float32)
        fusion_zero_prototype = torch.zeros([1, self.fusion_embed_dim], device=self.device, dtype=torch.float32)

        # Calculate prototypes for each modality
        for modality_name in self.sensor_list + ['fusion']:
            # Calculate prototype sum and total weight per class
            modality_prototype_sum = torch.matmul(label_one_hot.T, modality_rep_boxes[modality_name])
            total_weight_per_class = torch.sum(label_one_hot.T, dim=1, keepdim=True)

            # Calculate prototypes (avoid division by zero)
            modality_prototypes = modality_prototype_sum / (total_weight_per_class + 1e-8)

            # Replace NaN values with zeros
            modality_prototypes = torch.where(torch.isnan(modality_prototypes),
                                              torch.full_like(modality_prototypes, 0.0),
                                              modality_prototypes)

            updated_prototypes[modality_name] = modality_prototypes.detach().cpu()

            for clas in range(num_classes):
                if not torch.equal(modality_prototypes[clas:clas + 1, :],
                                   unimodal_zero_prototype) and modality_name != 'fusion':
                    updated_prototypes[modality_name][clas:clas + 1, :] = modality_prototypes[clas:clas + 1, :]
                elif not torch.equal(modality_prototypes[clas:clas + 1, :],
                                     fusion_zero_prototype) and modality_name == 'fusion':
                    updated_prototypes["fusion"][clas:clas + 1, :] = modality_prototypes[clas:clas + 1, :]

        return updated_prototypes


class ClipLossHalf(nn.Module):
    def __init__(self, cache_labels=False):
        super().__init__()
        self.cache_labels = cache_labels
        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def forward(self, image_features, text_features, logit_scale):
        device = image_features.device
        logits_per_image = logit_scale * image_features @ text_features.T

        # calculated ground-truth and cache if enabled
        num_logits = logits_per_image.shape[0]
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]
        loss = F.cross_entropy(logits_per_image, labels)
        return loss