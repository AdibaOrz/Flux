# models/method_specific/flux.py

import torch
import torch.nn as nn


class FLUX_Model(nn.Module):
    def __init__(self, args, encoders, fusion, head, head_input_dim, dataset_cfg):
        super().__init__()
        self.args = args
        self.encoders = nn.ModuleList(encoders)
        self.fusion = fusion
        self.head = head

        self.sensor_to_feat_cols = dataset_cfg['sensor_to_featureColumn']
        self.encoder_out_dim = dataset_cfg['conv_embed_dim']
        self.head_output_dim = dataset_cfg['num_classes']

        self.unimodal_confidence_heads = nn.ModuleList([
            ConfidenceHead(self.args, head_input_dim, dataset_cfg['num_classes'])
            for _ in range(len(self.encoders))
        ])

    def forward(self, inputs, data_presence_mask):
        unimodal_embeddings = []
        modality_presence_mask = []

        unimodal_logits, unimodal_logvars = [], []

        for sensor_idx, feat_cols in self.sensor_to_feat_cols.items():
            present = data_presence_mask[:, feat_cols]
            cur_modality_mask = []
            for i in range(len(inputs)):
                cur_modality_mask.append(0 if torch.all(present[i] == 0) else 1)
            modality_presence_mask.append(cur_modality_mask)

            x_m = inputs[:, feat_cols]
            x_m_encoded = self._encode_unimodal_data(self.encoders[sensor_idx], x_m, cur_modality_mask, self.encoder_out_dim)
            unimodal_embeddings.append(x_m_encoded)

            x_m_encoded_detached = x_m_encoded.detach()
            uni_logits, uni_logvar = self._pass_via_confidence_head(
                self.unimodal_confidence_heads[sensor_idx], x_m_encoded_detached, cur_modality_mask)
            unimodal_logits.append(uni_logits)
            unimodal_logvars.append(uni_logvar)

        modality_presence_mask = torch.tensor(modality_presence_mask).T.to(inputs.device)
        valid_samples_mask = modality_presence_mask.sum(dim=1) > 0

        unimodal_embeddings = [embedding[valid_samples_mask] for embedding in unimodal_embeddings]
        unimodal_logits = [logits[valid_samples_mask] for logits in unimodal_logits]
        unimodal_logvars = [logvar[valid_samples_mask] for logvar in unimodal_logvars]
        modality_presence_mask = modality_presence_mask[valid_samples_mask]

        unimodal_logits = torch.stack(unimodal_logits, dim=0)
        unimodal_logvars = torch.stack(unimodal_logvars, dim=0)

        fused_output, attention_scores = self.fusion(unimodal_embeddings, modality_presence_mask)
        mm_logits = self.head(fused_output)

        return mm_logits, unimodal_logits, fused_output, unimodal_logvars, modality_presence_mask, valid_samples_mask

    def _encode_unimodal_data(self, encoder, x_m, mask, dim):
        x_m_encoded = torch.zeros(x_m.shape[0], dim, device=x_m.device)
        if isinstance(mask, list):
            mask = torch.tensor(mask, dtype=torch.bool, device=x_m.device)
        if mask.any():
            x_m_encoded[mask] = encoder(x_m[mask])
        return x_m_encoded

    def _pass_via_confidence_head(self, confidence_head, x_m_encoded, mask):
        logits = torch.zeros(x_m_encoded.shape[0], self.head_output_dim, device=x_m_encoded.device)
        log_var = torch.zeros(x_m_encoded.shape[0], self.head_output_dim, device=x_m_encoded.device)

        if isinstance(mask, list):
            mask = torch.tensor(mask, dtype=torch.bool, device=x_m_encoded.device)
        if mask.any():
            logits[mask], log_var[mask] = confidence_head(x_m_encoded[mask])
        return logits, log_var


class ConfidenceHead(nn.Module):
    """Per-modality head: logits plus log-variance for confidence tempering (temperature scaling)."""

    def __init__(self, args, input_dim, num_classes, dropout=0.1):
        super().__init__()
        hidden_dim = args.flux_hidden_dim

        self.fe = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.logits_head = nn.Linear(hidden_dim // 2, num_classes)
        self.logvar_head = nn.Linear(hidden_dim // 2, 1)

    def forward(self, x):
        h = self.fe(x)
        logits = self.logits_head(h)
        log_var = self.logvar_head(h)
        log_var = log_var.expand_as(logits)
        return logits, log_var
