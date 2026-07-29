# models/method_specific/pepsy_model.py
# PEPSY Model Implementation

import torch
import torch.nn as nn
from models.method_specific.pepsy_components import (SimilarityProjector, Pool,
                                                     SpecificEmbedding, ContrastiveWeight, AggregationRebuild,
                                                     GlobalFuser,
                                                     Classifier)  # Fusion and classification components from PEPSY


class PEPSYModel(nn.Module):
    """PEPSY Model for federated learning with probabilistic prompts."""

    def __init__(self, encoders, dataset_configs, args=None):
        super(PEPSYModel, self).__init__()
        self.num_modalities = dataset_configs['num_sensors']
        self.hidden_dim = dataset_configs['conv_embed_dim']
        self.encoder_out_dim = dataset_configs['conv_embed_dim']
        self.num_classes = dataset_configs['num_classes']
        self.sensor_to_feat_cols = dataset_configs['sensor_to_featureColumn']

        # Use our unimodal encoders (consistent across baselines; handles multi-channel modalities correctly)
        self.encoders = nn.ModuleList(encoders)

        # PEPSY-specific hyperparameters
        self.top_k = getattr(args, 'pepsy_top_k', 3) if args else 3
        self.pool_size = getattr(args, 'pepsy_pool_size', 10) if args else 10

        # PEPSY's similarity projectors for each modality
        self.sim_projectors = nn.ModuleList()
        for i in range(self.num_modalities):
            self.sim_projectors.append(SimilarityProjector(embed_dim=self.hidden_dim))

        # PEPSY-specific components: embeddings, pooling, classifier, contrastive weight, rebuild, fuser
        self.embeddings = SpecificEmbedding(num_modalities=self.num_modalities, hidden_dim=self.hidden_dim)
        self.pool = Pool(embed_dim=self.hidden_dim, pool_size=self.pool_size, top_k=self.top_k)
        self.classifier = Classifier(num_modalities=self.num_modalities, embed_dim=self.hidden_dim,
                                     num_classes=self.num_classes)
        self.contrastive_weight_module = ContrastiveWeight(num_modalities=self.num_modalities)
        self.rebuilder = AggregationRebuild()
        self.global_fuser = GlobalFuser()

        # Track trained prompts
        self.trained_prompts_checklist = torch.zeros(self.pool.prompt.shape[0], dtype=torch.float32)

    def checking_trained_prompt(self):
        """Track which prompts were used during training."""
        if self.pool.top_k_idx.device != self.trained_prompts_checklist.device:
            self.trained_prompts_checklist = self.trained_prompts_checklist.to(self.pool.top_k_idx.device)
        self.trained_prompts_checklist[self.pool.top_k_idx] += 1.0

    def reset_trained_prompts_checklist(self):
        """Reset the trained prompts checklist."""
        self.trained_prompts_checklist = torch.zeros(self.pool.prompt.shape[0], dtype=torch.float32)

    def forward(self, inputs, data_presence_mask, labels=None, contrastive_weight=0.1):
        batch_size = inputs.shape[0]
        features = torch.zeros(
            size=(batch_size, self.num_modalities, self.hidden_dim),
            dtype=torch.float32,
            device=inputs.device)

        sim_features_intra = torch.zeros(
            size=(batch_size, self.num_modalities, self.hidden_dim),
            dtype=torch.float32,
            device=inputs.device)

        # Reuse MultimodalModel's encoding logic for consistency
        unimodal_outputs = []
        modality_presence_mask = []

        for sensor_idx, feat_cols in self.sensor_to_feat_cols.items():
            present = data_presence_mask[:, feat_cols]  # (batch_size, num_feat_cols, seq_len)
            cur_modality_mask = []
            for i in range(len(inputs)):
                # Mark value as absent if ALL of the features in the modality are missing
                cur_modality_mask.append(0 if torch.all(present[i] == 0) else 1)
            modality_presence_mask.append(cur_modality_mask)

            # Pass through the encoder for the current modality
            x_m = inputs[:, feat_cols]  # (batch_size, num_feat_cols, seq_len)
            x_m_encoded = self._encode_unimodal_data(self.encoders[sensor_idx], x_m, cur_modality_mask)
            unimodal_outputs.append(x_m_encoded)

        # PEPSY-specific processing
        # Convert to tensor: (batch_size, num_modalities) where 1=present, 0=missing
        modality_presence_mask = torch.tensor(modality_presence_mask).T.to(inputs.device)

        valid_samples = modality_presence_mask.sum(dim=1) > 0  # (batch_size, )
        # Valid samples should be equal to batch_size during training, check it
        if torch.any(~valid_samples):
            raise ValueError("Some samples have all modalities missing!")

        # PEPSY-specific: Convert presence mask to missing mask and reshape
        # missing_mask_total: (batch_size, num_modalities, 1) where 1=missing, 0=present
        missing_mask_total = (1 - modality_presence_mask).unsqueeze(-1)  # (batch_size, num_modalities, 1)

        # Store encoded features in the features tensor
        for modality_idx in range(self.num_modalities):
            features[:, modality_idx:(modality_idx + 1), :] = unimodal_outputs[modality_idx].unsqueeze(1)

        # Impute missing modalities with mean of present modalities
        denominator = torch.sum(1 - missing_mask_total, dim=1, keepdim=True)  # (batch_size, 1, 1)
        if torch.any(denominator == 0):
            raise ValueError("Denominator has zeros, indicating all modalities are missing for some samples.")

        mean_shared = torch.sum(features, dim=1, keepdim=True) / denominator  # (batch_size, 1, hidden_dim)
        features = features * (1 - missing_mask_total) + mean_shared * missing_mask_total

        ori_features_intra = features.permute(1, 0, 2).reshape(-1,
                                                               self.hidden_dim)  # (num_modalities * batch_size, hidden_dim)

        # Add specific embeddings
        specific_emb = self.embeddings(inputs)  # (1, num_modalities, hidden_dim)
        specific_emb = specific_emb * torch.ones_like(features).to(inputs.device)
        features = torch.cat([specific_emb, features], dim=-1)  # (batch_size, num_modalities, 2 * hidden_dim)

        # Add global prompts
        global_prompts = []
        reduced_sims = 0.0
        for modality_idx in list(range(self.num_modalities)):
            query = features[:, modality_idx:(modality_idx + 1), :]  # (batch_size, 1, 2 * hidden_dim)
            reduce_sim, batched_prompts = self.pool(query.squeeze(1), cls_features=None)
            reduced_sims += reduce_sim
            global_prompts.append(torch.mean(batched_prompts, dim=1, keepdim=True))  # (batch_size, 1, hidden_dim)

        global_prompts = torch.cat(global_prompts, dim=1)  # (batch_size, num_modalities, hidden_dim)
        features = torch.cat([features, global_prompts], dim=-1)  # (batch_size, num_modalities, 3 * hidden_dim)
        reduced_sims = reduced_sims / self.num_modalities

        # Project features through similarity projectors
        for modality_idx in list(range(self.num_modalities)):
            dense_feature = self.sim_projectors[modality_idx](features[:, modality_idx:(modality_idx + 1), :])
            sim_features_intra[:, modality_idx:(modality_idx + 1), :] = dense_feature

        sim_features_intra = sim_features_intra.permute(1, 0, 2).reshape(-1,
                                                                         self.hidden_dim)  # (num_modalities * batch_size, hidden_dim)

        # Modality shared contrastive loss
        intra_loss, intra_similarity_matrix, _, _ = self.contrastive_weight_module(sim_features_intra)
        ori_intra_loss, ori_intra_similarity_matrix, _, _ = self.contrastive_weight_module(ori_features_intra)
        contrastive_loss = (intra_loss * 1.0 + ori_intra_loss * 0.5)

        # Reaggregation to ensure consistencies
        _, rebuilt_features = self.rebuilder(
            intra_similarity_matrix[:batch_size * self.num_modalities, :batch_size * self.num_modalities],
            features.reshape(-1, self.hidden_dim * 3)
        )

        rebuilt_features, alpha_old = self.global_fuser(
            features.reshape(batch_size, -1, self.hidden_dim),
            rebuilt_features.reshape(batch_size, -1, self.hidden_dim)
        )

        # Classifier head
        outputs = self.classifier(rebuilt_features.reshape(batch_size, -1))

        # Compute loss (classification loss will be computed outside with labels)
        # For now, we return the components needed for loss computation
        self.checking_trained_prompt()

        # If labels provided, compute total loss
        if labels is not None:
            criterion = nn.CrossEntropyLoss()
            cls_loss = criterion(outputs, labels.long())
            total_loss = cls_loss + contrastive_weight * contrastive_loss + 0.1 * reduced_sims
            return outputs, total_loss
        else:
            # For inference, return logits and valid_samples_mask (standard interface)
            return outputs

    def _encode_unimodal_data(self, encoder, x_m, mask):
        """
        Encodes only present unimodal data using the specified modality-specific encoder.
            encoder: modality-specific encoder
            x_m: (batch_size, num_cols {e.g., 1 for EDA, 3 for ACC}, feat_dim) tensor for the current modality
            mask: (batch_size, ) tensor indicating presence of the current modality
        """
        # Output tensor with zeros initially
        x_m_encoded = torch.zeros(x_m.shape[0], self.encoder_out_dim, device=x_m.device)

        if isinstance(mask, list):
            mask = torch.tensor(mask, dtype=torch.bool, device=x_m.device)
        if mask.any():  # only encode samples where the modality is present
            x_m_encoded[mask] = encoder(x_m[mask])  # encode only present samples

        # print(f'mask: {mask}, Num present: {sum(mask)}, num absent: {sum(~mask)}')
        # print(f'x_m.shape: {x_m.shape}, x_m[mask].shape: {x_m[mask].shape}, x_m[~mask].shape: {x_m[~mask].shape}')
        return x_m_encoded
