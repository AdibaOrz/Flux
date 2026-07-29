# models/multimodal_model.py

import torch
import torch.nn as nn




class MultimodalModel(nn.Module):
    def __init__(self, encoders, fusion, head, dataset_configs, moon_prj_head=None):
        super(MultimodalModel, self).__init__()
        self.encoders = nn.ModuleList(encoders)  # list of encoder modules
        self.fusion = fusion  # fusion module (e.g., Attention)
        self.head = head  # classification head (e.g., MLP)
        self.sensor_to_feat_cols = dataset_configs['sensor_to_featureColumn']
        self.encoder_out_dim = dataset_configs['conv_embed_dim']

        # Projection head for MOON
        self.moon_prj_head = moon_prj_head if moon_prj_head is not None else None

    def forward(self, inputs, data_presence_mask,
                return_unimodal_embeds=False, return_binary_mask=False, return_projection_output=False,
                return_for_fedrod=False, return_for_dualroute=False):
        """
        Parameters
        ----------
        inputs             : list of tensors, one per sensor modality.
                             Each tensor should have shape [batch_size, seq_len, feat_dim]
        data_presence_mask : mask annotating existing and missing values in the given sample
        return_unimodal_embeds   : bool, if True  -> return (logits, unimodal_outputs, modality_presence_mask, att_scores)
                                   if False -> return (logits, att_scores)
        """
        unimodal_outputs = []
        modality_presence_mask = []

        for sensor_idx, feat_cols in self.sensor_to_feat_cols.items():
            present = data_presence_mask[:, feat_cols]  # current modality presence mask
            cur_modality_mask = []
            for i in range(len(inputs)):
                # Mark value as absent if ALL of the features in the modality are missing
                cur_modality_mask.append(0 if torch.all(present[i] == 0) else 1)
            modality_presence_mask.append(cur_modality_mask)

            # Pass through the encoder for the current modality
            x_m = inputs[:, feat_cols]  # current modality input
            x_m_encoded = self._encode_unimodal_data(self.encoders[sensor_idx], x_m, cur_modality_mask)
            unimodal_outputs.append(x_m_encoded)
        modality_presence_mask = torch.tensor(modality_presence_mask).T.to(
            inputs.device)  # shape: [batch_size, num_modalities]

        # Check if there is any sample with all modalities absent
        # Exclude the samples where all modalities are absent
        valid_samples_mask = modality_presence_mask.sum(dim=1) > 0
        # print(f'Valid samples mask: {valid_samples_mask}, Num valid samples: {valid_samples_mask.sum()}')
        unimodal_outputs = [output[valid_samples_mask] for output in unimodal_outputs]
        modality_presence_mask = modality_presence_mask[valid_samples_mask]

        fusion_output, att_scores = self.fusion(unimodal_outputs, modality_presence_mask)

        logits = self.head(fusion_output)

        if return_unimodal_embeds:
            return logits, unimodal_outputs, modality_presence_mask, att_scores, valid_samples_mask
        if return_binary_mask:
            return logits, att_scores, modality_presence_mask, valid_samples_mask
        if return_projection_output:
            projected_fusion_output = self.moon_prj_head(fusion_output)
            return logits, projected_fusion_output, att_scores, valid_samples_mask
        elif return_for_fedrod:
            return logits, unimodal_outputs, modality_presence_mask, fusion_output, valid_samples_mask
        elif return_for_dualroute:
            return logits, unimodal_outputs, modality_presence_mask, fusion_output, valid_samples_mask

        return logits, att_scores, valid_samples_mask

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

