import torch
import torch.nn as nn


class PmcmFLModel(nn.Module):
    def __init__(self, encoders, fusion, head, dataset_configs):
        super(PmcmFLModel, self).__init__()
        self.encoders = nn.ModuleList(encoders)  # list of encoder modules
        self.fusion = fusion  # fusion module (e.g., Attention)
        self.head = head  # classification head (e.g., MLP)
        self.sensor_to_feat_cols = dataset_configs['sensor_to_featureColumn']
        self.encoder_out_dim = dataset_configs['conv_embed_dim']
        self.num_modalities = dataset_configs['num_sensors']
        self.sensor_list = dataset_configs['sensor_list']

    def forward(self, inputs, labels, data_presence_mask, prototypes,
                return_embeds=False, type='train'):
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
                # Mark value as absent if any of the features in the modality are missing
                cur_modality_mask.append(0 if torch.all(present[i] == 0) else 1)
            modality_presence_mask.append(cur_modality_mask)

            # Pass through the encoder for the current modality
            x_m = inputs[:, feat_cols]  # current modality input
            x_m_encoded = self._encode_unimodal_data(self.encoders[sensor_idx], x_m, cur_modality_mask)
            unimodal_outputs.append(x_m_encoded)
        modality_presence_mask = torch.tensor(modality_presence_mask).T.to(
            inputs.device)  # shape: [batch_size, num_modalities]

        # PmcmFL specific
        missing_modality_mask = self._get_missing_modality_mask(data_presence_mask)  # [batch_size, num_modalities]

        # Training
        if labels is not None:
            for modality_idx in range(self.num_modalities):
                if missing_modality_mask[:, modality_idx].any():  # if there are samples with missing modality
                    modality_key = self.sensor_list[modality_idx]
                    prototypes[modality_key] = prototypes[modality_key].to(labels.device)

                    # Replace only missing samples with prototypes
                    missing_indices = missing_modality_mask[:, modality_idx]
                    unimodal_outputs[modality_idx][missing_indices] = prototypes[modality_key][labels[missing_indices]]

        # Inference
        if labels is None and type == 'test':

            available_modality_mask = ~missing_modality_mask  # [batch_size, num_modalities]

            # Calculate closest class for each sample
            closest_classes = self._get_closest_class_per_sample(unimodal_outputs, prototypes, available_modality_mask)

            # Apply prototypes for missing modalities for each sample
            for modality_idx in range(self.num_modalities):
                if missing_modality_mask[:, modality_idx].any():
                    modality_key = self.sensor_list[modality_idx]
                    prototypes[modality_key] = prototypes[modality_key].to(closest_classes.device)

                    # Replace missing samples with prototypes of corresponding class
                    missing_indices = missing_modality_mask[:, modality_idx]
                    unimodal_outputs[modality_idx][missing_indices] = prototypes[modality_key][
                        closest_classes[missing_indices]]

        # Check if there is any sample with all modalities absent
        # Exclude the samples where all modalities are absent
        valid_samples_mask = modality_presence_mask.sum(dim=1) > 0
        # print(f'Valid samples mask: {valid_samples_mask}, Num valid samples: {valid_samples_mask.sum()}')
        unimodal_outputs = [output[valid_samples_mask] for output in unimodal_outputs]
        modality_presence_mask = modality_presence_mask[valid_samples_mask]

        fusion_output, att_scores = self.fusion(unimodal_outputs, modality_presence_mask)
        logits = self.head(fusion_output)

        if return_embeds:
            return logits, unimodal_outputs, fusion_output, modality_presence_mask, att_scores, valid_samples_mask
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

    def _get_missing_modality_mask(self, mask):
        """
        Create a mask indicating which modality is missing for each sample in a vectorized manner
        Returns: [batch_size, num_modalities] boolean tensor
        """
        batch_size = mask.shape[0]
        missing_mask = torch.zeros(batch_size, self.num_modalities, dtype=torch.bool, device=mask.device)

        for modality_idx in range(self.num_modalities):
            feat_cols = self.sensor_to_feat_cols[modality_idx]
            curr_modality_mask = mask[:, feat_cols, :]  # [batch_size, num_feat_cols, seq_len]
            # Missing if all features and all time steps of the modality are 0
            missing_mask[:, modality_idx] = torch.all(torch.all(curr_modality_mask == 0, dim=2), dim=1)

        return missing_mask

    def _get_closest_class_per_sample(self, unimodal_outputs, prototypes, available_modality_mask):
        """
        Find the closest class for each sample based on average embedding of available modalities
        """
        batch_size = available_modality_mask.shape[0]
        device = available_modality_mask.device

        closest_classes = torch.zeros(batch_size, dtype=torch.long, device=device)

        for sample_idx in range(batch_size):
            # Available modalities for the current sample
            available_modalities = torch.where(available_modality_mask[sample_idx])[0]

            # Average embedding of available modalities
            available_embeddings = [unimodal_outputs[mod_idx][sample_idx] for mod_idx in available_modalities]
            avg_embedding = torch.stack(available_embeddings).mean(dim=0)  # [embedding_dim]

            # Average prototype of available modalities
            available_prototypes = [prototypes[self.sensor_list[mod_idx]].to(device) for mod_idx in
                                    available_modalities]
            avg_prototype = torch.stack(available_prototypes).mean(dim=0)  # [num_classes, embedding_dim]

            # Calculate distances
            distances = torch.norm(avg_embedding.unsqueeze(0) - avg_prototype, dim=1)  # [num_classes]
            closest_classes[sample_idx] = distances.argmin()

        return closest_classes

