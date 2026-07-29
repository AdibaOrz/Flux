# ml_federated/server/PEPSY_Server.py

import copy
import torch
import torch.nn as nn
from ml_federated.client.PEPSY_Client import PEPSY_Client
from ml_federated.server.FedAvg_Server import FedAvg_Server
from models.method_specific.pepsy_nonparametric_agg import NonparametricAgg


class PEPSY_Server(FedAvg_Server):
    """PEPSY Server with non-parametric prompt aggregation."""

    def __init__(self, model, dataset_configs, device, args):
        super().__init__(model, dataset_configs, device, args)
        # PEPSY-specific configurations
        self.num_outer_loops = getattr(args, 'pepsy_num_outer_loops', 10) if args else 10
        self.model.to(self.device)

    def _create_client(self, client_id):
        """Create a PEPSY client instance."""
        return PEPSY_Client(client_id, self.dataset_configs, self.device, self.args)

    def aggregate_models(self, updated_models):
        """
        Aggregate models with PEPSY-specific logic:
        - Standard weighted averaging for most components
        - Non-parametric aggregation for prompts (data-missing profiles)
        """
        client_ids = list(updated_models.keys())
        client_n, client_per_modality_n = self._get_client_data_sizes(client_ids)
        client_state_dicts = {cid: updated_models[cid].state_dict() for cid in client_ids}

        # Set up aggregation
        new_model = copy.deepcopy(self.model)
        global_state_dict = new_model.state_dict()

        # Track which keys have been aggregated
        aggregated_keys = set()
        # Aggregate all components except prompts using standard weighted averaging
        self._aggregate_pepsy_components(client_ids, client_n, client_per_modality_n,
                                         client_state_dicts, global_state_dict, aggregated_keys)

        # Aggregate prompts (non-parametric)
        self._aggregate_prompts_nonparametric(client_ids, updated_models, global_state_dict)
        aggregated_keys.add('pool.prompt')

        # Load weights to the new_model
        # new_model.load_state_dict(global_state_dict) # Since prompt size is dynamic
        # Remove 'pool.prompt' from state_dict since we'll assign it directly (matching original PEPSY)
        # This handles dynamic pool size changes
        prompt_param = global_state_dict.pop('pool.prompt', None)

        # Load weights to the new_model (excluding pool.prompt)
        new_model.load_state_dict(global_state_dict, strict=False)

        # Directly assign the prompt parameter (matching original PEPSY behavior)
        # This allows dynamic pool size changes
        if prompt_param is not None:
            with torch.no_grad():
                new_model.pool.prompt = nn.Parameter(prompt_param, requires_grad=True)
                # Also update trained_prompts_checklist size if it changed
                if new_model.pool.prompt.shape[0] != new_model.trained_prompts_checklist.shape[0]:
                    new_model.trained_prompts_checklist = torch.zeros(
                        new_model.pool.prompt.shape[0],
                        dtype=torch.float32,
                        device=new_model.pool.prompt.device
                    )

        return new_model

    def _aggregate_pepsy_components(self, client_ids, client_n, client_per_modality_n,
                                    client_state_dicts, global_state_dict, aggregated_keys):
        """Aggregate PEPSY-specific components using weighted averaging."""
        total_cnt = sum(client_n[cid] for cid in client_ids)

        # Aggregate modality-specific components (sim_projectors, encoders)
        for m in range(self.num_modalities):
            total_m_cnt = sum(client_per_modality_n[cid][m] for cid in client_ids)
            if total_m_cnt == 0:
                print('⚠️ Warning: No data for modality', m, 'across all clients, skipping its aggregation.')
                continue

            # EMA rate for stability in per-modal aggregation (optional)
            # Default: False (match original PEPSY), can be enabled via args
            use_ema = getattr(self.args, 'pepsy_use_ema_per_mod', False)
            if use_ema:
                r_m = max(total_m_cnt / total_cnt, 0.05)  # Clip at 5% minimum

            sim_proj_key_prefix = f'sim_projectors.{m}.'
            encoder_key_prefix = f'encoders.{m}.'

            for key in global_state_dict.keys():
                if key.startswith(sim_proj_key_prefix):
                    agg = torch.zeros_like(global_state_dict[key])
                    for cid in client_ids:
                        if client_per_modality_n[cid][m] > 0:
                            w = client_per_modality_n[cid][m] / total_m_cnt
                            agg.add_(client_state_dicts[cid][key], alpha=w)
                    if use_ema:
                        global_state_dict[key] = (1 - r_m) * global_state_dict[key] + r_m * agg
                    else:
                        global_state_dict[key] = agg  # Original PEPSY behavior
                    aggregated_keys.add(key)

                elif key.startswith(encoder_key_prefix):
                    # Aggregate encoder parameters using modality-specific weighting
                    agg = torch.zeros_like(global_state_dict[key])
                    for cid in client_ids:
                        if client_per_modality_n[cid][m] > 0:
                            w = client_per_modality_n[cid][m] / total_m_cnt
                            agg.add_(client_state_dicts[cid][key], alpha=w)
                    if use_ema:
                        global_state_dict[key] = (1 - r_m) * global_state_dict[key] + r_m * agg
                    else:
                        global_state_dict[key] = agg  # Original PEPSY behavior
                    aggregated_keys.add(key)

        # Aggregate shared components (classifier, global_fuser, embeddings, pool.features_proj)
        components_to_aggregate = [
            'classifier',
            'global_fuser',
            'embeddings',
            'pool.features_proj'
        ]

        for component_prefix in components_to_aggregate:
            for key in global_state_dict.keys():
                if key.startswith(component_prefix):
                    agg = torch.zeros_like(global_state_dict[key])
                    for cid in client_ids:
                        w = client_n[cid] / total_cnt
                        agg.add_(client_state_dicts[cid][key], alpha=w)
                    global_state_dict[key] = agg
                    aggregated_keys.add(key)

    def _aggregate_prompts_nonparametric(self, client_ids, updated_models, global_state_dict):
        """
        Aggregate prompts using non-parametric aggregation.
        This is the key PEPSY innovation.
        """
        # Collect trained prompts from all clients
        union_prompts_checklist = torch.zeros(
            self.model.pool.prompt.shape[0], dtype=torch.int, device=self.model.pool.prompt.device
        )

        # Build union of trained prompts
        nonzero_index = torch.nonzero(self.model.trained_prompts_checklist).flatten()
        union_prompts_checklist[nonzero_index] = 1

        for cid in client_ids:
            client_model = updated_models[cid]
            nonzero_index = torch.nonzero(client_model.trained_prompts_checklist).flatten()
            union_prompts_checklist[nonzero_index] = 1

        # Extract trained prompts from each client
        trained_prompt_indices = torch.nonzero(union_prompts_checklist).flatten()
        if len(trained_prompt_indices) == 0:
            raise ValueError("Error: No trained prompts found, using standard aggregation")

        local_prompts_list = []
        for cid in client_ids:
            client_model = updated_models[cid]
            # CRITICAL: Detach before cloning to ensure no computation graph connection
            local_prompts = client_model.pool.prompt[trained_prompt_indices].detach().clone()

            # Validate prompts: check for NaN/Inf
            if torch.isnan(local_prompts).any() or torch.isinf(local_prompts).any():
                raise ValueError(f"Error Client {cid} has NaN/Inf in prompts at round {self.current_round}. "
                                 f"NaN count: {torch.isnan(local_prompts).sum().item()}, "
                                 f"Inf count: {torch.isinf(local_prompts).sum().item()}")

            local_prompts_list.append(local_prompts)

        # Stack: (n_clients, n_trained_prompts, prompt_dim)
        local_prompts_tensor = torch.stack(local_prompts_list, dim=0)

        # Non-parametric aggregation
        prompt_dim = local_prompts_tensor.shape[-1]
        agg = NonparametricAgg(prompt_dim, n_hidden=64).to(local_prompts_tensor.device)
        aggregated_prompts = agg(local_prompts_tensor, outer_loop=self.num_outer_loops)
        del agg

        # Note: The aggregated pool size may differ from the original pool size.
        # Update global state dict (allowing dynamic pool size, matching original PEPSY)
        with torch.no_grad():
            global_state_dict['pool.prompt'] = nn.Parameter(aggregated_prompts, requires_grad=True)

        # Update client models' prompts
        for cid in client_ids:
            with torch.no_grad():
                updated_models[cid].pool.prompt = nn.Parameter(aggregated_prompts.clone(), requires_grad=True)

        # Reset trained prompts checklist
        self.model.reset_trained_prompts_checklist()
        for cid in client_ids:
            updated_models[cid].reset_trained_prompts_checklist()

        return aggregated_prompts

