# ml_federated/server/PmcmFL_Server.py

import copy
import torch
import wandb
import numpy as np
from tqdm import trange
from ml_federated.server.FedAvg_Server import FedAvg_Server
from ml_federated.client.PmcmFL_Client import PmcmFL_Client


class PmcmFL_Server(FedAvg_Server):
    """PmcmFL Server."""

    def __init__(self, model, dataset_configs, device, args):
        super().__init__(model, dataset_configs, device, args)
        self.model.to(self.device)
        self.num_modalities = dataset_configs['num_sensors']

        # Define prototypes for each modality and fusion
        self.prototypes = {}
        for i in range(self.num_modalities):
            modality_key = dataset_configs['sensor_list'][i]
            self.prototypes[modality_key] = torch.randn(
                [dataset_configs['num_classes'], dataset_configs['conv_embed_dim']], dtype=torch.float32,
                device=self.device)
        self.prototypes["fusion"] = torch.randn([dataset_configs['num_classes'], dataset_configs['conv_embed_dim']],
                                                dtype=torch.float32, device=self.device)

    def _create_client(self, client_id):
        """Create a FedAvg client instance."""
        return PmcmFL_Client(client_id, self.dataset_configs, self.device, self.args)

    def run_federated_training(self):
        """ Federated learning process. In each round:
            1. Sample clients
            2. Let selected clients train their local models
            3. Aggregate the local model updates
            4. Eval the global model on each client's val set (every N rounds)
        """
        for rnd in trange(1, self.global_rounds + 1, desc=f'Training FL ({self.method.upper()})'):
            self.perf_stats[rnd] = {}
            self.current_round = rnd
            # 1. Sample clients
            selected_clients = self.select_clients()
            self.perf_stats[rnd]['selected_clients'] = selected_clients

            # 2. Local train
            losses = []
            updated_models = {}  # (client_id: updated_model)
            updated_prototypes = {}  # (client_id: updated_prototypes)
            for cid in selected_clients:
                client = self.clients[cid]
                updated_model, updated_prototype, loss = client.local_train(copy.deepcopy(self.model),
                                                                            copy.deepcopy(self.prototypes))
                updated_models[cid] = updated_model
                updated_prototypes[cid] = updated_prototype
                losses.append(loss)

            # 3. Aggregate trained models
            self.model = self.aggregate_models(updated_models)
            self.prototypes = self.aggregate_prototypes(updated_prototypes)

            # Log server's average loss for the round
            round_loss = np.mean(losses)
            self.perf_stats[rnd]['round_loss'] = round_loss
            wandb.log({'round_loss': round_loss}, step=rnd) if self.args.wandb else None

            # 4. Evaluate the global model on each client's validation set
            val_f1_str = 'Calculation skipped'
            if rnd == 1 or rnd % self.eval_every == 0:
                self.run_federated_evaluation(eval_on='val')
                val_f1_str = '{:.4f} ± {:.4f}'.format(self.perf_stats[rnd].get('val_f1_mean', 'N/A'),
                                                      self.perf_stats[rnd].get('val_f1_std', 'N/A'))

                # Perform global model selection if applicable (e.g., FedAvg, FedProx)
                if self._should_use_global_model_selection():
                    self._perform_global_model_selection(rnd)
            # Log to console
            print(f'Round {rnd} - Loss: {round_loss:.4f} | Val F1: {val_f1_str}')

    def run_federated_evaluation(self, eval_on):
        """ Final evaluation of the global model"""
        # Use the best model if eval_on is 'test', otherwise use the current model
        if eval_on == 'test' and self._should_use_global_model_selection():
            evaluation_model = self.best_model
        else:
            evaluation_model = self.model
        all_client_f1s = {}
        self.perf_stats[self.current_round][f'client_{eval_on}_f1'] = {}
        for cid in self.client_ids:
            client = self.clients[cid]
            f1 = client.local_test(copy.deepcopy(evaluation_model), copy.deepcopy(self.prototypes), eval_on)
            all_client_f1s[cid] = f1
            self.perf_stats[self.current_round][f'client_{eval_on}_f1'][cid] = f1
            wandb.log({f'{self.dataset_name}/{eval_on}_f1_{cid}': f1},
                      step=self.current_round) if self.args.wandb else None

        # Calculate and log mean & std F1 scores across all clients
        mean_all_f1 = np.mean([f1 for f1 in all_client_f1s.values()])
        std_all_f1 = np.std([f1 for f1 in all_client_f1s.values()])
        self.perf_stats[self.current_round][f'{eval_on}_f1_mean'] = mean_all_f1
        self.perf_stats[self.current_round][f'{eval_on}_f1_std'] = std_all_f1

        wandb.log({f'{eval_on}_f1_mean': mean_all_f1, f'{eval_on}_f1_std': std_all_f1},
                  step=self.current_round) if self.args.wandb else None

        if eval_on == 'test':
            print("====================== Final Test Evaluation ====================")
            print(f'Final Test F1 across all clients: {mean_all_f1:.4f} ± {std_all_f1:.4f}')
            wandb.log({'test_f1': mean_all_f1}) if self.args.wandb else None

    def aggregate_prototypes(self, updated_prototypes):
        """
        Aggregate local prototypes from all clients.

        Args:
            updated_prototypes: dict with client_id as key and prototype dict as value
                           Each prototype dict contains modality-specific prototypes
        """
        if not updated_prototypes:
            return self.prototypes

        # Get client IDs and their data sizes for weighting
        client_ids = list(updated_prototypes.keys())
        client_n, client_per_modality_n = self._get_client_data_sizes(client_ids)

        # Initialize global prototypes
        global_prototypes = {}
        for modality in self.prototypes.keys():
            global_prototypes[modality] = torch.zeros_like(self.prototypes[modality], device='cpu')

        # Calculate aggregation weights based on data sizes
        total_samples_per_modality = {}
        for modality_idx, modality in enumerate(self.prototypes.keys()):
            if modality == 'fusion':
                # For fusion modality, use total data size
                total_samples_per_modality[modality] = sum(client_n.values())
            else:
                # For unimodal modalities, sum the specific modality counts
                total_samples_per_modality[modality] = sum(
                    client_per_modality_n[cid][modality_idx] for cid in client_ids
                )

        # Aggregate prototypes with weighted averaging
        for client_id in client_ids:
            client_prototypes = updated_prototypes[client_id]

            for modality_idx, modality in enumerate(self.prototypes.keys()):
                if modality in client_prototypes:
                    # Calculate weight for this client's modality
                    if modality == 'fusion':
                        client_weight = client_n.get(client_id, 0)
                    else:
                        client_weight = client_per_modality_n[client_id][modality_idx]

                    total_weight = total_samples_per_modality[modality]
                    if total_weight > 0:
                        weight = client_weight / total_weight
                        global_prototypes[modality] += client_prototypes[modality] * weight

        # Handle zero prototypes (clients that don't have certain modalities)
        zero_proto_unimodal = torch.zeros([1, self.dataset_configs['conv_embed_dim']],
                                          device='cpu', dtype=torch.float32)
        zero_proto_fusion = torch.zeros([1, self.dataset_configs['conv_embed_dim']],
                                        device='cpu', dtype=torch.float32)

        # Track total weights per class for each modality
        total_weight_per_class = {}
        for modality in self.prototypes.keys():
            num_classes = self.dataset_configs['num_classes']
            total_weight_per_class[modality] = torch.zeros(num_classes, device='cpu')

        # Calculate total weights per class for each modality
        for client_id in client_ids:
            client_prototypes = updated_prototypes[client_id]

            for modality_idx, modality in enumerate(self.prototypes.keys()):
                if modality in client_prototypes:
                    # Calculate weight for this client's modality
                    if modality == 'fusion':
                        client_weight = client_n.get(client_id, 0)
                    else:
                        client_weight = client_per_modality_n[client_id][modality_idx]

                    total_weight = total_samples_per_modality[modality]
                    if total_weight > 0:
                        weight = client_weight / total_weight

                        # Check which classes have non-zero prototypes
                        zero_proto = zero_proto_fusion if modality == 'fusion' else zero_proto_unimodal
                        for clas in range(self.dataset_configs['num_classes']):
                            if not torch.equal(client_prototypes[modality][clas:clas + 1, :], zero_proto):
                                total_weight_per_class[modality][clas] += weight

        # Normalize by total weights per class
        for modality in self.prototypes.keys():
            for clas in range(self.dataset_configs['num_classes']):
                if total_weight_per_class[modality][clas] > 1e-4:  # Avoid division by very small numbers
                    global_prototypes[modality][clas:clas + 1, :] /= total_weight_per_class[modality][clas]

        return global_prototypes

    def aggregate_models(self, updated_models):
        # Prepare data
        client_ids = list(updated_models.keys())
        client_n, client_per_modality_n = self._get_client_data_sizes(client_ids)
        client_state_dicts = {cid: updated_models[cid].state_dict() for cid in client_ids}

        # Set up aggregation
        new_model = copy.deepcopy(self.model)
        global_state_dict = new_model.state_dict()
        encoder_keys, non_encoder_keys = self._partition_model_parameters(global_state_dict)

        # Aggregate encoder params
        self._aggregate_encoder_parameters(client_ids, client_n, client_per_modality_n, client_state_dicts,
                                           encoder_keys, global_state_dict)
        # Aggregate non-encoder params
        self._aggregate_non_encoder_parameters(client_ids, client_n, client_state_dicts,
                                               non_encoder_keys, global_state_dict)
        # Load weights to the new_model
        new_model.load_state_dict(global_state_dict)
        return new_model