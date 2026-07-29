# ml_federated/server/FedAvg_Server.py

import copy
import torch
import wandb
import numpy as np
from tqdm import trange
from collections import defaultdict
from utils.pretty_print import print_header
from ml_federated.server.Base_Server import Base_Server
from ml_federated.client.FedAvg_Client import FedAvg_Client



class FedAvg_Server(Base_Server):
    """FedAvg Server - the most fundamental federated learning algorithm."""
    def __init__(self, model,dataset_configs, device, args):
        super().__init__(model, dataset_configs, device, args)
        # Any additional FedAvg-specific initialization can go here
        self.model.to(self.device)

    def _create_client(self, client_id):
        """Create a FedAvg client instance."""
        return FedAvg_Client(client_id, self.dataset_configs, self.device, self.args)


    def run(self):
        print_header(f'{self.__class__.__name__} is running...')
        self.run_federated_training()
        self.run_federated_evaluation(eval_on='test')


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
            for cid in selected_clients:
                client = self.clients[cid]
                updated_model, loss = client.local_train(copy.deepcopy(self.model))
                updated_models[cid] = updated_model
                losses.append(loss)

            # 3. Aggregate trained models
            self.model = self.aggregate_models(updated_models)

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


    def _perform_global_model_selection(self, rnd):
        cur_val_f1 = self.perf_stats[rnd].get('val_f1_mean', 0.0)
        if cur_val_f1 > self.best_val_f1:
            self.best_val_f1 = cur_val_f1
            self.best_model = copy.deepcopy(self.model)
            self.perf_stats[rnd]['best_model_updated'] = True
            print(f'New best model found at round {rnd} with val F1: {cur_val_f1:.4f}')


    def _should_use_global_model_selection(self):
        # Determine if global model selection is used. Override in subclass if needed.
        # E.g., FedAvg and FedProx use the global model selection, FedPer does not.
        return True


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
            f1 = client.local_test(copy.deepcopy(evaluation_model), eval_on)
            all_client_f1s[cid] = f1
            self.perf_stats[self.current_round][f'client_{eval_on}_f1'][cid] = f1
            wandb.log({f'{self.dataset_name}/{eval_on}_f1_{cid}': f1}, step=self.current_round) if self.args.wandb else None

        # Calculate and log mean & std F1 scores across all clients
        mean_all_f1 = np.mean([f1 for f1 in all_client_f1s.values()])
        std_all_f1  = np.std([f1 for f1 in all_client_f1s.values()])
        self.perf_stats[self.current_round][f'{eval_on}_f1_mean'] = mean_all_f1
        self.perf_stats[self.current_round][f'{eval_on}_f1_std'] = std_all_f1

        wandb.log({f'{eval_on}_f1_mean': mean_all_f1, f'{eval_on}_f1_std': std_all_f1},
                  step=self.current_round) if self.args.wandb else None

        if eval_on == 'test':
            print("====================== Final Test Evaluation ====================")
            print(f'Final Test F1 across all clients: {mean_all_f1:.4f} ± {std_all_f1:.4f}')
            wandb.log({'test_f1': mean_all_f1}) if self.args.wandb else None


    def select_clients(self):
        selected_client_list = np.random.choice(self.client_ids, size=self.num_selected_clients, replace=False)
        return selected_client_list


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


    def _get_client_data_sizes(self, client_ids):
        # Extract client data sizes for weighting
        client_n = {}
        client_per_modality_n = {}
        for cid in client_ids:
            client = self.clients[cid]
            client_n[cid] = client.get_data_size('train')
            client_per_modality_n[cid] = client.get_modality_data_size('train')
        return client_n, client_per_modality_n


    def _partition_model_parameters(self, global_state_dict):
        # Partition model parameters into encoder and non-encoder keys
        encoder_keys = defaultdict(list)
        non_encoder_keys = []

        for key in global_state_dict.keys():
            if key.startswith('encoder'):
                parts = key.split('.')
                encoder_idx = int(parts[1])
                assert encoder_idx < self.num_modalities, f"Encoder index {encoder_idx} out of range"
                encoder_keys[encoder_idx].append(key)
            else:
                non_encoder_keys.append(key)

        self._assert_model_param_keys(encoder_keys, non_encoder_keys, global_state_dict)
        return encoder_keys, non_encoder_keys


    def _aggregate_encoder_parameters(self, client_ids, client_n, client_per_modality_n,
                                      client_state_dicts, encoder_keys, global_state_dict):
        # Aggregate encoder params using modality-specific data size weighting
        total_all = sum(client_n.values())

        for m in range(self.num_modalities):
            keys = encoder_keys.get(m, [])
            total_m_cnt = sum(client_per_modality_n[c][m] for c in client_ids)
            if not keys or total_m_cnt == 0:
                # print(f'[Agg] Skipping modality {m} aggregation due to no clients or zero samples')
                continue

            keys_to_aggregate = [k for k in keys if self._should_aggregate_parameter(k)]
            if not keys_to_aggregate:
                continue

            # EMA rate for stability in per-modal aggregation
            # To avoid bias towards a small number of clients with such a modality, apply EMA for modality aggregation
            r_m = max(total_m_cnt / total_all, 0.05)  # Clip at 5% so that rare-modality encoders are not fossilized

            for k in keys_to_aggregate:
                agg = torch.zeros_like(global_state_dict[k])
                contrib_str = ""
                for cid in client_ids:
                    w = client_per_modality_n[cid][m] / total_m_cnt
                    agg.add_(client_state_dicts[cid][k], alpha=w)
                    contrib_str += f"{cid}: {w:.4f} | "

                if self.args.disable_ema_per_mod:
                    global_state_dict[k] = agg
                else:  # By default, apply EMA for per-modal aggregation for stability
                    # print(f'[Agg] [Key={k}] (r_m={r_m}) M Contributions: {contrib_str}')
                    global_state_dict[k] = (1 - r_m) * global_state_dict[k] + r_m * agg


    def _aggregate_non_encoder_parameters(self, client_ids, client_n, client_state_dicts,
                                          non_encoder_keys, global_state_dict):
        # Aggregate non-encoder params using data size weighting
        total_cnt = sum(client_n[cid] for cid in client_ids)

        for key in non_encoder_keys:
            if not self._should_aggregate_parameter(key):
                continue # Skip parameters that should not be aggregated (e.g., private params in FedPer)

            agg = torch.zeros_like(global_state_dict[key])
            contrib_str = ""
            for cid in client_ids:
                w = client_n[cid] / total_cnt
                agg.add_(client_state_dicts[cid][key], alpha=w)
                contrib_str += f"{cid}: {w:.4f} | "
            # print(f'[Agg] [Key={key}] Contributions: {contrib_str}')
            global_state_dict[key] = agg


    def _should_aggregate_parameter(self,  key):
        """Determine if a parameter should be aggregated. Override in subclass (e.g., FedPer)"""
        return True # FedAvg aggregates all non-encoder parameters by default.


    def _assert_model_param_keys(self, encoder_keys, non_encoder_keys, global_state_dict):
        encoder_param_count = 0
        for mk in encoder_keys.values():
            for k in mk:
                encoder_param_count += global_state_dict[k].numel()
        non_encoder_param_count = 0
        for k in non_encoder_keys:
            non_encoder_param_count += global_state_dict[k].numel()

        total_param_cnt = sum(param.numel() for param in global_state_dict.values())
        assert encoder_param_count + non_encoder_param_count == total_param_cnt, \
            f"Total parameter count mismatch: {encoder_param_count} + {non_encoder_param_count} != {total_param_cnt}"
        # print(f'[Agg] Total parameters: {total_param_cnt} | Encoders: {encoder_param_count} | Non-encoders: {non_encoder_param_count}')