# ml_federated/server/FLUX_Server.py

import copy
import wandb
import numpy as np
from tqdm import trange
from collections import defaultdict
from ml_federated.server.FedAvg_Server import FedAvg_Server
from ml_federated.client.FLUX_Client import FLUX_Client


class FLUX_Server(FedAvg_Server):
    def __init__(self, model, dataset_configs, device, args):
        super().__init__(model, dataset_configs, device, args)

    def _create_client(self, client_id):
        return FLUX_Client(client_id, self.dataset_configs, self.device, self.args)

    def _should_use_global_model_selection(self):
        return False

    def run_federated_training(self):
        for rnd in trange(1, self.global_rounds + 1, desc=f'Training FL ({self.method.upper()})'):
            self.perf_stats[rnd] = {}
            self.current_round = rnd

            selected_clients = self.select_clients()
            self.perf_stats[rnd]['selected_clients'] = selected_clients

            losses = []
            metas = {}
            updated_models = {}
            for cid in selected_clients:
                client = self.clients[cid]
                updated_model, loss, meta = client.local_train(copy.deepcopy(self.model))
                updated_models[cid] = updated_model
                losses.append(loss)
                metas[cid] = meta

            self.model = self.aggregate_models(updated_models)

            round_loss = np.mean(losses)
            self.perf_stats[rnd]['round_loss'] = round_loss
            wandb.log({'round_loss': round_loss}, step=rnd) if self.args.wandb else None

            avg_meta = {}
            for cid, meta in metas.items():
                for k, v in meta.items():
                    if k not in avg_meta:
                        avg_meta[k] = []
                    avg_meta[k].append(v)
                    if self.args.wandb and self.args.wandb_meta:
                        wandb.log({f'Meta/c:{cid}_{k}': v}, step=rnd)
            for k, v in avg_meta.items():
                valid_vals = [val for val in v if val is not None]
                avg_meta[k] = None if len(valid_vals) == 0 else np.mean(valid_vals)
                if self.args.wandb and self.args.wandb_meta:
                    wandb.log({f'Meta/Avg_{k}': avg_meta[k]}, step=rnd)

            val_f1_str = 'Calculation skipped'
            if rnd == 1 or rnd % self.eval_every == 0:
                self.run_federated_evaluation(eval_on='val')
                val_f1_str = '{:.4f} ± {:.4f}'.format(self.perf_stats[rnd].get('val_f1_mean', 'N/A'),
                                                      self.perf_stats[rnd].get('val_f1_std', 'N/A'))
                if self._should_use_global_model_selection():
                    self._perform_global_model_selection(rnd)

            print(f'Round {rnd} | Loss: {round_loss:.4f} | Val F1: {val_f1_str}')

    def _partition_model_parameters(self, global_state_dict):
        encoder_keys = defaultdict(list)
        non_encoder_keys = []
        for key in global_state_dict.keys():
            if key.startswith('encoder') or key.startswith('unimodal_confidence_heads'):
                parts = key.split('.')
                encoder_idx = int(parts[1])
                assert encoder_idx < self.num_modalities, f'Encoder index {encoder_idx} out of range'
                encoder_keys[encoder_idx].append(key)
            else:
                non_encoder_keys.append(key)

        self._assert_model_param_keys(encoder_keys, non_encoder_keys, global_state_dict)
        return encoder_keys, non_encoder_keys

    def aggregate_models(self, updated_models):
        client_ids = list(updated_models.keys())
        client_n, client_per_modality_n = self._get_client_data_sizes(client_ids)
        client_state_dicts = {cid: updated_models[cid].state_dict() for cid in client_ids}

        new_model = copy.deepcopy(self.model)
        global_state_dict = new_model.state_dict()
        encoder_keys, non_encoder_keys = self._partition_model_parameters(global_state_dict)

        # Per-modality aggregation for encoders and unimodal confidence heads
        self._aggregate_encoder_parameters(client_ids, client_n, client_per_modality_n,
                                           client_state_dicts, encoder_keys, global_state_dict)
        self._aggregate_non_encoder_parameters(client_ids, client_n, client_state_dicts,
                                               non_encoder_keys, global_state_dict)
        new_model.load_state_dict(global_state_dict)
        return new_model
