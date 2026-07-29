"""FedRoD server for the ADNI multimodal pipeline.

Aggregation is identical to FedAvg. The only difference is that global model
selection is disabled -- each client tracks its own best snapshot via local_test.
"""

import copy
import time
import torch
import numpy as np
from tqdm import tqdm, trange
from collections import defaultdict
from adni_pipeline.fl.adni_fedrod_client import ADNI_FedRoD_Client


class ADNI_FedRoD_Server:

    def __init__(self, model, cfg, device, args):
        self.model = model
        self.cfg = cfg
        self.device = device
        self.args = args
        self.modalities = cfg["modalities"]
        self.global_rounds = cfg["global_rounds"]
        self.eval_every = getattr(args, "eval_every", 5)

        self.client_ids = cfg["client_ids"]
        self.num_total_clients = cfg["num_total_clients"]
        self.client_selection_rate = cfg["client_selection_rate"]
        self.num_selected_clients = max(1, int(self.num_total_clients * self.client_selection_rate))

        self.current_round = 1
        self.perf_stats = {}
        self.use_wandb = getattr(args, "wandb", False)

        self.clients = {}
        self.model.to(self.device)
        self._setup_clients()

    # ------------------------------------------------------------------ #
    # Client setup
    # ------------------------------------------------------------------ #

    def _setup_clients(self):
        start_time = time.time()
        for cid in tqdm(self.client_ids, desc="Initializing ADNI FedRoD clients"):
            client = ADNI_FedRoD_Client(cid, self.cfg, self.device)
            self.clients[cid] = client
        print(f"Server initialized with {len(self.client_ids)} FedRoD clients in {time.time() - start_time:.2f}s")

    # ------------------------------------------------------------------ #
    # Main FL loop
    # ------------------------------------------------------------------ #

    def run(self):
        print("=" * 60)
        print("ADNI FedRoD Server is running...")
        print("=" * 60)
        self.run_federated_training()
        self._round_log = {}
        self.run_federated_evaluation(eval_on="test")

    def run_federated_training(self):
        for rnd in trange(1, self.global_rounds + 1, desc="FL Training (FedRoD)"):
            self.perf_stats[rnd] = {}
            self.current_round = rnd
            self._round_log = {}

            selected_clients = self.select_clients()
            self.perf_stats[rnd]["selected_clients"] = list(selected_clients)

            losses = []
            updated_models = {}
            for cid in selected_clients:
                client = self.clients[cid]
                updated_model, loss = client.local_train(copy.deepcopy(self.model))
                updated_models[cid] = updated_model
                losses.append(loss)

            self.model = self.aggregate_models(updated_models)

            round_loss = np.mean(losses)
            self.perf_stats[rnd]["round_loss"] = round_loss
            self._round_log["round_loss"] = round_loss

            val_f1_str = "skipped"
            if rnd == 1 or rnd % self.eval_every == 0:
                self.run_federated_evaluation(eval_on="val")
                val_mean = self.perf_stats[rnd].get("val_f1_mean", 0)
                val_std = self.perf_stats[rnd].get("val_f1_std", 0)
                val_f1_str = f"{val_mean:.4f} +/- {val_std:.4f}"

            self._wandb_log(self._round_log, step=rnd)
            print(f"Round {rnd} - Loss: {round_loss:.4f} | Val F1: {val_f1_str}")

    # ------------------------------------------------------------------ #
    # Evaluation
    # ------------------------------------------------------------------ #

    def run_federated_evaluation(self, eval_on):
        all_client_f1s = {}
        self.perf_stats[self.current_round][f"client_{eval_on}_f1"] = {}
        for cid in self.client_ids:
            client = self.clients[cid]
            f1 = client.local_test(copy.deepcopy(self.model), eval_on)
            all_client_f1s[cid] = f1
            self.perf_stats[self.current_round][f"client_{eval_on}_f1"][cid] = f1
            self._round_log[f"{eval_on}_f1_{cid}"] = f1

        mean_f1 = np.mean(list(all_client_f1s.values()))
        std_f1 = np.std(list(all_client_f1s.values()))
        self.perf_stats[self.current_round][f"{eval_on}_f1_mean"] = mean_f1
        self.perf_stats[self.current_round][f"{eval_on}_f1_std"] = std_f1
        self._round_log[f"{eval_on}_f1_mean"] = mean_f1
        self._round_log[f"{eval_on}_f1_std"] = std_f1

        if eval_on == "test":
            self._wandb_log(self._round_log, step=self.current_round)
            print("=" * 60)
            print(f"Final Test F1 across all clients: {mean_f1:.4f} +/- {std_f1:.4f}")

    # ------------------------------------------------------------------ #
    # Client selection
    # ------------------------------------------------------------------ #

    def select_clients(self):
        return np.random.choice(self.client_ids, size=self.num_selected_clients, replace=False)

    # ------------------------------------------------------------------ #
    # Aggregation (identical to FedAvg -- p_head is never in the state_dict)
    # ------------------------------------------------------------------ #

    def aggregate_models(self, updated_models):
        client_ids = list(updated_models.keys())
        client_n, client_per_modality_n = self._get_client_data_sizes(client_ids)
        client_state_dicts = {cid: updated_models[cid].state_dict() for cid in client_ids}

        new_model = copy.deepcopy(self.model)
        global_state_dict = new_model.state_dict()
        encoder_keys, non_encoder_keys = self._partition_model_parameters(global_state_dict)

        self._aggregate_encoder_parameters(
            client_ids, client_n, client_per_modality_n,
            client_state_dicts, encoder_keys, global_state_dict,
        )
        self._aggregate_non_encoder_parameters(
            client_ids, client_n, client_state_dicts,
            non_encoder_keys, global_state_dict,
        )

        new_model.load_state_dict(global_state_dict)
        return new_model

    def _get_client_data_sizes(self, client_ids):
        client_n = {}
        client_per_modality_n = {}
        for cid in client_ids:
            client = self.clients[cid]
            client_n[cid] = client.get_data_size("train")
            client_per_modality_n[cid] = client.get_modality_data_size("train")
        return client_n, client_per_modality_n

    def _partition_model_parameters(self, global_state_dict):
        mod_to_idx = {m: i for i, m in enumerate(self.modalities)}
        encoder_keys = defaultdict(list)
        non_encoder_keys = []

        for key in global_state_dict.keys():
            if key.startswith("encoders."):
                mod_name = key.split(".")[1]
                if mod_name not in mod_to_idx:
                    raise ValueError(f"Unknown modality in state_dict key: {key}")
                encoder_keys[mod_to_idx[mod_name]].append(key)
            else:
                non_encoder_keys.append(key)

        return encoder_keys, non_encoder_keys

    def _aggregate_encoder_parameters(self, client_ids, client_n, client_per_modality_n,
                                      client_state_dicts, encoder_keys, global_state_dict):
        total_all = sum(client_n.values())
        num_modalities = len(self.modalities)
        disable_ema = getattr(self.args, "disable_ema_per_mod", False)

        for m in range(num_modalities):
            keys = encoder_keys.get(m, [])
            total_m_cnt = sum(client_per_modality_n[c][m] for c in client_ids)
            if not keys or total_m_cnt == 0:
                continue

            r_m = max(total_m_cnt / total_all, 0.05)

            for k in keys:
                agg = torch.zeros_like(global_state_dict[k])
                for cid in client_ids:
                    w = client_per_modality_n[cid][m] / total_m_cnt
                    agg.add_(client_state_dicts[cid][k], alpha=w)

                if disable_ema:
                    global_state_dict[k] = agg
                else:
                    global_state_dict[k] = (1 - r_m) * global_state_dict[k] + r_m * agg

    def _aggregate_non_encoder_parameters(self, client_ids, client_n,
                                          client_state_dicts, non_encoder_keys,
                                          global_state_dict):
        total_cnt = sum(client_n[cid] for cid in client_ids)
        for key in non_encoder_keys:
            agg = torch.zeros_like(global_state_dict[key])
            for cid in client_ids:
                w = client_n[cid] / total_cnt
                agg.add_(client_state_dicts[cid][key], alpha=w)
            global_state_dict[key] = agg

    # ------------------------------------------------------------------ #
    # Logging
    # ------------------------------------------------------------------ #

    def _wandb_log(self, payload, step=None):
        if not self.use_wandb:
            return
        import wandb
        wandb.log(payload, step=step)
