"""MOON server for the ADNI multimodal pipeline."""

import copy
import time
import torch
import numpy as np
from tqdm import tqdm, trange
from collections import defaultdict
from adni_pipeline.fl.adni_moon_client import ADNI_Moon_Client



class ADNI_Moon_Server:
    """MOON server tailored for the ADNI multimodal pipeline."""

    def __init__(self, model, cfg, device, args):
        """
        Args:
            model: ADNIMoonModel instance.
            cfg: dict with ADNI_DATASET_CFG keys plus 'adni_root'.
            device: torch device.
            args: argparse namespace (eval_every, wandb, seed, disable_ema_per_mod, ...).
        """
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
        self.best_model = None
        self.best_val_f1 = 0.0
        self.use_wandb = getattr(args, "wandb", False)

        self.clients = {}
        self.model.to(self.device)
        self._setup_clients()

    # ------------------------------------------------------------------ #
    # Client setup
    # ------------------------------------------------------------------ #

    def _setup_clients(self):
        start_time = time.time()
        for cid in tqdm(self.client_ids, desc="Initializing ADNI MOON clients"):
            client = ADNI_Moon_Client(cid, self.cfg, self.device)
            self.clients[cid] = client
        print(f"Server initialized with {len(self.client_ids)} MOON clients in {time.time() - start_time:.2f}s")

    # ------------------------------------------------------------------ #
    # Main FL loop
    # ------------------------------------------------------------------ #

    def run(self):
        print("=" * 60)
        print("ADNI MOON Server is running...")
        print("=" * 60)
        self.run_federated_training()
        self._round_log = {}
        self.run_federated_evaluation(eval_on="test")


    def run_federated_training(self):
        for rnd in trange(1, self.global_rounds + 1, desc="FL Training (MOON)"):
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
                self._perform_global_model_selection(rnd)

            self._wandb_log(self._round_log, step=rnd)
            print(f"Round {rnd} - Loss: {round_loss:.4f} | Val F1: {val_f1_str}")

    # ------------------------------------------------------------------ #
    # Evaluation
    # ------------------------------------------------------------------ #

    def run_federated_evaluation(self, eval_on):
        if eval_on == "test" and self.best_model is not None:
            evaluation_model = self.best_model
        else:
            evaluation_model = self.model

        all_client_f1s = {}
        self.perf_stats[self.current_round][f"client_{eval_on}_f1"] = {}
        for cid in self.client_ids:
            client = self.clients[cid]
            f1 = client.local_test(copy.deepcopy(evaluation_model), eval_on)
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

    def _perform_global_model_selection(self, rnd):
        cur_val_f1 = self.perf_stats[rnd].get("val_f1_mean", 0.0)
        if cur_val_f1 > self.best_val_f1:
            self.best_val_f1 = cur_val_f1
            self.best_model = copy.deepcopy(self.model)
            self.perf_stats[rnd]["best_model_updated"] = True
            print(f"New best model at round {rnd} with val F1: {cur_val_f1:.4f}")

    # ------------------------------------------------------------------ #
    # Client selection
    # ------------------------------------------------------------------ #

    def select_clients(self):
        return np.random.choice(self.client_ids, size=self.num_selected_clients, replace=False)

    # ------------------------------------------------------------------ #
    # Aggregation
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
        """Partition state_dict keys into per-modality encoder groups and non-encoder keys.

        Encoder keys follow the pattern ``encoders.{modality_name}.*`` from ModuleDict.
        The modality name is mapped to an integer index matching the order in
        self.modalities so that per-modality sample counts can be looked up.
        """
        mod_to_idx = {m: i for i, m in enumerate(self.modalities)}
        encoder_keys = defaultdict(list)  # int index -> list of keys
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
        # log_detail = (self.current_round == 1)
        # if log_detail:
        #     print("\n[Aggregation] Encoder-level modality-weighted aggregation (Round 1)")
        #     print(f"  Total samples across selected clients: {total_all}")
        #     for cid in client_ids:
        #         print(f"  {cid}: total={client_n[cid]}, "
        #               f"per_mod={dict(zip(self.modalities, client_per_modality_n[cid]))}")

        for m in range(num_modalities):
            keys = encoder_keys.get(m, [])
            total_m_cnt = sum(client_per_modality_n[c][m] for c in client_ids)
            if not keys or total_m_cnt == 0:
                # if log_detail:
                #     print(f"  [{self.modalities[m]}] no samples -> skipped")
                continue

            r_m = max(total_m_cnt / total_all, 0.05)

            # if log_detail:
            #     weights = {cid: client_per_modality_n[cid][m] / total_m_cnt for cid in client_ids}
            #     print(f"  [{self.modalities[m]}] total_m={total_m_cnt}, r_m={r_m:.4f}, "
            #           f"ema={'off' if disable_ema else 'on'}, "
            #           f"weights={{ {', '.join(f'{c}: {w:.4f}' for c, w in weights.items())} }}")

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
