"""PEPSY server for the ADNI multimodal pipeline.

Key features:
- Non-parametric aggregation for prompt pool (dynamic pool size)
- Per-modality weighted aggregation for encoders and sim_projectors
- Standard weighted aggregation for classifier, global_fuser, embeddings, pool.features_proj
- Global model selection (best model on val F1)
"""

import copy
import time
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm, trange
from adni_pipeline.fl.adni_pepsy_client import ADNI_PEPSY_Client
from adni_pipeline.models.pepsy_nonparametric_agg import NonparametricAgg


class ADNI_PEPSY_Server:

    def __init__(self, model, cfg, device, args):
        self.model = model
        self.cfg = cfg
        self.device = device
        self.args = args
        self.modalities = cfg["modalities"]
        self.global_rounds = cfg["global_rounds"]
        self.eval_every = getattr(args, "eval_every", 5)
        self.num_outer_loops = cfg.get("pepsy_num_outer_loops", 10)

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

    def _setup_clients(self):
        start_time = time.time()
        for cid in tqdm(self.client_ids, desc="Initializing ADNI PEPSY clients"):
            self.clients[cid] = ADNI_PEPSY_Client(cid, self.cfg, self.device)
        print(f"Server initialized with {len(self.client_ids)} PEPSY clients "
              f"in {time.time() - start_time:.2f}s")

    # ------------------------------------------------------------------ #
    # Main FL loop
    # ------------------------------------------------------------------ #

    def run(self):
        print("=" * 60)
        print("ADNI PEPSY Server is running...")
        print("=" * 60)
        self.run_federated_training()
        self._round_log = {}
        self.run_federated_evaluation(eval_on="test")

    def run_federated_training(self):
        for rnd in trange(1, self.global_rounds + 1, desc="FL Training (PEPSY)"):
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

        aggregated_keys = set()

        self._aggregate_per_modality_parameters(
            client_ids, client_n, client_per_modality_n,
            client_state_dicts, global_state_dict, aggregated_keys,
        )

        self._aggregate_shared_parameters(
            client_ids, client_n, client_state_dicts,
            global_state_dict, aggregated_keys,
        )

        self._aggregate_prompts_nonparametric(
            client_ids, updated_models, global_state_dict,
        )
        aggregated_keys.add("pool.prompt")

        prompt_param = global_state_dict.pop("pool.prompt", None)
        new_model.load_state_dict(global_state_dict, strict=False)

        if prompt_param is not None:
            with torch.no_grad():
                new_model.pool.prompt = nn.Parameter(prompt_param, requires_grad=True)
                if new_model.pool.prompt.shape[0] != new_model.trained_prompts_checklist.shape[0]:
                    new_model.trained_prompts_checklist = torch.zeros(
                        new_model.pool.prompt.shape[0],
                        dtype=torch.float32,
                        device=new_model.pool.prompt.device,
                    )

        return new_model

    def _get_client_data_sizes(self, client_ids):
        client_n = {}
        client_per_modality_n = {}
        for cid in client_ids:
            client = self.clients[cid]
            client_n[cid] = client.get_data_size("train")
            client_per_modality_n[cid] = client.get_modality_data_size("train")
        return client_n, client_per_modality_n

    def _aggregate_per_modality_parameters(self, client_ids, client_n, client_per_modality_n,
                                           client_state_dicts, global_state_dict, aggregated_keys):
        """Weighted averaging for encoders.{mod}.* and sim_projectors.{mod}.* with optional EMA."""
        total_all = sum(client_n.values())
        disable_ema = getattr(self.args, "disable_ema_per_mod", False)
        mod_to_idx = {m: i for i, m in enumerate(self.modalities)}

        per_mod_prefixes = ["encoders.", "sim_projectors."]

        for key in list(global_state_dict.keys()):
            matched_prefix = None
            for prefix in per_mod_prefixes:
                if key.startswith(prefix):
                    matched_prefix = prefix
                    break
            if matched_prefix is None:
                continue

            mod_name = key.split(".")[1]
            if mod_name not in mod_to_idx:
                continue
            m = mod_to_idx[mod_name]
            total_m_cnt = sum(client_per_modality_n[c][m] for c in client_ids)
            if total_m_cnt == 0:
                aggregated_keys.add(key)
                continue

            r_m = max(total_m_cnt / total_all, 0.05)

            agg = torch.zeros_like(global_state_dict[key])
            for cid in client_ids:
                if client_per_modality_n[cid][m] > 0:
                    w = client_per_modality_n[cid][m] / total_m_cnt
                    agg.add_(client_state_dicts[cid][key], alpha=w)

            if disable_ema:
                global_state_dict[key] = agg
            else:
                global_state_dict[key] = (1 - r_m) * global_state_dict[key] + r_m * agg

            aggregated_keys.add(key)

    def _aggregate_shared_parameters(self, client_ids, client_n,
                                     client_state_dicts, global_state_dict, aggregated_keys):
        """Standard weighted averaging for all non-per-modality, non-prompt parameters."""
        total_cnt = sum(client_n[cid] for cid in client_ids)
        skip_prefixes = ("encoders.", "sim_projectors.", "pool.prompt")

        for key in global_state_dict.keys():
            if key in aggregated_keys:
                continue
            if any(key.startswith(p) for p in skip_prefixes):
                continue

            agg = torch.zeros_like(global_state_dict[key])
            for cid in client_ids:
                w = client_n[cid] / total_cnt
                agg.add_(client_state_dicts[cid][key], alpha=w)
            global_state_dict[key] = agg
            aggregated_keys.add(key)

    def _aggregate_prompts_nonparametric(self, client_ids, updated_models, global_state_dict):
        """Non-parametric prompt aggregation (core PEPSY innovation)."""
        union_checklist = torch.zeros(
            self.model.pool.prompt.shape[0], dtype=torch.int,
            device=self.model.pool.prompt.device,
        )

        nonzero_idx = torch.nonzero(self.model.trained_prompts_checklist).flatten()
        union_checklist[nonzero_idx] = 1

        for cid in client_ids:
            cm = updated_models[cid]
            nonzero_idx = torch.nonzero(cm.trained_prompts_checklist).flatten()
            union_checklist[nonzero_idx] = 1

        trained_indices = torch.nonzero(union_checklist).flatten()
        if len(trained_indices) == 0:
            raise ValueError("No trained prompts found across any client.")

        local_prompts_list = []
        for cid in client_ids:
            lp = updated_models[cid].pool.prompt[trained_indices].detach().clone()
            if torch.isnan(lp).any() or torch.isinf(lp).any():
                raise ValueError(
                    f"Client {cid} has NaN/Inf in prompts at round {self.current_round}.")
            local_prompts_list.append(lp)

        local_prompts_tensor = torch.stack(local_prompts_list, dim=0)

        prompt_dim = local_prompts_tensor.shape[-1]
        agg = NonparametricAgg(prompt_dim, n_hidden=64).to(local_prompts_tensor.device)
        aggregated_prompts = agg(local_prompts_tensor, outer_loop=self.num_outer_loops)
        del agg

        with torch.no_grad():
            global_state_dict["pool.prompt"] = nn.Parameter(
                aggregated_prompts, requires_grad=True)

        for cid in client_ids:
            with torch.no_grad():
                updated_models[cid].pool.prompt = nn.Parameter(
                    aggregated_prompts.clone(), requires_grad=True)

        self.model.reset_trained_prompts_checklist()
        for cid in client_ids:
            updated_models[cid].reset_trained_prompts_checklist()

        return aggregated_prompts

    # ------------------------------------------------------------------ #
    # Logging
    # ------------------------------------------------------------------ #

    def _wandb_log(self, payload, step=None):
        if not self.use_wandb:
            return
        import wandb
        wandb.log(payload, step=step)
