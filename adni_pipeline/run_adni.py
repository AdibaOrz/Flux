"""Main entry point for ADNI federated learning (all methods)."""

import os
import torch
import random
import argparse
import numpy as np
from adni_pipeline.models.adni_model import build_adni_model
from adni_pipeline.configs.adni_config import load_adni_fl_config, ADNI_PARTITION_MAP
from adni_pipeline.dataset_prep.ADNI.path_config import load_adni_paths
from utils.wandb_config import apply_wandb_cli_defaults


def get_adni_server(method, model, cfg, device, args):
    if method == "fedavg":
        from adni_pipeline.fl.adni_fedavg_server import ADNI_FedAvg_Server
        return ADNI_FedAvg_Server(model, cfg, device, args)
    elif method == "fedprox":
        from adni_pipeline.fl.adni_fedprox_server import ADNI_FedProx_Server
        return ADNI_FedProx_Server(model, cfg, device, args)
    elif method == "fedper":
        from adni_pipeline.fl.adni_fedper_server import ADNI_FedPer_Server
        return ADNI_FedPer_Server(model, cfg, device, args)
    elif method == "moon":
        from adni_pipeline.fl.adni_moon_server import ADNI_Moon_Server
        return ADNI_Moon_Server(model, cfg, device, args)
    elif method == "fedrod":
        from adni_pipeline.fl.adni_fedrod_server import ADNI_FedRoD_Server
        return ADNI_FedRoD_Server(model, cfg, device, args)
    elif method == "pmcmfl":
        from adni_pipeline.fl.adni_pmcmfl_server import ADNI_PmcmFL_Server
        return ADNI_PmcmFL_Server(model, cfg, device, args)
    elif method == "pepsy":
        from adni_pipeline.fl.adni_pepsy_server import ADNI_PEPSY_Server
        return ADNI_PEPSY_Server(model, cfg, device, args)
    elif method == "flux":
        from adni_pipeline.fl.adni_flux_server import ADNI_FLUX_Server
        return ADNI_FLUX_Server(model, cfg, device, args)
    else:
        raise NotImplementedError(f"Method '{method}' is not yet implemented for ADNI. ")


def parse_args():
    p = argparse.ArgumentParser(description="ADNI Experiments")
    p.add_argument("--method", type=str, choices = ["fedavg", "fedprox", "fedper", "moon", "fedrod", "pmcmfl", "pepsy", "flux"],
                   default="fedavg", help=f"FL method to run.")
    p.add_argument("--path_config", type=str, default=None,
                   help="Path to adni_paths.json (default: ADNI_PATH_CONFIG or dataset_prep/ADNI/adni_paths.json).")
    p.add_argument("--adni_root", type=str, default=None,
                   help="Override the ADNI dataset root from the path config.")
    p.add_argument("--partition_name", type=str, default=None,
                   choices=list(ADNI_PARTITION_MAP.keys()),
                   help="Silo split type (e.g. stratified, modality_unique, med_intra_0p0, low_intra_0p0).")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--global_rounds", type=int, default=None)
    p.add_argument("--local_epochs", type=int, default=None)
    p.add_argument("--local_lr", type=float, default=None)
    p.add_argument("--local_bs", type=int, default=None)
    p.add_argument("--fedprox_mu", type=float, default=None, help="FedProx proximal regularization strength.")
    p.add_argument("--moon_mu", type=float, default=None, help="MOON contrastive loss weight.")
    p.add_argument("--moon_temp", type=float, default=None, help="MOON contrastive temperature.")

    p.add_argument("--client_selection_rate", type=float, default=None)
    p.add_argument("--eval_every", type=int, default=5)
    p.add_argument("--disable_ema_per_mod", action="store_true")
    p.add_argument("--pmcmfl_clip_loss_weight", type=float, default=None, help="CLIP loss weight for PmcmFL.")
    p.add_argument("--flux_hidden_dim", type=int, default=None, help="Hidden dim for FLUX confidence heads.")
    p.add_argument("--pepsy_contrastive_weight", type=float, default=None, help="Contrastive loss weight for PEPSY.")
    p.add_argument("--pepsy_pool_size", type=int, default=None, help="Prompt pool size for PEPSY.")
    p.add_argument("--pepsy_top_k", type=int, default=None, help="Top-k prompt selection for PEPSY.")
    p.add_argument(
        "--wandb",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Log to Weights & Biases (default: on). Pass --no-wandb to disable.",
    )
    p.add_argument("--wandb_project", type=str, default=None,
                   help="W&B project (default: adni_project in wandb_config.json)")
    p.add_argument("--wandb_entity", type=str, default=None,
                   help="W&B entity (default: entity in wandb_config.json)")
    p.add_argument("--note", type=str, default="")
    p.add_argument(
        "--debug",
        action="store_true",
        help="Print temporary wiring/debug logs (smoke tests).",
    )
    return p.parse_args()


def main():
    args = parse_args()
    if args.wandb:
        apply_wandb_cli_defaults(args, pipeline="adni")
    set_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    adni_root = args.adni_root or str(load_adni_paths(args.path_config)["adni_root"])

    cfg = load_adni_fl_config(args.method)
    cfg["adni_root"] = adni_root
    cfg["debug"] = args.debug

    cli_overrides = ["global_rounds", "local_epochs", "local_lr",
                     "local_bs", "client_selection_rate", "partition_name",
                     "fedprox_mu", "moon_mu", "moon_temp",
                     "pmcmfl_clip_loss_weight", "flux_hidden_dim",
                     "pepsy_contrastive_weight", "pepsy_pool_size", "pepsy_top_k"]

    for key in cli_overrides:
        val = getattr(args, key, None)
        if val is not None:
            print(f"  [CLI override] {key}: {cfg.get(key)} -> {val}")
            cfg[key] = val

    if args.method == "pepsy" and args.pepsy_contrastive_weight is None:
        from adni_pipeline.configs.adni_config import resolve_pepsy_contrastive_weight
        cfg["pepsy_contrastive_weight"] = resolve_pepsy_contrastive_weight(cfg["partition_name"])
        print(f"  [PEPSY] pepsy_contrastive_weight from partition "
              f"'{cfg['partition_name']}': {cfg['pepsy_contrastive_weight']}")

    print("=" * 60)
    print(f"ADNI FL  |  method = {args.method}")
    print("=" * 60)
    for k in ["global_rounds", "local_epochs", "local_lr",
              "local_bs", "client_selection_rate",
              "partition_name", "num_total_clients"]:
        print(f"  {k}: {cfg[k]}")
    method_prefix = f"{args.method}_"
    for k in sorted(key for key in cfg.keys() if key.startswith(method_prefix)):
        print(f"  {k}: {cfg[k]}")
    print(f"  device: {device}")
    print(f"  seed: {args.seed}")
    print()

    if args.wandb:
        import wandb
        run_name = args.note or f"{args.method}_s{args.seed}"
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            notes=args.note,
            config={**cfg, "seed": args.seed, "device": str(device), "note": args.note},
            tags=[args.method, cfg["partition_name"], f"seed_{args.seed}"],
        )

    if args.method == "flux":
        from adni_pipeline.models.adni_flux_model import build_adni_flux_model
        model = build_adni_flux_model(cfg)
    elif args.method == "pmcmfl":
        from adni_pipeline.models.adni_pmcmfl_model import build_adni_pmcmfl_model
        model = build_adni_pmcmfl_model(cfg)
    elif args.method == "pepsy":
        from adni_pipeline.models.adni_pepsy_model import build_adni_pepsy_model
        model = build_adni_pepsy_model(cfg)
    elif args.method == "moon":
        from adni_pipeline.models.adni_moon_model import build_adni_moon_model
        model = build_adni_moon_model(cfg)
    else:
        model = build_adni_model(cfg)
    server = get_adni_server(args.method, model, cfg, device, args)
    server.run()

    if args.wandb:
        wandb.finish()


#############################
##### Utility functions #####
#############################

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # for multi-GPU
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)



if __name__ == "__main__":
    main()
