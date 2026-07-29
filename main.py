# main.py

import torch
import utils.pretty_print as pp
from arg_parser import parse_arguments
from models.model_utils import get_model
from utils.utils import set_seed, init_wandb
from utils.wandb_config import apply_wandb_cli_defaults
from ml_federated.server.server_utils import get_server
from configs import load_dataset_configs, get_tuned_hyperparameters, is_method_tuned, get_method_hp_keys, apply_default_optimizer



def main():
    args = parse_arguments()
    if args.wandb:
        apply_wandb_cli_defaults(args, pipeline="wearable")
    set_seed(args.seed)
    dataset_name = args.dataset
    device = torch.device(f"cuda:{args.gpu}") if torch.cuda.is_available() else "cpu"
    dataset_configs = load_dataset_configs(dataset_name)

    if is_method_tuned(dataset_name, args.method):
        tuned_hps = get_tuned_hyperparameters(dataset_name, args.method)
        for en_id, (hp, value) in enumerate(tuned_hps.items()):
            print(f'>>---[HP={en_id+1}]---<< Using ({args.method}-specific) tuned hp for method={args.method} and dataset={args.dataset} // {hp} = {value}')
            dataset_configs[hp] = value
    else:
        hp_keys = get_method_hp_keys(args.method)
        if len(hp_keys) > 0:
            print(f"Warning: No tuned hp found for {args.method} on {dataset_name}. Using values from args.")
        for hp in hp_keys:
            dataset_configs[hp] = getattr(args, hp)
            print(f"[!! ({args.method}-specific) NOT Tuned HP !!] Setting {hp} in method {args.method}: {dataset_configs[hp]}")

    apply_default_optimizer(dataset_configs, args.method)

    args.dataset_configs = dataset_configs # to check from wandb
    if args.verbose: print_params(args)
    if args.wandb: init_wandb(args)        # init wandb

    model = get_model(dataset_configs, args.fusion_type, args.method, args=args)
    server = get_server(model, dataset_configs, device, args)
    server.run()


def print_params(args):
    print('Parameters:\n', vars(args))
    print('Dataset Configs:\n', args.dataset_configs)
    print('Tuned Hyperparameters:')
    if is_method_tuned(args.dataset, args.method):
        tuned_hps = get_tuned_hyperparameters(args.dataset, args.method)
        print(tuned_hps)
    else:
        print(f'No tuned hyperparameters found for {args.method} on {args.dataset}.')


if __name__ == '__main__':
    pp.print_string("FLUX")
    main()