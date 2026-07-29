import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description='FLUX federated learning experiments')
    parser.add_argument('--project', type=str, default=None,
                        help='W&B project (default: wearable_project in wandb_config.json)')
    parser.add_argument('--seed', type=int, default=42, help='Seed for reproducibility')

    parser.add_argument('--method', type=str, default='fedavg',
                        choices=['fedavg', 'fedprox', 'fedper', 'moon', 'pmcmfl', 'fedrod', 'pepsy', 'flux'],
                        help='Method to apply')
    parser.add_argument('--dataset', type=str, default='pamap2',
                        choices=['pamap2', 'realworldhar', 'sleepedf20'], help='Dataset name')
    parser.add_argument('--eval_every', type=int, default=5, help='Evaluate (with validation set) every N rounds')
    parser.add_argument('--fusion_type', type=str, default='attention', choices=['attention'],
                        help='Fusion type to merge features from multiple modalities')

    # Missingness
    parser.add_argument('--availability', type=str, default='complete', choices=['low', 'medium', 'complete'],
                        help='Sensor availability level. `complete` means all sensors are available.')
    parser.add_argument('--stability', type=str, default='complete', choices=['low', 'medium', 'complete'],
                        help='Sensor stability level. `complete` means all sensors are stable.')
    # Imputation: zero-fill is applied in create_tensor_data.py when building *_tensors.pt
    parser.add_argument('--disable_ema_per_mod', action='store_true',
                        help='Disable EMA for per-modal aggregation. By default, EMA is applied.')

    # Hyperparameters
    # Same for all health-sensing datasets
    # Optimizer: SGD (momentum 0.9) is set in configs.py via apply_default_optimizer()
    parser.add_argument('--global_rounds', type=int, default=200, help='Number of global rounds')
    parser.add_argument('--local_bs', type=int, default=32, help='Local batch size')

    # Dataset-specific (method agnostic)
    parser.add_argument('--client_selection_rate', type=float, default=0.5, help='Client selection rate')
    parser.add_argument('--local_epochs', type=int, default=5, help='Number of local epochs')
    parser.add_argument('--local_lr', type=float, default=0.01, help='Local learning rate')
    parser.add_argument('--local_wd', type=float, default=0.001, help='Local weight decay')

    # Architecture parameters
    parser.add_argument('--unimodal_type', type=str, default='cnn', choices=['cnn'],
                        help='Unimodal encoder type (only CNN is wired in get_model)')
    parser.add_argument('--num_conv_layers', type=int, default=1, help='Number of convolutional layers')
    parser.add_argument('--conv_kernel_size', type=int, default=3, help='Kernel size of the convolutional layers')
    parser.add_argument('--conv_channels', type=int, default=32, help='Number of channels in convolutional layers')
    parser.add_argument('--conv_embed_dim', type=int, default=64, help='Embedding dimension of the convolutional layers')
    parser.add_argument('--conv_dropout', type=float, default=0.1, help='Dropout rate for convolutional layers')

    # Attention
    parser.add_argument('--att_hid_dim', type=int, default=256, help='Hidden dimension for self-attention')
    parser.add_argument('--att_proj_dropout', type=float, default=0.2, help='Dropout rate for attention projection')
    parser.add_argument('--att_num_contexts', type=int, default=4, help='Number of contexts for multi-context attention')

    # ---------------------------------------- #
    # ====== Method-specific parameters ====== #
    # ---------------------------------------- #
    # FedProx
    parser.add_argument('--fedprox_mu', type=float, default=0.001, help='FedProx regularization parameter')

    # MOON
    parser.add_argument('--moon_mu', type=float, default=5.0, help='MOON regularization parameter')
    parser.add_argument('--moon_temp', type=float, default=0.5, help='MOON temperature')

    # FedPer
    parser.add_argument('--private_fusion', action='store_true',
                        help='Enable private fusion in FedPer. By default, only head (classifier) is private.')

    # PmcmFL
    parser.add_argument('--pmcmfl_clip_loss_weight', type=float, default=0.1, help='Weight for CLIP loss in PmcmFL')

    # Fed-RoD
    parser.add_argument('--fedrod_brm_mode', type=str, default='none', choices=['balanced_softmax', 'none'],
                        help='BRM mode for Fed-RoD. `balanced_softmax` applies balanced softmax, `none` applies typical cross-entropy loss')

    # PEPSY
    parser.add_argument('--pepsy_pool_size', type=int, default=10, help='Size of the prompt pool')
    parser.add_argument('--pepsy_top_k', type=int, default=3, help='Top K prompts to retrieve from the pool')
    parser.add_argument('--pepsy_contrastive_weight', type=float, default=0.1,
                        help='Weight for contrastive loss in PEPSY (default: 0.1)')
    parser.add_argument('--pepsy_num_outer_loops', type=int, default=10,
                        help='Number of outer optimization loops for non-parametric prompt aggregation (default: 10)')
    parser.add_argument('--pepsy_use_ema_per_mod', action='store_true',
                        help='Use EMA for per-modal prompt aggregation in PEPSY (by default, disabled)')

    # Others: wandb (on by default), gpu, note, verbose
    parser.add_argument('--wandb', action=argparse.BooleanOptionalAction, default=True,
                        help='Log to Weights & Biases (default: on). Pass --no-wandb to disable.')
    parser.add_argument('--wandb_meta', action=argparse.BooleanOptionalAction, default=False,
                        help='Log FLUX confidence/tempering metrics under Meta/ in W&B (default: off).')
    parser.add_argument('--wandb_entity', type=str, default=None,
                        help='W&B entity (default: entity in wandb_config.json)')
    parser.add_argument('--gpu', type=int, default=7, help='GPU ID')
    parser.add_argument('--note', type=str, default='', help='Note for the experiment')
    parser.add_argument('--verbose', action='store_true', help='Verbose mode')

    # FLUX
    parser.add_argument('--flux_hidden_dim', type=int, default=512, help='Hidden dimension for FLUX confidence heads')

    args = parser.parse_args()
    return args