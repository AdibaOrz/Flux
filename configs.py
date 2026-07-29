"""
Configurations for the datasets (metadata & training hyperparameters) used in the experiments.
"""



PAMAP2Opt = {
    # Data parameters
    'name': 'pamap2',
    'num_total_clients': 8,
    'client_ids': [1, 2, 3, 4, 5, 6, 7, 8],
    'input_dim': 18,
    'classes': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    'num_classes': 12,
    'sampling_rate': 100,
    'seq_len': 100 * 2,   # 2 seconds * sampling_rate
    'sensor_list': ['hand_acc', 'hand_gyro',
                    'chest_acc', 'chest_gyro',
                    'ankle_acc', 'ankle_gyro'],
    'num_sensors': 6,
    'feature_columns': ['hand_acc_x', 'hand_acc_y', 'hand_acc_z', 'hand_gyro_x', 'hand_gyro_y', 'hand_gyro_z',
                        'chest_acc_x', 'chest_acc_y', 'chest_acc_z', 'chest_gyro_x', 'chest_gyro_y', 'chest_gyro_z',
                        'ankle_acc_x', 'ankle_acc_y', 'ankle_acc_z', 'ankle_gyro_x', 'ankle_gyro_y', 'ankle_gyro_z'],
    'all_columns_ordered': ['timestamp', 'label',
                            'hand_acc_x', 'hand_acc_y', 'hand_acc_z', 'hand_gyro_x', 'hand_gyro_y', 'hand_gyro_z',
                            'chest_acc_x', 'chest_acc_y', 'chest_acc_z', 'chest_gyro_x', 'chest_gyro_y', 'chest_gyro_z',
                            'ankle_acc_x', 'ankle_acc_y', 'ankle_acc_z', 'ankle_gyro_x', 'ankle_gyro_y', 'ankle_gyro_z',
                            'user_id'],
    'sensor_to_featureColumn': {0: [0, 1, 2], 1: [3, 4, 5], 2: [6, 7, 8],
                                3: [9, 10, 11], 4: [12, 13, 14], 5: [15, 16, 17]},

    # Client selection rate
    "client_selection_rate": 0.5,  # 4 clients

    # Architecture (unimodal)
    "unimodal_type": "cnn",
    "num_conv_layers": 1,
    "conv_kernel_size": 3,
    "conv_channels": 64,
    "conv_embed_dim": 128,
    "conv_dropout": 0.1,

    # Self-attention (multimodal fusion)
    "att_hid_dim": 256,
    "att_proj_dropout": 0.2,
    "att_num_contexts": 4,

    # Fused representation dim (for auxiliary uses)
    "fused_output_dim": 128

}


RealWorldHAROpt = {
    # Data parameters
    'name': 'realworldhar',
    'num_total_clients': 15,
    'client_ids': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
    'input_dim': 30,
    'classes': [0, 1, 2, 3, 4, 5, 6, 7],
    'num_classes': 8,
    'sampling_rate': 50,
    'seq_len': 3 * 50,   # 3 seconds * sampling_rate
    'sensor_list': [
                    'chest_acc', 'head_acc',
                    'shin_acc', 'upperarm_acc',
                    'waist_acc', 'chest_gyro',
                    'head_gyro', 'shin_gyro',
                    'upperarm_gyro', 'waist_gyro'],
    'num_sensors': 10,
    'feature_columns': [
                    'chest_acc_x', 'chest_acc_y', 'chest_acc_z',
                    'head_acc_x', 'head_acc_y', 'head_acc_z',
                    'shin_acc_x', 'shin_acc_y', 'shin_acc_z',
                    'upperarm_acc_x', 'upperarm_acc_y', 'upperarm_acc_z',
                    'waist_acc_x', 'waist_acc_y', 'waist_acc_z',
                    'chest_gyro_x', 'chest_gyro_y', 'chest_gyro_z',
                    'head_gyro_x', 'head_gyro_y', 'head_gyro_z',
                    'shin_gyro_x', 'shin_gyro_y', 'shin_gyro_z',
                    'upperarm_gyro_x', 'upperarm_gyro_y', 'upperarm_gyro_z',
                    'waist_gyro_x', 'waist_gyro_y', 'waist_gyro_z'],
    'all_columns_ordered': [
                    'chest_acc_x', 'chest_acc_y', 'chest_acc_z',
                    'head_acc_x', 'head_acc_y', 'head_acc_z',
                    'shin_acc_x', 'shin_acc_y', 'shin_acc_z',
                    'upperarm_acc_x', 'upperarm_acc_y', 'upperarm_acc_z',
                    'waist_acc_x', 'waist_acc_y', 'waist_acc_z',
                    'chest_gyro_x', 'chest_gyro_y', 'chest_gyro_z',
                    'head_gyro_x', 'head_gyro_y', 'head_gyro_z',
                    'shin_gyro_x', 'shin_gyro_y', 'shin_gyro_z',
                    'upperarm_gyro_x', 'upperarm_gyro_y', 'upperarm_gyro_z',
                    'waist_gyro_x', 'waist_gyro_y', 'waist_gyro_z', 'label', 'user_id'],
    'sensor_to_featureColumn': {0: [0, 1, 2], 1: [3, 4, 5], 2: [6, 7, 8],
                                3: [9, 10, 11], 4: [12, 13, 14], 5: [15, 16, 17],
                                6: [18, 19, 20], 7: [21, 22, 23], 8: [24, 25, 26],
                                9: [27, 28, 29]},

    # Client selection rate
    "client_selection_rate": 0.5,  # 4 clients

    # Architecture (unimodal)
    "unimodal_type": "cnn",
    "num_conv_layers": 1,
    "conv_kernel_size": 3,
    "conv_channels": 64,
    "conv_embed_dim": 128,
    "conv_dropout": 0.1,

    # Self-attention (multimodal fusion)
    "att_hid_dim": 256,
    "att_proj_dropout": 0.2,
    "att_num_contexts": 4,

}


SleepEDF20Opt = {
    # Data parameters
    'name': 'sleepedf20',
    'num_total_clients': 20,
    'client_ids': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    'input_dim': 5,
    'classes': [0, 1, 2, 3, 4],
    'num_classes': 5,
    'sampling_rate': 100,
    'seq_len': 30 * 100,   # 30 seconds * 100 Hz
    'sensor_list': ['EEG Fpz-Cz', 'EEG Pz-Oz', 'EOG horizontal', 'Resp oro-nasal', 'EMG submental'],
    'num_sensors': 5,
    'feature_columns': ['EEG Fpz-Cz', 'EEG Pz-Oz', 'EOG horizontal', 'Resp oro-nasal', 'EMG submental'],
    'all_columns_ordered': ['EEG Fpz-Cz', 'EEG Pz-Oz', 'EOG horizontal', 'Resp oro-nasal', 'EMG submental',
                            'label', 'user_id'],  # in .csv file => later we change 'user' to 'user_id'
    'sensor_to_featureColumn': {0: [0], 1: [1], 2: [2], 3: [3], 4: [4]},

    # Client selection rate
    "client_selection_rate": 0.3,

    # Architecture (unimodal)
    "unimodal_type": "cnn",
    "num_conv_layers": 1,
    "conv_kernel_size": 3,
    "conv_channels": 64,
    "conv_embed_dim": 128,
    "conv_dropout": 0.1,

    # Self-attention (multimodal fusion)
    "att_hid_dim": 256,
    "att_proj_dropout": 0.2,
    "att_num_contexts": 4,

}


# ================================================= #
# Tuned hyperparameters for each method and dataset #
# ================================================= #
METHOD_CONFIGS = {
    "pamap2": {
        "fedavg":    {'local_lr': 0.001},
        "fedper":    {'local_lr': 0.001},
        "fedprox":   {'local_lr': 0.001, 'fedprox_mu': 0.1},
        "moon":      {'local_lr': 0.001, 'moon_mu': 10, 'moon_temp': 0.5},
        "pmcmfl":    {'local_lr': 0.001, 'pmcmfl_clip_loss_weight': 0.1},
        "fedrod":    {'local_lr': 0.001},
        "pepsy":     {'local_lr': 0.01},
        "flux":      {'local_lr': 0.001},
    },

    "realworldhar": {
        "fedavg":    {'local_lr': 0.03},
        "fedper":    {'local_lr': 0.03},
        "fedprox":   {'local_lr': 0.03, 'fedprox_mu': 0.01},
        "moon":      {'local_lr': 0.03, 'moon_mu': 0.1, 'moon_temp': 0.5},
        "pmcmfl":    {'local_lr': 0.03, 'pmcmfl_clip_loss_weight': 0.01},
        "fedrod":    {'local_lr': 0.03},
        "pepsy":     {'local_lr': 0.01},
        "flux":      {'local_lr': 0.001},
    },

    "sleepedf20": {
        "fedavg":    {'local_lr': 0.03},
        "fedper":    {'local_lr': 0.001},
        "fedprox":   {'local_lr': 0.01, 'fedprox_mu': 0.01},
        "moon":      {'local_lr': 0.03, 'moon_mu': 10, 'moon_temp': 1},
        "pmcmfl":    {'local_lr': 0.001, 'pmcmfl_clip_loss_weight': 0.5},
        "fedrod":    {'local_lr': 0.01},
        "pepsy":     {'local_lr': 0.001},
        "flux":      {'local_lr': 0.001},
    },

}

# Hyperparameter keys for each method
METHOD_HP_KEYS = {
    "fedavg":    ['local_lr'],
    "fedper":    ['local_lr'],
    "fedprox":   ['local_lr', 'fedprox_mu'],
    "moon":      ['local_lr', 'moon_mu', 'moon_temp'],
    "pmcmfl":    ['local_lr', 'pmcmfl_clip_loss_weight'],
    "fedrod":    ['local_lr'],
    "flux":      ['local_lr'],
    "pepsy":     ['local_lr'],
}

# Common to all datasets
COMMON_CONFIGS = {
    'global_rounds': 200,
    'local_bs': 32,
    'optimizer': None,
    'local_epochs': 3,
    'local_wd': 5e-5
}

DEFAULT_OPTIMIZER = 'sgd'
METHOD_DEFAULT_OPTIMIZER = {
    "new_method": "your_default_optimizer", # Add if you have method-specific defaults
}

def apply_default_optimizer(dataset_configs, method_name):
    """
    Fill optimizer with a method-aware default when tuned HPs or CLI overrides
    did not set it.
    """
    if dataset_configs.get('optimizer') is None:
        dataset_configs['optimizer'] = METHOD_DEFAULT_OPTIMIZER.get(
            method_name, DEFAULT_OPTIMIZER
        )

def load_dataset_configs(dataset_name):
    if dataset_name == 'pamap2':
        dataset_configs = PAMAP2Opt
    elif dataset_name == 'realworldhar':
        dataset_configs = RealWorldHAROpt
    elif dataset_name == 'sleepedf20':
        dataset_configs = SleepEDF20Opt
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    for cfg in COMMON_CONFIGS:
        dataset_configs[cfg] = COMMON_CONFIGS[cfg]
    return dataset_configs


def get_tuned_hyperparameters(dataset_name, method_name):
    if is_method_tuned(dataset_name, method_name):
        return METHOD_CONFIGS[dataset_name][method_name]
    else:
        raise ValueError(f"Tuned hyperparameters for {method_name} on {dataset_name} are not available.")


def is_method_tuned(dataset_name, method_name):
    return dataset_name in METHOD_CONFIGS and method_name in METHOD_CONFIGS[dataset_name]


def get_method_hp_keys(method_name):
    if method_name in METHOD_HP_KEYS:
        return METHOD_HP_KEYS[method_name]
    else:
        raise ValueError(f"Unknown method: {method_name}. Available methods: {list(METHOD_HP_KEYS.keys())}")


def get_sensor_to_featureColumn_mapping(dataset_name):
    if dataset_name == 'pamap2':
        return PAMAP2Opt['sensor_to_featureColumn']
    elif dataset_name == 'realworldhar':
        return RealWorldHAROpt['sensor_to_featureColumn']
    elif dataset_name == 'sleepedf20':
        return SleepEDF20Opt['sensor_to_featureColumn']
    else:
        raise ValueError(f"Unknown or unimplemented dataset: {dataset_name}")