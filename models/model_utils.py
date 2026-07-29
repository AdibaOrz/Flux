# models/model_utils.py

from models.common.task_heads import MLP
from models.common.unimodals import CNNTimeSeries
from models.common.multimodal_model import MultimodalModel
from models.common.fusions import MultiContextAdditiveAttention



def get_model(dataset_configs, fusion_type, method, args=None):
    # ============================================== #
    ## ==== Method-agnostic (common) components ==== ##
    # ============================================== #

    ## ==== Unimodal encoders === ##
    encoders = []
    concat_embed_dim = 0
    sensor_to_feat_col = dataset_configs['sensor_to_featureColumn']  # {0: [0], 1: [1], 2: [2], 3: [3], 4: [4]}
    for sensor_id, feat_cols in sensor_to_feat_col.items():
        if dataset_configs['unimodal_type'] == 'cnn':
            encoder = CNNTimeSeries(feat_dim=len(feat_cols),
                                    embed_dim=dataset_configs['conv_embed_dim'],
                                    conv_channels=dataset_configs['conv_channels'],
                                    num_conv_layers=dataset_configs['num_conv_layers'],
                                    kernel_size=dataset_configs['conv_kernel_size'],
                                    dropout=dataset_configs['conv_dropout'])
            concat_embed_dim += dataset_configs['conv_embed_dim']
        else:
            raise ValueError(f"Invalid unimodal_type {dataset_configs['unimodal_type']}.")
        encoders.append(encoder)

    ## ==== Multimodal fusion and classification head === ##
    # 1. Multimodal fusion model
    if fusion_type == 'attention':
        head_input_dim = dataset_configs['conv_embed_dim']
        fusion = MultiContextAdditiveAttention(input_dim=dataset_configs['conv_embed_dim'],
                                               hidden_dim=dataset_configs['att_hid_dim'],
                                               num_contexts=dataset_configs['att_num_contexts'],
                                               proj_dropout=dataset_configs['att_proj_dropout'])
    else:
        raise ValueError(f"Invalid fusion type {fusion_type}.")

    # PEPSY concatenates [SpecificEmb, Feature, Prompt] (3x) for each modality
    # and then flattens all modalities (Num_Sensors * 3 * Embed_Dim)
    if method == 'pepsy':
        num_sensors = dataset_configs['num_sensors']
        embed_dim = dataset_configs['conv_embed_dim']
        head_input_dim = num_sensors * 3 * embed_dim

    # 2. Classification head
    num_classes = dataset_configs['num_classes']
    head = MLP(head_input_dim, num_classes)


    # ====================================== #
    ## ==== Method-specific components ==== ##
    # ====================================== #
    if method in ['fedavg', 'fedprox', 'fedper', 'fedrod']:
        return MultimodalModel(encoders, fusion, head, dataset_configs)
    elif method == 'moon':
        from models.method_specific.moon import ProjectionHead
        moon_prj_head = ProjectionHead(head_input_dim)
        return MultimodalModel(encoders, fusion, head, dataset_configs, moon_prj_head=moon_prj_head)
    elif method == 'pmcmfl':
        from models.method_specific.pmcmfl import PmcmFLModel
        return PmcmFLModel(encoders, fusion, head, dataset_configs)
    elif method == 'pepsy':
        from models.method_specific.pepsy_model import PEPSYModel
        return PEPSYModel(encoders, dataset_configs, args)
    elif method == 'flux':
        from models.method_specific.flux import FLUX_Model
        return FLUX_Model(args, encoders, fusion, head, head_input_dim, dataset_configs)
    else:
        raise ValueError(f'Unsupported method {method}. Check the method name or implement it!')

