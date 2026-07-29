# ml_federated/server/server_utils.py

from ml_federated.server.FedAvg_Server import FedAvg_Server
from ml_federated.server.FedProx_Server import FedProx_Server
from ml_federated.server.FedPer_Server import FedPer_Server
from ml_federated.server.FedRoD_Server import FedRoD_Server
from ml_federated.server.PmcmFL_Server import PmcmFL_Server
from ml_federated.server.MOON_Server import MOON_Server
from ml_federated.server.PEPSY_Server import PEPSY_Server
from ml_federated.server.FLUX_Server import FLUX_Server


def get_server(model, dataset_configs, device, args):
    if args.method == "fedavg":
        target_server = FedAvg_Server(model, dataset_configs, device, args)
    elif args.method == "fedprox":
        target_server = FedProx_Server(model, dataset_configs, device, args)
    elif args.method == "moon":
        target_server = MOON_Server(model, dataset_configs, device, args)
    elif args.method == "fedper":
        target_server = FedPer_Server(model, dataset_configs, device, args)
    elif args.method == "pmcmfl":
        target_server = PmcmFL_Server(model, dataset_configs, device, args)
    elif args.method == "fedrod":
        target_server = FedRoD_Server(model, dataset_configs, device, args)
    elif args.method == "pepsy":
        target_server = PEPSY_Server(model, dataset_configs, device, args)
    elif args.method == "flux":
        target_server = FLUX_Server(model, dataset_configs, device, args)
    else:
        raise NotImplementedError(f"Method {args.method} not implemented")
    return target_server