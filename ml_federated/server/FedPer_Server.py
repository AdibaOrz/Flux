# ml_federated/server/FedPer_Server.py

from ml_federated.server.FedAvg_Server import FedAvg_Server
from ml_federated.client.FedPer_Client import FedPer_Client


class FedPer_Server(FedAvg_Server):
    """FedPer Server. During aggregation, ignore private parameters
     - private params: head (+ fusion if --private_fusion is set)
    """

    def __init__(self, model, dataset_configs, device, args):
        super().__init__(model, dataset_configs, device, args)
        self.private_fusion = args.private_fusion

    def _create_client(self, client_id):
        return FedPer_Client(client_id, self.dataset_configs, self.device, self.args)

    def _should_use_global_model_selection(self):
        return False

    def _should_aggregate_parameter(self, key):
        """FedPer only aggregates shared parameters."""
        # Always aggregate encoders
        if key.startswith('encoders.'):
            return True

        # Head is always private
        if key.startswith('head.'):
            return False

        # Fusion is private if private_fusion is True
        if key.startswith('fusion.') and self.private_fusion:
            return False

        # All other parameters are shared
        return True