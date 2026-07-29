# ml_federated/server/FedRoD_Server.py

from ml_federated.server.FedAvg_Server import FedAvg_Server
from ml_federated.client.FedRoD_Client import FedRoD_Client



class FedRoD_Server(FedAvg_Server):
    """
    Server for FedRoD.
    """
    def _create_client(self, client_id):
        return FedRoD_Client(client_id, self.dataset_configs, self.device, self.args)


    def _should_use_global_model_selection(self):
        return False # Will use client-specific model selection