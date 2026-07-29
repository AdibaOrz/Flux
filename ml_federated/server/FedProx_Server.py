# ml_federated/server/FedProx_Server.py

from ml_federated.server.FedAvg_Server import FedAvg_Server
from ml_federated.client.FedProx_Client import FedProx_Client



class FedProx_Server(FedAvg_Server):
    """FedProx Server - a variant of FedAvg that introduces a proximal term to the loss function
     to mitigate the heterogeneity of client data."""
    def __init__(self, model, dataset_configs, device, args):
        super().__init__(model, dataset_configs, device, args)

    def _create_client(self, client_id):
        """Create a FedProx client instance."""
        return FedProx_Client(client_id, self.dataset_configs, self.device, self.args)

    # All the remaining methods are exactly the same as in FedAvg_Server