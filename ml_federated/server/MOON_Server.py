# ml_federated/server/MOON_Server.py

from ml_federated.server.FedAvg_Server import FedAvg_Server
from ml_federated.client.MOON_Client import MOON_Client



class MOON_Server(FedAvg_Server):
    """MOON Server class for federated learning."""
    def __init__(self, model, dataset_configs, device, args):
        super().__init__(model, dataset_configs, device, args)


    def _create_client(self, client_id):
        """Create a MOON client instance."""
        return MOON_Client(client_id, self.dataset_configs, self.device, self.args)
