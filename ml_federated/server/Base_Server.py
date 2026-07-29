# ml_federated/base/BaseServer.py

import time
from tqdm import tqdm
from abc import ABC, abstractmethod



class Base_Server(ABC):
    """Base class for federated learning servers with common functionality."""

    def __init__(self, model, dataset_configs, device, args):
        self.model = model
        self.dataset_configs = dataset_configs
        self.dataset_name = dataset_configs['name']
        self.global_rounds = dataset_configs['global_rounds']
        self.device = device
        self.args = args
        self.method = args.method
        self.num_modalities = self.dataset_configs['num_sensors']

        self.eval_every = args.eval_every
        self.current_round = 1
        self.perf_stats = {}     # Performance statistics for each round
        self.best_model = None   # Best model found during training
        self.best_val_f1 = 0     # Best F1 macro score found during training

        # Client-related attributes
        self.clients = {}
        self.client_ids = dataset_configs['client_ids']
        self.num_total_clients = dataset_configs['num_total_clients']
        self.client_selection_rate = dataset_configs['client_selection_rate']
        self.num_selected_clients = int(self.num_total_clients * self.client_selection_rate)

        self.initialize_server()

    def initialize_server(self):
        """Common server initialization."""
        start_time = time.time()
        # Initialize clients and build their data profiles
        self._setup_clients()
        print("Server initialized in {:.2f} seconds".format(time.time() - start_time))


    @abstractmethod
    def _create_client(self, client_id):
        """Create a client instance - to be implemented by subclasses."""
        pass


    def _setup_clients(self):
        """Common client setup logic."""
        remove_clients = []
        for client_id in tqdm(self.client_ids, desc="Initializing clients"):
            client = self._create_client(client_id)
            if client.train_loader is None:
                # remove it from self.client_ids
                remove_clients.append(client_id)
                print(f"Client {client_id} has no training data, excluding from FL process...")
                continue # Exclude if a client has no training data
            self.clients[client_id] = self._create_client(client_id)
        # Remove clients with no training data
        self.client_ids = [cid for cid in self.client_ids if cid not in remove_clients]
        print(f'Final client ids: {self.client_ids}')

    @abstractmethod
    def run(self):
        """Run federated learning & evaluation process - to be implemented by subclasses."""
        pass


    @abstractmethod
    def run_federated_training(self):
        """Run federated training - to be implemented by subclasses."""
        pass


    @abstractmethod
    def run_federated_evaluation(self, eval_on):
        """Run federated evaluation - to be implemented by subclasses."""
        pass


    @abstractmethod
    def select_clients(self):
        """Select clients for the current round - to be implemented by subclasses."""
        pass

