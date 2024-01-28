import torch
import numpy as np
from torch_geometric.utils import dense_to_sparse
from utils import load_all_midis
from torch_geometric_temporal.signal import StaticGraphTemporalSignal
    

class MIDIDatasetLoader(object):

    def __init__(self):
        super(MIDIDatasetLoader, self).__init__()
        self._read_midi_data()

    def _read_midi_data(self):

        # build adjacency matrix for Vogel's Tonnnetz graph
        A = np.zeros((128, 128))
        for i in range(128):
            # octave
            if i + 12 < 128:
                A[i, i + 12] = 1
                A[i + 12, i] = 1
            # fifth
            if i + 7 < 128:
                A[i, i + 7] = 1
                A[i + 7, i] = 1
            # major triad
            if i + 4 < 128:
                A[i, i + 4] = 1
                A[i + 4, i] = 1
            # minor triad
            if i + 3 < 128:
                A[i, i + 3] = 1
                A[i + 3, i] = 1
        X = load_all_midis()

        # Normalise as in DCRNN paper (via Z-Score Method)
        means = np.mean(X, axis=(0, 2))
        X = X - means.reshape(1, -1, 1)
        stds = np.std(X, axis=(0, 2))
        X = X / stds.reshape(1, -1, 1)

        self.A = torch.from_numpy(A)
        self.X = torch.from_numpy(X)

    def _get_edges_and_weights(self):
        edge_indices, values = dense_to_sparse(self.A)
        edge_indices = edge_indices.numpy()
        values = values.numpy()
        self.edges = edge_indices
        self.edge_weights = values

    def _generate_task(self, num_timesteps_in: int = 12, num_timesteps_out: int = 12):
        """Uses the node features of the graph and generates a feature/target
        relationship of the shape
        (num_nodes, num_node_features, num_timesteps_in) -> (num_nodes, num_timesteps_out)
        predicting the average traffic speed using num_timesteps_in to predict the
        traffic conditions in the next num_timesteps_out

        Args:
            num_timesteps_in (int): number of timesteps the sequence model sees
            num_timesteps_out (int): number of timesteps the sequence model has to predict
        """
        indices = [
            (i, i + (num_timesteps_in + num_timesteps_out))
            for i in range(self.X.shape[2] - (num_timesteps_in + num_timesteps_out) + 1)
        ]

        # Generate observations
        features, target = [], []
        for i, j in indices:
            features.append((self.X[:, :, i : i + num_timesteps_in]).numpy())
            target.append((self.X[:, 0, i + num_timesteps_in : j]).numpy())

        self.features = features
        self.targets = target

    def get_dataset(
        self, num_timesteps_in: int = 12, num_timesteps_out: int = 12
    ) -> StaticGraphTemporalSignal:
        """Returns data iterator for METR-LA dataset as an instance of the
        static graph temporal signal class.

        Return types:
            * **dataset** *(StaticGraphTemporalSignal)* - The METR-LA traffic
                forecasting dataset.
        """
        self._get_edges_and_weights()
        self._generate_task(num_timesteps_in, num_timesteps_out)
        dataset = StaticGraphTemporalSignal(
            self.edges, self.edge_weights, self.features, self.targets
        )

        return dataset