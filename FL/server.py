"""FL server for regression (FedAvg aggregation)."""

from typing import List, Dict, Optional
import numpy as np
from model import LinearRegressionModel

class FLServer:
    def __init__(self, input_dim: int, seed: int = 0):
        """
        Initialize the global model.
        Args:
            input_dim: Number of features
            seed: Random seed
        """
        self.global_model = LinearRegressionModel(input_dim, seed=seed)
        self.round = 0

    def get_global_weights(self) -> Dict[str, np.ndarray]:
        """Return current global model weights."""
        return self.global_model.get_weights()

    def set_global_weights(self, weights: Dict[str, np.ndarray]) -> None:
        """Set global model weights."""
        self.global_model.set_weights(weights)

    def aggregate(self, client_weights: List[Dict[str, np.ndarray]],
                  scales: Optional[List[float]] = None) -> Dict[str, np.ndarray]:
        """
        Aggregate client models using (weighted) FedAvg.
        Args:
            client_weights: List of client weight dictionaries {'w': ..., 'b': ...}
            scales: Optional list of weights for weighted average
        Returns:
            Updated global weights
        """
        if not client_weights:
            raise ValueError("No client weights provided for aggregation.")

        n = len(client_weights)
        if scales is None:
            scales = [1.0 / n] * n
        else:
            total = sum(scales)
            scales = [float(x)/total for x in scales]

        # Initialize accumulators
        acc_w = np.zeros_like(client_weights[0]['w'])
        acc_b = 0.0

        for cw, sc in zip(client_weights, scales):
            acc_w += cw['w'] * sc
            acc_b += float(cw['b']) * sc

        new_weights = {'w': acc_w, 'b': acc_b}
        self.set_global_weights(new_weights)
        self.round += 1

        return new_weights



'''
"""FL server for regression (FedAvg aggregation)."""
import numpy as np
from model import LinearRegressionModel

class FLServer:
    def __init__(self, input_dim, seed=0):
        self.global_model = LinearRegressionModel(input_dim, seed=seed)
        self.round = 0

    def get_global_weights(self):
        return self.global_model.get_weights()

    def set_global_weights(self, weights):
        self.global_model.set_weights(weights)

    def aggregate(self, client_weights, scales=None):
        if not client_weights:
            return
        n = len(client_weights)
        if scales is None:
            scales = [1.0/n] * n
        else:
            s = sum(scales); scales = [float(x)/s for x in scales]
        acc_w = None
        acc_b = 0.0
        for cw, sc in zip(client_weights, scales):
            if acc_w is None:
                acc_w = cw['w'] * sc
            else:
                acc_w += cw['w'] * sc
            acc_b += float(cw['b']) * sc
        new_weights = {'w': acc_w, 'b': acc_b}
        self.set_global_weights(new_weights)
        self.round += 1
        return new_weights
'''