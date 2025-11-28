import pandas as pd
import numpy as np
from model import LinearRegressionModel
from typing import Dict, Any

class FLClient:
    def __init__(self, client_id: int, data_path: str, input_cols: list, label_col: str, seed: int = 0):
        self.client_id = client_id
        self.data_path = data_path
        self.input_cols = input_cols
        self.label_col = label_col

        # Load dataset
        df = pd.read_csv(self.data_path)

        # Ensure all expected columns exist
        for col in self.input_cols:
            if col not in df.columns:
                df[col] = 0.0  # Fill missing input columns with 0

        if self.label_col not in df.columns:
            df[self.label_col] = 0.0  # Fill missing label with 0

        self.X = df[self.input_cols].values
        self.y = df[self.label_col].values

        # Initialize model
        self.model = LinearRegressionModel(input_dim=len(self.input_cols), seed=seed)

    def get_weights(self) -> Dict[str, Any]:
        return self.model.get_weights()

    def set_weights(self, weights: Dict[str, Any]):
        self.model.set_weights(weights)

    def local_train(self, epochs: int = 1, batch_size: int = 32, lr: float = 0.01) -> Dict[str, Any]:
        N = self.X.shape[0]

        for epoch in range(epochs):
            # Shuffle data
            perm = np.random.permutation(N)
            X_shuffled = self.X[perm]
            y_shuffled = self.y[perm]

            for i in range(0, N, batch_size):
                X_batch = X_shuffled[i:i + batch_size]
                y_batch = y_shuffled[i:i + batch_size]

                loss, grads = self.model.loss_and_grad(X_batch, y_batch)
                self.model.apply_gradients(grads, lr=lr)

        # Return updated weights
        return self.get_weights()

#///////////////////////////////////////////////////////////////
'''
"""FL client (Service Provider) for regression task."""

import numpy as np
from model import LinearRegressionModel
from typing import List, Dict, Any

class FLClient:
    def __init__(
        self,
        client_id: int,
        csv_path: str,
        input_cols: List[str],
        label_col: str = 'TrustIndex',
        seed: int = 0
    ):
        """
        Initialize FL client with dataset and linear regression model.
        Args:
            client_id: Unique client identifier
            csv_path: Path to the client's CSV dataset
            input_cols: List of feature column names
            label_col: Name of target column
            seed: Random seed for reproducibility
        """
        self.client_id = client_id
        self.csv_path = csv_path
        self.input_cols = input_cols
        self.label_col = label_col
        self._load_data()
        self.model = LinearRegressionModel(input_dim=len(input_cols), seed=seed + client_id * 13)
        self.local_steps = 0
        self.rng = np.random.default_rng(seed + client_id)

    def _load_data(self) -> None:
        """Load CSV data into memory."""
        import csv
        self.rows = []
        with open(self.csv_path, newline='') as f:
            reader = csv.DictReader(f)
            for r in reader:
                self.rows.append(r)

    def _encode(self, r: Dict[str, str], col: str) -> float:
        """
        Encode column value to float for regression.
        Args:
            r: Row dictionary
            col: Column name
        Returns:
            Encoded float value
        """
        if col == 'Certification':
            v = r.get(col, '').strip()
            return 1.0 if v.lower().startswith('y') else 0.0
        try:
            return float(r[col])
        except (ValueError, KeyError):
            return 0.0  # fallback for missing/invalid data

    def get_batch(self, batch_size: int = 32) -> (np.ndarray, np.ndarray):
        """
        Sample a random batch from local data.
        Args:
            batch_size: Number of samples in batch
        Returns:
            X: Feature array, shape (batch_size, input_dim)
            y: Target array, shape (batch_size,)
        """
        idx = self.rng.choice(len(self.rows), size=min(batch_size, len(self.rows)), replace=False)
        X, y = [], []
        for i in idx:
            r = self.rows[i]
            X.append([self._encode(r, c) for c in self.input_cols])
            y.append(float(r.get(self.label_col, 0.0)))
        return np.array(X, dtype=float), np.array(y, dtype=float)

    def get_weights(self) -> Dict[str, Any]:
        """Return model weights."""
        return self.model.get_weights()

    def set_weights(self, weights: Dict[str, Any]) -> None:
        """Set model weights."""
        self.model.set_weights(weights)

    def local_train(self, epochs: int = 1, batch_size: int = 32, lr: float = 0.01) -> Dict[str, Any]:
        """
        Train local model for given epochs.
        Args:
            epochs: Number of training epochs
            batch_size: Size of each mini-batch
            lr: Learning rate
        Returns:
            Dictionary with client_id, updated weights, and average loss
        """
        total_loss = 0.0
        iters = 0
        for _ in range(epochs):
            X, y = self.get_batch(batch_size)
            loss, grads = self.model.loss_and_grad(X, y)
            self.model.apply_gradients(grads, lr=lr)
            total_loss += float(loss)
            iters += 1
            self.local_steps += 1
        avg_loss = total_loss / max(1, iters)
        return {'client_id': self.client_id, 'weights': self.get_weights(), 'loss': avg_loss}





#////////////////////////////////////////////////////////////////////////////
import numpy as np
from model import LinearRegressionModel

class FLClient:
    def __init__(self, client_id, csv_path, input_cols, label_col='TrustIndex', seed=0):
        self.client_id = client_id
        self.csv_path = csv_path
        self.input_cols = input_cols
        self.label_col = label_col
        self._load_data()
        self.model = LinearRegressionModel(input_dim=len(input_cols), seed=seed+client_id*13)
        self.local_steps = 0

    def _load_data(self):
        import csv
        rows = []
        with open(self.csv_path, newline='') as f:
            reader = csv.DictReader(f)
            for r in reader:
                rows.append(r)
        self.rows = rows

    def _encode(self, r, col):
        # handle Certification column (Yes/No) -> 1.0/0.0 otherwise cast to float
        if col == 'Certification':
            v = r.get(col, '').strip()
            return 1.0 if v.lower().startswith('y') else 0.0
        return float(r[col])

    def get_batch(self, batch_size=32):
        idx = np.random.choice(len(self.rows), size=min(batch_size, len(self.rows)), replace=False)
        X = []
        y = []
        for i in idx:
            r = self.rows[i]
            X.append([self._encode(r, c) for c in self.input_cols])
            y.append(float(r[self.label_col]))
        return np.array(X, dtype=float), np.array(y, dtype=float)

    def get_weights(self):
        return self.model.get_weights()

    def set_weights(self, weights):
        self.model.set_weights(weights)

    def local_train(self, epochs=1, batch_size=32, lr=0.01):
        total_loss = 0.0
        iters = 0
        for e in range(epochs):
            X, y = self.get_batch(batch_size)
            loss, grads = self.model.loss_and_grad(X, y)
            self.model.apply_gradients(grads, lr=lr)
            total_loss += float(loss)
            iters += 1
            self.local_steps += 1
        avg_loss = total_loss / max(1, iters)
        return {'client_id': self.client_id, 'weights': self.get_weights(), 'loss': avg_loss}
        '''
