import numpy as np
from typing import Dict, Any

class LinearRegressionModel:
    def __init__(self, input_dim: int, seed: int = 0):
        """
        Initialize weights and bias for linear regression.
        Args:
            input_dim: Number of features.
            seed: Random seed for reproducibility.
        """
        rng = np.random.RandomState(seed)
        self.w = rng.randn(input_dim) * 0.01
        self.b = 0.0

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Compute linear predictions.
        Args:
            x: Input data, shape (N, input_dim) or (input_dim,)
        Returns:
            Predictions, shape (N,)
        """
        x = np.atleast_2d(x)
        return x.dot(self.w) + self.b

    def get_weights(self) -> Dict[str, Any]:
        """Return a copy of model weights."""
        return {'w': self.w.copy(), 'b': float(self.b)}

    def set_weights(self, weights: Dict[str, Any]) -> None:
        """Set model weights from a dictionary."""
        self.w = weights['w'].copy()
        self.b = float(weights['b'])

    def loss_and_grad(self, x: np.ndarray, y: np.ndarray) -> (float, Dict[str, np.ndarray]):
        """
        Compute MSE loss and gradients for weights and bias.
        Args:
            x: Input data, shape (N, input_dim)
            y: Target values, shape (N,)
        Returns:
            loss: Mean squared error
            grads: Dictionary with gradients {'w': ..., 'b': ...}
        """
        x = np.atleast_2d(x)
        y = np.atleast_1d(y)
        N = x.shape[0]

        preds = x.dot(self.w) + self.b
        error = preds - y
        loss = float(np.mean(error ** 2))

        grad_w = (2.0 / N) * x.T.dot(error)
        grad_b = (2.0 / N) * np.sum(error)

        return loss, {'w': grad_w, 'b': grad_b}

    def apply_gradients(self, grads: Dict[str, np.ndarray], lr: float = 0.01) -> None:
        """
        Update model parameters using gradients.
        Args:
            grads: Gradients dictionary {'w': ..., 'b': ...}
            lr: Learning rate
        """
        self.w -= lr * grads['w']
        self.b -= lr * grads['b']
