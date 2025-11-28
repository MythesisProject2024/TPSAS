# utils.py

import numpy as np

def evaluate_global(model, clients, scaler_X, scaler_y):
    """
    Evaluate global linear regression model on all client data (original scale).

    Args:
        model: LinearRegressionModel trained in scaled space
        clients: list of FLClient
        scaler_X: fitted StandardScaler for features
        scaler_y: fitted StandardScaler for target
    """
    # Collect all client data
    all_X, all_y = [], []
    for c in clients:
        Xi, yi = c.get_full_data()
        all_X.append(Xi)
        all_y.append(yi)
    X = np.vstack(all_X)
    y = np.concatenate(all_y)

    # Scale inputs
    Xs = scaler_X.transform(X)

    # Predict in scaled space
    preds_scaled = model.predict(Xs).reshape(-1, 1)

    # Inverse transform to original scale
    preds = scaler_y.inverse_transform(preds_scaled).ravel()

    # Compute metrics in original scale
    mse = float(np.mean((preds - y) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(preds - y)))
    ss_res = float(np.sum((y - preds) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    print(f" Global Eval -> RMSE={rmse:.4f}, MAE={mae:.4f}, R2={r2:.4f}")
    return {'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2}

