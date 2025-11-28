import os
import pickle
import numpy as np
from client import FLClient
from server import FLServer
from sklearn.preprocessing import StandardScaler

# Config
NUM_CLIENTS = 5

'''
INPUT_COLS = [
    'ResponseTime','Availability','ReliabilityScore','Usability',
    'Credibility','Certification','CostSatisfaction','Prestige','Security'
]

INPUT_COLS = [
   'ReliabilityScore'
]
'''
INPUT_COLS = ['ResponseTime(ms)','Availability(%)','ReliabilityScore','Usability(%)','Credibility(%)','Certification','CostSatisfaction(%)','Prestige(1-5)','Security(%)']
LABEL_COL = 'TrustIndex'



LABEL_COL = 'TrustIndex'
LOCAL_EPOCHS = 3
BATCH_SIZE = 32
LR = 0.01
FED_ROUNDS = 10


def main():
    # Initialize clients
    clients = [
        FLClient(i, f"data/client_{i}.csv", INPUT_COLS, LABEL_COL)
        for i in range(NUM_CLIENTS)
    ]

    print("📊 Client dataset sizes:")
    for i, c in enumerate(clients):
        print(f" - Client {i}: {len(c.y)} samples")

    # Fit scalers on combined data
    all_X = np.vstack([c.X for c in clients])
    all_y = np.hstack([c.y for c in clients])

    scaler_X = StandardScaler().fit(all_X)
    scaler_y = StandardScaler().fit(all_y.reshape(-1, 1))

    # Apply scaling to each client
    for c in clients:
        c.X = scaler_X.transform(c.X)
        c.y = scaler_y.transform(c.y.reshape(-1, 1)).flatten()

    # Save scalers for inference
    with open("scaler_X.pkl", "wb") as f:
        pickle.dump(scaler_X, f)
    with open("scaler_y.pkl", "wb") as f:
        pickle.dump(scaler_y, f)

    # Initialize server
    server = FLServer(input_dim=len(INPUT_COLS))

    # Federated training loop
    for round in range(1, FED_ROUNDS + 1):
        print(f"\n=== Federated Round {round}/{FED_ROUNDS} ===")
        global_weights = server.get_global_weights()

        client_weights = []
        client_scales = []

        for c in clients:
            # Sync global weights
            c.set_weights(global_weights)

            # Local training
            updated_weights = c.local_train(
                epochs=LOCAL_EPOCHS,
                batch_size=BATCH_SIZE,
                lr=LR
            )
            client_weights.append(updated_weights)
            client_scales.append(len(c.y))  # weighted avg by dataset size

        # Aggregate on server
        new_weights = server.aggregate(client_weights, scales=client_scales)

        # Evaluate on all clients combined
        total_X = np.vstack([c.X for c in clients])
        total_y = np.hstack([c.y for c in clients])
        preds = total_X.dot(new_weights['w']) + new_weights['b']
        rmse = np.sqrt(np.mean((preds - total_y) ** 2))
        ss_res = np.sum((total_y - preds) ** 2)
        ss_tot = np.sum((total_y - np.mean(total_y)) ** 2)
        r2 = 1 - (ss_res / ss_tot)

        print(f"📉 Global model: RMSE={rmse:.4f}, R²={r2:.4f}")

    # Save final global model
    np.savez("global_model.npz", w=new_weights['w'], b=new_weights['b'])
    print("\n✅ Training complete. Saved global_model.npz")


if __name__ == "__main__":
    main()


'''
import os
import pickle
import numpy as np
from client import FLClient
from sklearn.preprocessing import StandardScaler

# Config
NUM_CLIENTS = 5
INPUT_COLS = [
    'ResponseTime','Availability','ReliabilityScore','Usability',
    'Credibility','Certification','CostSatisfaction','Prestige','Security'
]
LABEL_COL = 'TrustIndex'
LOCAL_EPOCHS = 5
BATCH_SIZE = 32
LR = 0.01
FED_ROUNDS = 50

def main():
    # Initialize clients
    clients = [
        FLClient(i, f"data/client_{i}.csv", INPUT_COLS, LABEL_COL)
        for i in range(NUM_CLIENTS)
    ]

    print("Client dataset sizes:")
    for i, c in enumerate(clients):
        print(f" - Client {i}: {len(c.y)} samples")

    # Fit scalers on all client data
    all_X = np.vstack([c.X for c in clients])
    all_y = np.hstack([c.y for c in clients])

    scaler_X = StandardScaler().fit(all_X)
    scaler_y = StandardScaler().fit(all_y.reshape(-1, 1))

    # Apply scaling to each client
    for c in clients:
        c.X = scaler_X.transform(c.X)
        c.y = scaler_y.transform(c.y.reshape(-1, 1)).flatten()

    # Save scalers
    with open("scaler_X.pkl", "wb") as f:
        pickle.dump(scaler_X, f)
    with open("scaler_y.pkl", "wb") as f:
        pickle.dump(scaler_y, f)

    # Initialize global weights
    global_weights = clients[0].get_weights()

    # Federated training
    for round in range(1, FED_ROUNDS + 1):
        print(f"=== Federated Round {round}/{FED_ROUNDS} ===")
        client_weights = []

        for c in clients:
            c.set_weights(global_weights)
            updated_weights = c.local_train(epochs=LOCAL_EPOCHS, batch_size=BATCH_SIZE, lr=LR)
            client_weights.append(updated_weights)

        # Aggregate weights (simple average)
        new_weights = {k: np.mean([cw[k] for cw in client_weights], axis=0)
                       for k in global_weights.keys()}
        global_weights = new_weights

        # Evaluate global model on all clients
        total_X = np.vstack([c.X for c in clients])
        total_y = np.hstack([c.y for c in clients])
        preds = total_X.dot(global_weights['w']) + global_weights['b']
        rmse = np.sqrt(np.mean((preds - total_y) ** 2))
        print(f"Global model evaluation: RMSE={rmse:.4f}")

    # Save global model
    np.savez("global_model.npz", w=global_weights['w'], b=global_weights['b'])
    print("Saved global_model.npz")

if __name__ == "__main__":
    main()


#//////////////////////////////////////////////////////////////////////
import numpy as np
import joblib
from client import FLClient
from model import LinearRegressionModel
from sklearn.preprocessing import StandardScaler

NUM_ROUNDS = 20
LOCAL_EPOCHS = 3
BATCH_SIZE = 32
LR = 0.001  # you can adjust smaller if divergence occurs
NB_CLIENTS = 5

def weighted_fedavg(updates):
    """Perform weighted FedAvg aggregation of client weights."""
    total_samples = sum(u['n_samples'] for u in updates)
    agg = None
    for u in updates:
        w = u['weights']
        n = u['n_samples']
        if agg is None:
            agg = {k: np.array(v, dtype=float) * (n / total_samples) for k, v in w.items()}
        else:
            for k in agg:
                agg[k] += np.array(w[k], dtype=float) * (n / total_samples)
    return agg

def evaluate_global_model(global_model, clients, scaler_X, scaler_y):
    """Evaluate the global model on all clients' raw data."""
    all_X = np.vstack([c.X for c in clients])
    all_y = np.concatenate([c.y for c in clients]).reshape(-1,1)

    Xs = scaler_X.transform(all_X)
    preds_scaled = global_model.predict(Xs).reshape(-1,1)
    preds = scaler_y.inverse_transform(preds_scaled).ravel()

    mse = float(np.mean((preds - all_y.ravel())**2))
    rmse = float(np.sqrt(mse))
    ss_res = float(np.sum((all_y.ravel() - preds)**2))
    ss_tot = float(np.sum((all_y.ravel() - np.mean(all_y))**2))
    r2 = 1.0 - ss_res/ss_tot if ss_tot > 0 else 0.0
    print(f"Global model evaluation: RMSE={rmse:.4f}, R2={r2:.4f}")

def main():
    INPUT_COLS = [
        'ResponseTime','Availability','Reliability','Usability',
        'Credibility','Certification','CostSatisfaction','Prestige','Security'
    ]
    LABEL_COL = 'TrustIndex'

    # initialize clients
    clients = [
        FLClient(i, f"data/client_{i}.csv", INPUT_COLS, label_col=LABEL_COL, seed=42)
        for i in range(NB_CLIENTS)
    ]

    print("Client dataset sizes:")
    for c in clients:
        print(f" - Client {c.client_id}: {len(c.X)} samples")

    all_X = np.vstack([c.X for c in clients])
    all_y = np.concatenate([c.y for c in clients]).reshape(-1,1)

    # compute and save global scalers
    scaler_X = StandardScaler().fit(all_X)
    scaler_y = StandardScaler().fit(all_y)
    joblib.dump(scaler_X, "scaler_X.pkl")
    joblib.dump(scaler_y, "scaler_y.pkl")
    print("Saved global scalers (scaler_X.pkl, scaler_y.pkl)")

    # distribute scalers to clients
    for c in clients:
        c.set_scalers(scaler_X.mean_, scaler_X.scale_, scaler_y.mean_[0], scaler_y.scale_[0])

    # initialize global model
    global_model = LinearRegressionModel(input_dim=all_X.shape[1])

    # federated training
    for rnd in range(NUM_ROUNDS):
        print(f"=== Federated Round {rnd+1}/{NUM_ROUNDS} ===")
        updates = []
        for c in clients:
            update = c.local_train(epochs=LOCAL_EPOCHS, batch_size=BATCH_SIZE, lr=LR)
            updates.append(update)
            print(f" Client {c.client_id} loss={update['loss']:.4f} (samples={update['n_samples']})")

        # aggregate client weights
        new_weights = weighted_fedavg(updates)
        global_model.set_weights(new_weights)

        # distribute global weights to clients
        for c in clients:
            c.set_weights(new_weights)

        # evaluate global model
        evaluate_global_model(global_model, clients, scaler_X, scaler_y)

    # save final global model
    np.savez("global_model.npz", **global_model.get_weights())
    print("Saved global_model.npz")

if __name__ == "__main__":
    main()



#/////////////////////////////////////////////////////////////////////////////////
# train_federated.py

"""Federated training loop for linear regression with global scaling and weighted FedAvg."""

import numpy as np
import joblib
from client import FLClient
from model import LinearRegressionModel
from sklearn.preprocessing import StandardScaler

# CONFIG
NUM_ROUNDS = 4
LOCAL_EPOCHS = 1
BATCH_SIZE = 32
LR = 0.001  # smaller LR for stability

def weighted_fedavg(updates):
    """Aggregate weights proportional to client dataset size."""
    total_samples = sum(u['n_samples'] for u in updates)
    agg = None
    for u in updates:
        w = u['weights']
        n = u['n_samples']
        if agg is None:
            agg = {k: np.array(v, dtype=float) * (n / total_samples) for k, v in w.items()}
        else:
            for k in agg:
                agg[k] += np.array(w[k], dtype=float) * (n / total_samples)
    return agg

def evaluate_global_model(global_model, clients, scaler_X, scaler_y):
    """Evaluate on all clients combined (scaled features, inverse-transform predictions)."""
    all_X = np.vstack([c.X for c in clients])
    all_y = np.concatenate([c.y for c in clients]).reshape(-1,1)

    Xs = scaler_X.transform(all_X)
    preds_scaled = global_model.predict(Xs).reshape(-1,1)
    preds = scaler_y.inverse_transform(preds_scaled).ravel()

    mse = float(np.mean((preds - all_y.ravel())**2))
    rmse = float(np.sqrt(mse))
    ss_res = float(np.sum((all_y.ravel() - preds)**2))
    ss_tot = float(np.sum((all_y.ravel() - np.mean(all_y))**2))
    r2 = 1.0 - ss_res/ss_tot if ss_tot>0 else 0.0
    print(f"Global model evaluation: RMSE={rmse:.4f}, R2={r2:.4f}")

def main():
    # ===== Load clients =====
    import os, csv
    INPUT_COLS = [
        'ResponseTime(ms)','Availability(%)','ReliabilityScore','Usability(%)',
        'Credibility(%)','Certification','CostSatisfaction(%)','Prestige(1-5)','Security(%)'
    ]
    LABEL_COL = 'TrustIndex'

    n_clients = 22
    clients = [
        FLClient(i, f"data/partition_sp_{i}.csv", INPUT_COLS, label_col=LABEL_COL, seed=42)
        for i in range(n_clients)
    ]

    # ===== Compute global scaler =====
    all_X = np.vstack([c.X for c in clients])
    all_y = np.concatenate([c.y for c in clients]).reshape(-1,1)

    scaler_X = StandardScaler().fit(all_X)
    scaler_y = StandardScaler().fit(all_y)

    joblib.dump(scaler_X, "scaler_X.pkl")
    joblib.dump(scaler_y, "scaler_y.pkl")

    # ===== Set global scaler to clients =====
    for c in clients:
        c.set_scalers(scaler_X.mean_, scaler_X.scale_, scaler_y.mean_[0], scaler_y.scale_[0])

    # ===== Initialize global model =====
    global_model = LinearRegressionModel(input_dim=all_X.shape[1])

    # ===== Federated rounds =====
    for rnd in range(NUM_ROUNDS):
        print(f"=== Federated Round {rnd+1}/{NUM_ROUNDS} ===")
        updates = []
        for c in clients:
            update = c.local_train(epochs=LOCAL_EPOCHS, batch_size=BATCH_SIZE, lr=LR)
            updates.append(update)
            print(f" Client {c.client_id} loss={update['loss']:.4f}")

        # Aggregate and update global model
        new_weights = weighted_fedavg(updates)
        global_model.set_weights(new_weights)
        for c in clients:
            c.set_weights(new_weights)

        # Evaluate
        evaluate_global_model(global_model, clients, scaler_X, scaler_y)

    # Save final global model
    np.savez("global_model.npz", **global_model.get_weights())
    print("Saved global_model.npz")

if __name__ == "__main__":
    main()

#////////////////////////////////////////////////////////////////////

"""Federated training orchestrator for linear regression."""

from server import FLServer
from client import FLClient
from generate import generate as generate_data
import numpy as np
import csv
from typing import List, Dict

INPUT_COLS = [
    'ResponseTime(ms)','Availability(%)','ReliabilityScore','Usability(%)',
    'Credibility(%)','Certification','CostSatisfaction(%)','Prestige(1-5)','Security(%)'
]
LABEL_COL = 'TrustIndex'

def evaluate_global(server: FLServer, global_csv: str, input_cols: List[str], label_col: str) -> Dict[str, float]:
    """Evaluate global model on provided CSV dataset."""
    X, y = [], []
    with open(global_csv, newline='') as f:
        reader = csv.DictReader(f)
        for r in reader:
            X.append([float(r[c]) if c != 'Certification' else (1.0 if r[c].lower().startswith('y') else 0.0) for c in input_cols])
            y.append(float(r[label_col]))
    X, y = np.array(X), np.array(y)
    preds = server.global_model.predict(X)
    mse = float(np.mean((preds - y) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(preds - y)))
    ss_res = float(np.sum((y - preds) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return {'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2}

def run_federated_demo(
    data_dir: str = 'data',
    records_per_service: int = 200,
    rounds: int = 5,
    local_epochs: int = 2,
    batch_size: int = 32,
    lr: float = 0.01
) -> List[Dict[str, float]]:
    """Run federated training demo."""
    # Generate synthetic dataset
    generate_data(out_dir=data_dir, records_per_service=records_per_service, seed=42)

    n_clients = 22
    server = FLServer(input_dim=len(INPUT_COLS), seed=1)
    clients = [
        FLClient(i, f"{data_dir}/partition_sp_{i}.csv", INPUT_COLS, label_col=LABEL_COL, seed=1)
        for i in range(n_clients)
    ]

    # Initialize clients with global weights
    gw = server.get_global_weights()
    for c in clients:
        c.set_weights(gw)

    history = []
    for r in range(1, rounds + 1):
        print(f"=== Federated Round {r}/{rounds} ===")
        client_weights = []
        for c in clients:
            res = c.local_train(epochs=local_epochs, batch_size=batch_size, lr=lr)
            print(f" Client {c.client_id} loss={res['loss']:.4f}")
            client_weights.append(res['weights'])

        server.aggregate(client_weights)
        metrics = evaluate_global(server, f"{data_dir}/global.csv", INPUT_COLS, LABEL_COL)
        print(f" Global model RMSE: {metrics['rmse']:.4f}, R2: {metrics['r2']:.4f}")
        history.append({'round': r, **metrics})

    # Save final global model
    gw = server.get_global_weights()
    np.savez('global_model.npz', w=gw['w'], b=gw['b'])
    print('Saved global_model.npz')

    return history

if __name__ == '__main__':
    run_federated_demo(rounds=4, records_per_service=200)





#////////////////////////////////////////////////////////////////////////
"""Federated training orchestrator for linear regression."""
from server import FLServer
from client import FLClient
from generate import generate as generate_data
import numpy as np

INPUT_COLS = ['ResponseTime(ms)','Availability(%)','ReliabilityScore','Usability(%)','Credibility(%)','Certification','CostSatisfaction(%)','Prestige(1-5)','Security(%)']
LABEL_COL = 'TrustIndex'

def evaluate_global(server, global_csv, input_cols, label_col):
    import csv, math
    X = []; y = []
    with open(global_csv, newline='') as f:
        reader = csv.DictReader(f)
        for r in reader:
            X.append([float(r[c]) if c != 'Certification' else (1.0 if r[c].lower().startswith('y') else 0.0) for c in input_cols])
            y.append(float(r[label_col]))
    X = np.array(X); y = np.array(y)
    preds = server.global_model.predict(X)
    mse = float(np.mean((preds - y)**2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(preds - y)))
    ss_res = float(np.sum((y - preds)**2))
    ss_tot = float(np.sum((y - np.mean(y))**2))
    r2 = 1.0 - ss_res/ss_tot if ss_tot > 0 else 0.0
    return {'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2}

def run_federated_demo(data_dir='data', records_per_service=200, rounds=5, local_epochs=2, batch_size=32, lr=0.01):
    generate_data(out_dir=data_dir, records_per_service=records_per_service, seed=42)
    n_clients = 22
    server = FLServer(input_dim=len(INPUT_COLS), seed=1)
    clients = [FLClient(i, f"{data_dir}/partition_sp_{i}.csv", INPUT_COLS, label_col=LABEL_COL, seed=1) for i in range(n_clients)]
    gw = server.get_global_weights()
    for c in clients:
        c.set_weights(gw)
    history = []
    for r in range(1, rounds+1):
        print(f"=== Federated Round {r}/{rounds} ===")
        client_weights = []
        for c in clients:
            res = c.local_train(epochs=local_epochs, batch_size=batch_size, lr=lr)
            print(f" Client {c.client_id} loss={res['loss']:.4f}")
            client_weights.append(res['weights'])
        server.aggregate(client_weights)
        metrics = evaluate_global(server, f"{data_dir}/global.csv", INPUT_COLS, LABEL_COL)
        print(f" Global model RMSE: {metrics['rmse']:.4f}, R2: {metrics['r2']:.4f}")
        history.append({'round': r, **metrics})
    gw = server.get_global_weights()
    np.savez('global_model.npz', w=gw['w'], b=gw['b'])
    print('Saved global_model.npz')
    return history

if __name__ == '__main__':
    run_federated_demo(rounds=4, records_per_service=200)
'''