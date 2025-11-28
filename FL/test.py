
#THIS VERSION DSPLAY PREDICT TRUSTINDEX VALUES AND SAVE IT
import numpy as np
import pandas as pd
import joblib
from model import LinearRegressionModel

def main():
    # Define the input features
    INPUT_COLS = [
        "ResponseTime(ms)",
        "Availability(%)",
        "ReliabilityScore",
        "Usability(%)",
        "Credibility(%)",
        "Certification",
        "CostSatisfaction(%)",
        "Prestige(1-5)",
        "Security(%)"
    ]

    LABEL_COL = 'TrustIndex'

    # Load global scalers
    scaler_X = joblib.load("scaler_X.pkl")
    scaler_y = joblib.load("scaler_y.pkl")

    # Load global test dataset
    nb = input('Enter client number to test its data from 0--4: ').strip()  # removes spaces
    dataTest = f"data/client_{nb}.csv"


    #test_df = pd.read_csv("data/global_test.csv")
    test_df = pd.read_csv(dataTest)

    # Split features/labels
    all_X = test_df[INPUT_COLS].values
    all_y = test_df[LABEL_COL].values.reshape(-1, 1)

    # Apply global scaling
    all_X_scaled = scaler_X.transform(all_X)

    # Load global model
    global_weights = np.load("global_model.npz")
    global_model = LinearRegressionModel(input_dim=len(INPUT_COLS))
    weights_dict = {k: global_weights[k] for k in global_weights.files}
    global_model.set_weights(weights_dict)

    # Predict
    preds_scaled = global_model.predict(all_X_scaled).reshape(-1, 1)
    preds = scaler_y.inverse_transform(preds_scaled).ravel()

    # Compute metrics
    mse = float(np.mean((preds - all_y.ravel()) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(preds - all_y.ravel())))
    ss_res = float(np.sum((all_y.ravel() - preds) ** 2))
    ss_tot = float(np.sum((all_y.ravel() - np.mean(all_y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # Display metrics
    print(f"✅ Global model evaluation on TEST dataset:{dataTest}")
    print(f"   RMSE={rmse:.4f}, MAE={mae:.4f}, R2={r2:.4f}")

    # Create a DataFrame with true vs predicted values
    results_df = test_df.copy()
    results_df['Predicted_TrustIndex'] = preds

    # Display first few rows
    print("\nSample of true vs predicted TrustIndex:")
    print(results_df[[LABEL_COL, 'Predicted_TrustIndex']].head(10))

    # Save results to CSV for further comparison
    results_df.to_csv("data/test_predictions.csv", index=False)
    print("\n✅ Predicted values saved to data/test_predictions.csv")

if __name__ == "__main__":
    main()



''''
import numpy as np
import pandas as pd
import joblib
from model import LinearRegressionModel

def main():
    # Define the input features
    INPUT_COLS = [
        "ResponseTime(ms)",
        "Availability(%)",
        "ReliabilityScore",
        "Usability(%)",
        "Credibility(%)",
        "Certification",
        "CostSatisfaction(%)",
        "Prestige(1-5)",
        "Security(%)"
    ]

    LABEL_COL = 'TrustIndex'

    # Load global scalers
    scaler_X = joblib.load("scaler_X.pkl")
    scaler_y = joblib.load("scaler_y.pkl")

    # ✅ Load global test dataset
    test_df = pd.read_csv("data/global_test.csv")
    #test_df = pd.read_csv("data/test.csv")
    #test_df = pd.read_csv("data/service_trust_dataset.csv")

    # Split features/labels
    all_X = test_df[INPUT_COLS].values
    all_y = test_df[LABEL_COL].values.reshape(-1, 1)

    # Apply global scaling
    all_X_scaled = scaler_X.transform(all_X)

    # Load global model
    global_weights = np.load("global_model.npz")
    global_model = LinearRegressionModel(input_dim=len(INPUT_COLS))
    weights_dict = {k: global_weights[k] for k in global_weights.files}
    global_model.set_weights(weights_dict)

    # Predict
    preds_scaled = global_model.predict(all_X_scaled).reshape(-1, 1)
    preds = scaler_y.inverse_transform(preds_scaled).ravel()

    # Compute metrics
    mse = float(np.mean((preds - all_y.ravel()) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(preds - all_y.ravel())))
    ss_res = float(np.sum((all_y.ravel() - preds) ** 2))
    ss_tot = float(np.sum((all_y.ravel() - np.mean(all_y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    print(f"✅ Global model evaluation on TEST dataset:")
    print(f"   RMSE={rmse:.4f}, MAE={mae:.4f}, R2={r2:.4f}")

if __name__ == "__main__":
    main()



# //////////////////////////////////////////THIS IS STABLE VERSION/////////////////////////////////////////////
import numpy as np
import joblib
from client import FLClient
from model import LinearRegressionModel

NB_CLIENTS = 5

def main():

    #INPUT_COLS = [
    #    'ResponseTime','Availability','ReliabilityScore','Usability',
    #    'Credibility','Certification','CostSatisfaction','Prestige','Security'
    #]
    LABEL_COL = 'TrustIndex'

    INPUT_COLS =    ['ResponseTime(ms)','Availability(%)','ReliabilityScore','Usability(%)','Credibility(%)','Certification','CostSatisfaction(%)','Prestige(1-5)','Security(%)']
    LABEL_COL = 'TrustIndex'

    # Load global scalers
    scaler_X = joblib.load("scaler_X.pkl")
    scaler_y = joblib.load("scaler_y.pkl")

    # Load clients
    clients = [
        FLClient(i, f"data/client_{i}.csv", INPUT_COLS, LABEL_COL)
        for i in range(NB_CLIENTS)
    ]

    # Gather all client data
    all_X = np.vstack([c.X for c in clients])
    all_y = np.concatenate([c.y for c in clients]).reshape(-1, 1)

    # Apply global scaling
    all_X_scaled = scaler_X.transform(all_X)

    # Load global model
    global_weights = np.load("global_model.npz")
    global_model = LinearRegressionModel(input_dim=len(INPUT_COLS))
    weights_dict = {k: global_weights[k] for k in global_weights.files}
    global_model.set_weights(weights_dict)

    # Predict
    preds_scaled = global_model.predict(all_X_scaled).reshape(-1, 1)
    preds = scaler_y.inverse_transform(preds_scaled).ravel()

    # Compute metrics
    mse = float(np.mean((preds - all_y.ravel()) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(preds - all_y.ravel())))
    ss_res = float(np.sum((all_y.ravel() - preds) ** 2))
    ss_tot = float(np.sum((all_y.ravel() - np.mean(all_y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    print(f"Test evaluation: RMSE={rmse:.4f}, MAE={mae:.4f}, R2={r2:.4f}")

if __name__ == "__main__":
    main()




#///////////////////////////////////////////////////////////////////////


"""Evaluate saved global linear model on all client data using global scalers."""

import numpy as np
import joblib
from client import FLClient
from model import LinearRegressionModel
from sklearn.preprocessing import StandardScaler

# CONFIG
INPUT_COLS = [
    'ResponseTime(ms)','Availability(%)','ReliabilityScore','Usability(%)',
    'Credibility(%)','Certification','CostSatisfaction(%)','Prestige(1-5)','Security(%)'
]
LABEL_COL = 'TrustIndex'
N_CLIENTS = 22  # must match train_federated.py

def load_clients():
    clients = []
    for i in range(N_CLIENTS):
        clients.append(FLClient(i, f"data/partition_sp_{i}.csv", INPUT_COLS, label_col=LABEL_COL, seed=42))
    return clients

def evaluate_global_model(global_model, clients, scaler_X, scaler_y):
    """Evaluate model across all client data."""
    all_X = np.vstack([c.X for c in clients])
    all_y = np.concatenate([c.y for c in clients]).reshape(-1,1)

    Xs = scaler_X.transform(all_X)
    preds_scaled = global_model.predict(Xs).reshape(-1,1)
    preds = scaler_y.inverse_transform(preds_scaled).ravel()

    mse = float(np.mean((preds - all_y.ravel())**2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(preds - all_y.ravel())))
    ss_res = float(np.sum((all_y.ravel() - preds)**2))
    ss_tot = float(np.sum((all_y.ravel() - np.mean(all_y))**2))
    r2 = 1.0 - ss_res/ss_tot if ss_tot>0 else 0.0

    print(f"Loaded model global_model.npz: RMSE={rmse:.4f}, MAE={mae:.4f}, R2={r2:.4f}")

def main():
    # Load clients
    clients = load_clients()

    # Load scalers
    scaler_X = joblib.load("scaler_X.pkl")
    scaler_y = joblib.load("scaler_y.pkl")

    # Set global scalers for clients
    for c in clients:
        c.set_scalers(scaler_X.mean_, scaler_X.scale_, scaler_y.mean_[0], scaler_y.scale_[0])

    # Load global model
    d = np.load("global_model.npz")
    global_model = LinearRegressionModel(input_dim=len(INPUT_COLS))
    global_model.set_weights({'w': d['w'], 'b': float(d['b'])})

    # Evaluate
    evaluate_global_model(global_model, clients, scaler_X, scaler_y)

if __name__ == "__main__":
    main()


#//////////////////////////////////////////////////////////////////////////
"""Evaluate saved global linear model on generated global data and print regression metrics."""

import numpy as np
import csv
import joblib
from model import LinearRegressionModel
from typing import List

INPUT_COLS: List[str] = [
    'ResponseTime(ms)','Availability(%)','ReliabilityScore','Usability(%)',
    'Credibility(%)','Certification','CostSatisfaction(%)','Prestige(1-5)','Security(%)'
]
LABEL_COL = 'TrustIndex'

def test_global(global_csv: str = 'data/global.csv', model_path: str = 'global_model.npz') -> None:
    """Evaluate the saved global linear regression model on the provided CSV dataset."""
    X, y = [], []
    with open(global_csv, newline='') as f:
        reader = csv.DictReader(f)
        for r in reader:
            X.append([float(r[c]) if c != 'Certification' else (1.0 if r[c].lower().startswith('y') else 0.0) for c in INPUT_COLS])
            y.append(float(r[LABEL_COL]))
    X, y = np.array(X), np.array(y)

    try:
        # === Load scalers ===
        scaler_X = joblib.load("scaler_X.pkl")
        scaler_y = joblib.load("scaler_y.pkl")

        # === Apply feature scaling ===
        Xs = scaler_X.transform(X)

        # === Load model (trained in scaled space) ===
        d = np.load(model_path)
        model = LinearRegressionModel(input_dim=X.shape[1])
        model.set_weights({'w': d['w'], 'b': float(d['b'])})

        # === Predict in scaled space, then inverse transform ===
        preds_scaled = model.predict(Xs).reshape(-1, 1)
        preds = scaler_y.inverse_transform(preds_scaled).ravel()

        # === Compute metrics on original scale ===
        mse = float(np.mean((preds - y) ** 2))
        rmse = float(np.sqrt(mse))
        mae = float(np.mean(np.abs(preds - y)))
        ss_res = float(np.sum((y - preds) ** 2))
        ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        print(f'Loaded model {model_path}: RMSE={rmse:.4f}, MAE={mae:.4f}, R2={r2:.4f}')
    except Exception as e:
        print('Failed to load model:', e)


if __name__ == '__main__':
    test_global()

#///////////////////////////////////////////////////////////////////////////

"""Evaluate saved global linear model on generated global data and print regression metrics."""

import numpy as np
from model import LinearRegressionModel
import csv
from typing import List

INPUT_COLS: List[str] = [
    'ResponseTime(ms)','Availability(%)','ReliabilityScore','Usability(%)',
    'Credibility(%)','Certification','CostSatisfaction(%)','Prestige(1-5)','Security(%)'
]
LABEL_COL = 'TrustIndex'

def test_global(global_csv: str = 'data/global.csv', model_path: str = 'global_model.npz') -> None:
    """
    Evaluate the saved global linear regression model on the provided CSV dataset.

    Args:
        global_csv: Path to the global CSV dataset
        model_path: Path to the saved global model (.npz)
    """
    X, y = [], []
    with open(global_csv, newline='') as f:
        reader = csv.DictReader(f)
        for r in reader:
            X.append([float(r[c]) if c != 'Certification' else (1.0 if r[c].lower().startswith('y') else 0.0) for c in INPUT_COLS])
            y.append(float(r[LABEL_COL]))
    X, y = np.array(X), np.array(y)

    try:
        d = np.load(model_path)
        model = LinearRegressionModel(input_dim=X.shape[1])
        model.set_weights({'w': d['w'], 'b': float(d['b'])})
        preds = model.predict(X)

        mse = float(np.mean((preds - y) ** 2))
        rmse = float(np.sqrt(mse))
        mae = float(np.mean(np.abs(preds - y)))
        ss_res = float(np.sum((y - preds) ** 2))
        ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        print(f'Loaded model {model_path}: RMSE={rmse:.4f}, MAE={mae:.4f}, R2={r2:.4f}')
    except Exception as e:
        print('Failed to load model:', e)


if __name__ == '__main__':
    test_global()

#///////////////////////////////////////////////////////////////////////////
"""Evaluate saved global linear model on generated global data and print regression metrics."""
import numpy as np
from model import LinearRegressionModel
import csv

INPUT_COLS = ['ResponseTime(ms)','Availability(%)','ReliabilityScore','Usability(%)','Credibility(%)','Certification','CostSatisfaction(%)','Prestige(1-5)','Security(%)']
LABEL_COL = 'TrustIndex'

def test_global(global_csv='data/global.csv', model_path='global_model.npz'):
    X = []; y = []
    with open(global_csv, newline='') as f:
        reader = csv.DictReader(f)
        for r in reader:
            X.append([float(r[c]) if c != 'Certification' else (1.0 if r[c].lower().startswith('y') else 0.0) for c in INPUT_COLS])
            y.append(float(r[LABEL_COL]))
    X = np.array(X); y = np.array(y)
    try:
        d = np.load(model_path)
        model = LinearRegressionModel(input_dim=X.shape[1])
        model.set_weights({'w': d['w'], 'b': float(d['b'])})
        preds = model.predict(X)
        mse = float(np.mean((preds - y)**2))
        rmse = float(np.sqrt(mse))
        mae = float(np.mean(np.abs(preds - y)))
        ss_res = float(np.sum((y - preds)**2))
        ss_tot = float(np.sum((y - float(np.mean(y)))**2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        print(f'Loaded model {model_path}: RMSE={rmse:.4f}, MAE={mae:.4f}, R2={r2:.4f}')
    except Exception as e:
        print('Failed to load model:', e)

if __name__ == '__main__':
    test_global()

'''
