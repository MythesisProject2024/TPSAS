import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from client import FLClient

# -----------------------------
# Configuration
# -----------------------------
INPUT_COLS = ['ResponseTime(ms)', 'Availability(%)', 'ReliabilityScore',
              'Usability(%)', 'Credibility(%)', 'Certification',
              'CostSatisfaction(%)', 'Prestige(1-5)', 'Security(%)']
LABEL_COL = 'TrustIndex'

NUM_CLIENTS = 5
FED_ROUNDS = 10
LR = 0.01
EPOCHS = 3

INITIAL_TRUST = 0.5
MIN_HONEST_TRUST = 0.4
MAX_TRUST_INCREMENT = 0.5
WINDOW_ROUNDS = 5  # rounds to confirm persistent outlier
outlier = 0

DATA_PATHS = [f'data/client_{i}.csv' for i in range(NUM_CLIENTS)]

# Thresholding / combination weights (tuneable)
ALPHA = 0.8  # weight for normalized global-eval RMSE in combined score
BETA = 0.2   # weight for weight divergence
COMBINED_THRESHOLD = 1.5  # threshold on combined normalized score to mark outlier (persistent)
EPS = 1e-12

# -----------------------------
# Initialize clients
# -----------------------------
clients = []
for i, path in enumerate(DATA_PATHS):
    c = FLClient(i, path, INPUT_COLS, LABEL_COL)
    c.trust_score = INITIAL_TRUST
    clients.append(c)

# Fit a global scaler using all client data
all_X = np.vstack([c.X for c in clients])
global_scaler = StandardScaler().fit(all_X)

# Apply same scaler to all clients and store scaled X
for c in clients:
    c.X_scaled = global_scaler.transform(c.X)

# Initialize global_weights as average of initial clients' weights
initial_weights = [c.get_weights() for c in clients]
w0 = np.mean([iw['w'] for iw in initial_weights], axis=0)
b0 = float(np.mean([iw['b'] for iw in initial_weights]))
global_weights = {'w': np.clip(w0, -1e2, 1e2), 'b': np.clip(b0, -1e2, 1e2)}

# Bookkeeping
rmse_history = {c.client_id: [] for c in clients}
confirmed_outliers = set()
score_history = {c.client_id: [] for c in clients}

# Round counter for exponential scaling
rounds_since_last_outlier = 0


# -----------------------------
# Helper functions
# -----------------------------
def eval_weights_on_client(c: FLClient, weights: dict) -> float:
    Xs = c.X_scaled
    preds = Xs.dot(weights['w']) + weights['b']
    return np.sqrt(mean_squared_error(c.y, preds))

def weight_divergence(local_w: np.ndarray, global_w: np.ndarray) -> float:
    num = np.linalg.norm(local_w - global_w)
    denom = np.linalg.norm(global_w) + EPS
    return num / denom


# -----------------------------
# Federated training loop
# -----------------------------
# -----------------------------
# Federated training loop (UPDATED)
# -----------------------------
for round_id in range(1, FED_ROUNDS + 1):
    print(f"\n=== Scenario 3/4 - Exponential Trust - Federated Round {round_id}/{FED_ROUNDS} ===")

    new_outlier_detected_this_round = False

    # ------------------ 1) Local training ------------------
    for c in clients:
        if c.client_id in confirmed_outliers:
            continue
        try:
            original_X = c.X.copy()
            c.X = c.X_scaled
            c.local_train(epochs=1, lr=LR)
            c.X = original_X
        except Exception as e:
            print(f"⚠️ Training failed for client {c.client_id}: {e}")

    # ------------------ 2) Collect metrics ------------------
    local_rmses = {}
    global_eval_rmses = {}
    divergences = {}
    local_weights_map = {}

    for c in clients:
        lw = c.get_weights()
        local_weights_map[c.client_id] = lw

        # Local RMSE
        try:
            preds_local = c.model.predict(c.X_scaled)
            local_rmses[c.client_id] = float(np.sqrt(mean_squared_error(c.y, preds_local)))
        except Exception:
            local_rmses[c.client_id] = float(np.inf)

        # Global-eval RMSE
        try:
            global_eval_rmses[c.client_id] = float(eval_weights_on_client(c, global_weights))
        except Exception:
            global_eval_rmses[c.client_id] = float(np.inf)

        # Weight divergence
        try:
            divergences[c.client_id] = float(weight_divergence(lw['w'], global_weights['w']))
        except Exception:
            divergences[c.client_id] = float(np.inf)

    # ------------------ 3) Build thresholds ------------------
    candidate_ids = [c.client_id for c in clients if c.client_id not in confirmed_outliers]
    global_eval_list = [global_eval_rmses[i] for i in candidate_ids]
    div_list = [divergences[i] for i in candidate_ids]

    median_global_eval = float(np.median(global_eval_list)) if global_eval_list else 1.0
    median_div = float(np.median(div_list)) if div_list else 1.0

    # ------------------ 4) Detect persistent outliers ------------------
    for c in clients:
        cid = c.client_id
        if cid in confirmed_outliers:
            continue

        gres = global_eval_rmses.get(cid, np.inf)
        div = divergences.get(cid, np.inf)

        norm_rmse = gres / (median_global_eval + EPS)
        norm_div = div / (median_div + EPS)

        combined = ALPHA * norm_rmse + BETA * norm_div

        # Record score history
        score_history[cid].append(combined)
        if len(score_history[cid]) > WINDOW_ROUNDS:
            score_history[cid].pop(0)

        rmse_history[cid].append(local_rmses.get(cid, np.inf))
        if len(rmse_history[cid]) > WINDOW_ROUNDS:
            rmse_history[cid].pop(0)

        # Persistent outlier check
        if len(score_history[cid]) == WINDOW_ROUNDS:
            avg_score = float(np.mean(score_history[cid]))
            if avg_score > COMBINED_THRESHOLD:
                confirmed_outliers.add(cid)
                new_outlier_detected_this_round = True
                print(f"❌ Client {cid} confirmed as persistent outlier (avg combined score={avg_score:.3f}).")

    # ------------------ 5) Exponential dynamic trust increment ------------------
    if new_outlier_detected_this_round:
        rounds_since_last_outlier = 0
    else:
        rounds_since_last_outlier += 1

    dynamic_increment = MAX_TRUST_INCREMENT * (1 + len(confirmed_outliers)) * (1.3 ** rounds_since_last_outlier)

    # ------------------ 5.1) Apply mild & strong penalties ------------------
    max_div = max(divergences.values()) if divergences else 1.0
    for c in clients:
        cid = c.client_id
        div = divergences.get(cid, 0.0)
        combined = np.mean(score_history.get(cid, [0.0]))

        # Mild penalty for suspected abnormal clients
        if cid not in confirmed_outliers and (div > 1.3 * median_div or combined > 1.3 * np.median(list(score_history.values()))):
            penalty = 0.01 * (div / (max_div + EPS))
            c.trust_score = max(c.trust_score - penalty, MIN_HONEST_TRUST)

        # Strong penalty for confirmed outliers
        if cid in confirmed_outliers:
            c.trust_score = max(c.trust_score * 0.8, 0.0)

    # ------------------ 6) Update trust for non-outliers ------------------
    for c in clients:
        if c.client_id in confirmed_outliers:
            continue
        gres = global_eval_rmses.get(c.client_id, np.inf)

        if gres == np.inf or median_global_eval == 0:
            reward = 0.0
        else:
            reward = dynamic_increment * max(0.0, (median_global_eval - gres) / (median_global_eval + EPS))
            stability_bonus = 0.02 * max(0.0, (median_div - divergences[c.client_id]) / (median_div + EPS))
            reward += stability_bonus

        c.trust_score = min(c.trust_score + reward, 1.0)
        c.trust_score = max(c.trust_score, MIN_HONEST_TRUST)

    # ------------------ 7) Aggregate trusted clients ------------------
    trusted_clients = [c for c in clients if c.client_id not in confirmed_outliers]
    if not trusted_clients:
        print("⚠️ No trusted clients remaining! Stopping.")
        break

    w_sum = np.sum([c.get_weights()['w'] * c.trust_score for c in trusted_clients], axis=0)
    b_sum = np.sum([c.get_weights()['b'] * c.trust_score for c in trusted_clients])
    total_trust = sum(c.trust_score for c in trusted_clients)

    new_w = np.clip(w_sum / (total_trust + EPS), -1e2, 1e2)
    new_b = np.clip(b_sum / (total_trust + EPS), -1e2, 1e2)
    global_weights = {'w': new_w, 'b': new_b}

    for c in clients:
        c.set_weights(global_weights)

    # ------------------ 8) Reporting ------------------
    avg_combined_scores = {cid: round(np.mean(scores), 3) for cid, scores in score_history.items() if scores}
    print("⚖️ Trust scores:", {c.client_id: round(c.trust_score, 3) for c in clients})
    print("📊 Local RMSEs:", {k: round(v, 4) for k, v in local_rmses.items()})
    print("📊 Global-eval RMSEs:", {k: round(v, 4) for k, v in global_eval_rmses.items()})
    print("📊 Weight divergences:", {k: round(v, 4) for k, v in divergences.items()})
    print("📊 Avg combined scores:", avg_combined_scores)
    print("🚫 Confirmed outliers:", sorted(list(confirmed_outliers)))
    print("dynamic_increment", round(dynamic_increment, 4))
    print("rounds_since_last_outlier", rounds_since_last_outlier)

    # ------------------ 9) Global aggregated performance ------------------
    global_y_true = np.concatenate([c.y for c in trusted_clients])
    global_y_pred = np.concatenate([np.clip(c.model.predict(c.X_scaled), -1e2, 1e2) for c in trusted_clients])
    global_rmse = np.sqrt(mean_squared_error(global_y_true, global_y_pred))
    global_r2 = r2_score(global_y_true, global_y_pred)
    print(f"🌐 Global RMSE (trusted): {global_rmse:.4f}, Global R² (trusted): {global_r2:.4f}")


# -----------------------------
# Save final global model
# -----------------------------
np.save('global_model_scenario3_exponential.npy', global_weights)
print("\n✅ Scenario 3/4 - Exponential Outliers complete. Saved global_model_scenario3_exponential.npy")



