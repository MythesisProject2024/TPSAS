

import numpy as np
import pandas as pd

np.random.seed(42)

def generate_client_dataset(client_id, n_services=20, samples_per_service=100):
    n_rows = n_services * samples_per_service

    # Client-specific bias for features (heterogeneity)
    bias_response = np.random.randint(-20, 20)
    bias_availability = np.random.uniform(-0.5, 0.5)
    bias_security = np.random.uniform(-1, 1)

    # Service IDs
    ServiceID = np.repeat([f"S{i}" for i in range(1, n_services + 1)], samples_per_service)

    # Features with slight client-specific bias
    ResponseTime = np.clip(np.random.randint(100, 1000, size=n_rows) + bias_response, 100, 1000)
    Availability = np.clip(np.round(np.random.uniform(97, 100, size=n_rows) + bias_availability, 2), 97, 100)
    ReliabilityScore = np.round(np.random.uniform(0.8, 1.0, size=n_rows), 3)
    Usability = np.round(np.random.uniform(60, 100, size=n_rows), 2)
    Credibility = np.round(np.random.uniform(70, 100, size=n_rows), 2)
    Certification = np.random.choice([0, 1], size=n_rows, p=[0.3, 0.7])
    CostSatisfaction = np.round(np.random.uniform(50, 100, size=n_rows), 2)
    Prestige = np.random.randint(1, 6, size=n_rows)
    Security = np.clip(np.round(np.random.uniform(70, 100, size=n_rows) + bias_security, 2), 70, 100)

    # Trust Index formula
    #trust_raw = (
    #    0.25 * (100 - ResponseTime) / 900 +
    #    0.25 * Availability / 100 +
    #    0.20 * Security / 100 +
    #    0.10 * ReliabilityScore +
    #    0.05 * Usability / 100 +
    #    0.05 * Credibility / 100 +
    #    0.05 * Certification +
    #    0.03 * CostSatisfaction / 100 +
    #    0.02 * (Prestige / 5)
    #)
    # Trust Index formula
    trust_raw = (
        0.23 * (100 - ResponseTime)/900 +  # normalized ResponseTime (lower is better)
        0.17  * ReliabilityScore +
        0.17 * CostSatisfaction/100 +
        0.15  * Security/100 +             # normalized Security
        0.11 * Availability/100 +          # normalized Availability
        0.08 * Credibility/100 +
        0.05 * Usability/100 +
        0.03 * Certification +
        0.01 * (Prestige/5)
    )

    # Gaussian noise
    trust_raw += np.random.normal(0, 0.02, size=n_rows)

    # Normalize to [0,1]
    TrustIndex = (trust_raw - trust_raw.min()) / (trust_raw.max() - trust_raw.min())
    TrustIndex = np.round(TrustIndex, 4)

    # DataFrame
    df = pd.DataFrame({
        "ServiceID": ServiceID,
        "ResponseTime(ms)": ResponseTime,
        "Availability(%)": Availability,
        "ReliabilityScore": ReliabilityScore,
        "Usability(%)": Usability,
        "Credibility(%)": Credibility,
        "Certification": Certification,
        "CostSatisfaction(%)": CostSatisfaction,
        "Prestige(1-5)": Prestige,
        "Security(%)": Security,
        "TrustIndex": TrustIndex
    })

    df.to_csv(f"data/client_{client_id}.csv", index=False)
    print(f"✅ Client {client_id} dataset created with {n_services} services × {samples_per_service} samples = {n_rows} rows.")

# Different dataset sizes for each client
client_configs = [
    (5, 80),     # very small
    (5, 200),   # small dataset
    (10, 500),  # medium
    (15, 800),  # large
    #(20, 1000), # large
    (25, 1500)  # very large
]

# Generate datasets
for i, (n_services, samples) in enumerate(client_configs):
    generate_client_dataset(i, n_services, samples)



'''
import numpy as np
import pandas as pd

# Base random seed for reproducibility
np.random.seed(42)

def generate_client_dataset(client_id, n_services=20, samples_per_service=100):
    """Generate a synthetic client dataset with heterogeneity and bias."""
    np.random.seed(42 + client_id)
    n_rows = n_services * samples_per_service

    # Client-specific heterogeneity
    bias_response = np.random.randint(-20, 20)
    bias_availability = np.random.uniform(-0.5, 0.5)
    bias_security = np.random.uniform(-1, 1)

    # Generate features
    ServiceID = np.repeat([f"S{i}" for i in range(1, n_services + 1)], samples_per_service)
    ResponseTime = np.clip(np.random.randint(100, 1000, size=n_rows) + bias_response, 100, 1000)
    Availability = np.clip(np.round(np.random.uniform(97, 100, size=n_rows) + bias_availability, 2), 97, 100)
    ReliabilityScore = np.round(np.random.uniform(0.8, 1.0, size=n_rows), 3)
    Usability = np.round(np.random.uniform(60, 100, size=n_rows), 2)
    Credibility = np.round(np.random.uniform(70, 100, size=n_rows), 2)
    Certification = np.random.choice([0, 1], size=n_rows, p=[0.3, 0.7])
    CostSatisfaction = np.round(np.random.uniform(50, 100, size=n_rows), 2)
    Prestige = np.random.randint(1, 6, size=n_rows)
    Security = np.clip(np.round(np.random.uniform(70, 100, size=n_rows) + bias_security, 2), 70, 100)

    # Trust Index
    trust_raw = (
        0.25 * (100 - ResponseTime) / 900 +
        0.25 * Availability / 100 +
        0.20 * Security / 100 +
        0.10 * ReliabilityScore +
        0.05 * Usability / 100 +
        0.05 * Credibility / 100 +
        0.05 * Certification +
        0.03 * CostSatisfaction / 100 +
        0.02 * (Prestige / 5)
    )

    trust_raw += np.random.normal(0, 0.02, size=n_rows)
    TrustIndex = (trust_raw - trust_raw.min()) / (trust_raw.max() - trust_raw.min() + 1e-9)
    TrustIndex = np.round(TrustIndex, 4)

    # Create DataFrame
    df = pd.DataFrame({
        "ServiceID": ServiceID,
        "ResponseTime(ms)": ResponseTime,
        "Availability(%)": Availability,
        "ReliabilityScore": ReliabilityScore,
        "Usability(%)": Usability,
        "Credibility(%)": Credibility,
        "Certification": Certification,
        "CostSatisfaction(%)": CostSatisfaction,
        "Prestige(1-5)": Prestige,
        "Security(%)": Security,
        "TrustIndex": TrustIndex
    })

    df.to_csv(f"data/client_{client_id}.csv", index=False)
    print(f"✅ Client {client_id} dataset created with {n_rows} samples ({n_services} services × {samples_per_service} each).")

# Heterogeneous dataset configuration
client_configs = [
    (5, 80),     # Client 1 → 400
    (8, 100),    # Client 2 → 800
    (10, 160),   # Client 3 → 1600
    (20, 250),   # Client 4 → 5000
    (25, 400)    # Client 5 → 10000
]

# Generate all datasets
for i, (n_services, samples) in enumerate(client_configs):
    generate_client_dataset(i, n_services, samples)

print("\n✅ All heterogeneous client datasets generated successfully.")





'''
