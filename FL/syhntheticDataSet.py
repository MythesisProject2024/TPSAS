import numpy as np
import pandas as pd

np.random.seed(42)

# Parameters
n_services = 20
samples_per_service = 1000
n_rows = n_services * samples_per_service

# Generate ServiceID
ServiceID = np.repeat([f"S{i}" for i in range(1, n_services + 1)], samples_per_service)

# Generate features with realistic ranges
ResponseTime = np.random.randint(100, 1000, size=n_rows)  # lower is better
Availability = np.round(np.random.uniform(97, 100, size=n_rows), 2)  # higher is better
ReliabilityScore = np.round(np.random.uniform(0.8, 1.0, size=n_rows), 3)
Usability = np.round(np.random.uniform(60, 100, size=n_rows), 2)
Credibility = np.round(np.random.uniform(70, 100, size=n_rows), 2)
Certification = np.random.choice([0, 1], size=n_rows, p=[0.3, 0.7])
CostSatisfaction = np.round(np.random.uniform(50, 100, size=n_rows), 2)
Prestige = np.random.randint(1, 6, size=n_rows)
Security = np.round(np.random.uniform(70, 100, size=n_rows), 2)

# Compute TrustIndex with high weight for critical features
# Critical: ResponseTime (low), Availability (high), Security (high)
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

# Add small Gaussian noise
trust_raw += np.random.normal(0, 0.02, size=n_rows)

# Normalize to [0,1]
TrustIndex = (trust_raw - trust_raw.min()) / (trust_raw.max() - trust_raw.min())
TrustIndex = np.round(TrustIndex, 4)

# Build DataFrame
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

# Save to CSV
df.to_csv("synthetic_trust_dataset.csv", index=False)

print("Synthetic dataset created with 20 services (1000 samples each) and saved as synthetic_trust_dataset.csv")
print(df.head())
print(df['ServiceID'].value_counts())





'''

import numpy as np
import pandas as pd

# Number of samples
N = 20000

# Random seed
np.random.seed(42)

# Generate features with realistic ranges
X = pd.DataFrame({
    'ResponseTime': np.random.uniform(100, 500, N),      # ms
    'Availability': np.random.uniform(90, 100, N),       # %
    'Reliability': np.random.uniform(0.8, 1.0, N),       # 0-1
    'Usability': np.random.uniform(50, 100, N),          # %
    'Credibility': np.random.uniform(0, 1, N),           # 0-1
    'Certification': np.random.randint(0, 2, N),         # 0 or 1
    'CostSatisfaction': np.random.uniform(0, 100, N),    # %
    'Prestige': np.random.randint(1, 6, N),              # 1-5
    'Security': np.random.uniform(50, 100, N)            # %
})

# True linear weights for TrustIndex
true_weights = np.array([-0.005, 0.05, 1.0, 0.02, 0.3, 0.2, 0.01, 0.1, 0.05])
bias = 0.5

# Compute TrustIndex with Gaussian noise
noise = np.random.normal(0, 0.05, N)
X['TrustIndex'] = X.values.dot(true_weights) + bias + noise

# Save dataset
X.to_csv('synthetic_service_trust_dataset.csv', index=False)
print("Synthetic dataset created with shape:", X.shape)


'''