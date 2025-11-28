# generate.py
import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

NB_CLIENTS = 5
TEST_SIZE = 0.2
traing_dataset = 'data/synthetic_trust_dataset.csv'

def generate(in_file=traing_dataset, out_dir='data', seed=42):
    np.random.seed(seed)
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # Load dataset
    df = pd.read_csv(in_file)
    print(f"Loaded dataset with {len(df)} rows from {in_file}")

    # Sort by ServiceID
    df['ServiceNumber'] = df['ServiceID'].str.extract('(\d+)').astype(int)
    df = df.sort_values(by='ServiceNumber').drop(columns=['ServiceNumber'])

    # Split into global train/test
    df_train, df_test = train_test_split(df, test_size=TEST_SIZE, random_state=seed)

    df_train.to_csv(os.path.join(out_dir, 'global_train.csv'), index=False)
    df_test.to_csv(os.path.join(out_dir, 'global_test.csv'), index=False)
    print(f"Saved global_train.csv ({len(df_train)}) and global_test.csv ({len(df_test)})")

    # Partition train among clients by ServiceID
    service_ids = df_train['ServiceID'].unique()
    if len(service_ids) < NB_CLIENTS:
        raise ValueError(f"Not enough unique ServiceIDs ({len(service_ids)}) for {NB_CLIENTS} clients")

    for client_id, service_id in enumerate(service_ids[:NB_CLIENTS]):
        client_data = df_train[df_train['ServiceID'] == service_id]
        client_data.to_csv(os.path.join(out_dir, f'client_{client_id}.csv'), index=False)
        print(f"Client {client_id} dataset size: {len(client_data)} (ServiceID={service_id})")

if __name__ == '__main__':
    generate()



'''

#THIS WITH RNADOMISE SAMPLES  ////////////////////////
import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

NB_CLIENTS = 5
TEST_SIZE = 0.2
#traing_dataset = 'data/service_trust_dataset.csv'
traing_dataset = 'data/synthetic_trust_dataset.csv'

def generate(in_file=traing_dataset, out_dir='data', seed=42):
    np.random.seed(seed)
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # Load dataset
    df = pd.read_csv(in_file)
    print(f"Loaded dataset with {len(df)} rows from {in_file}")

    # Split into train/test
    df_train, df_test = train_test_split(df, test_size=TEST_SIZE, random_state=seed)

    # Save global train/test
    df_train.to_csv(os.path.join(out_dir, 'global_train.csv'), index=False)
    df_test.to_csv(os.path.join(out_dir, 'global_test.csv'), index=False)
    print(f"Saved global_train.csv ({len(df_train)}) and global_test.csv ({len(df_test)})")

    n_samples = len(df_train) // NB_CLIENTS
    start = 0
    for client_id in range(NB_CLIENTS):
        if client_id == NB_CLIENTS - 1:
            # Last client takes the remaining samples
            client_data = df_train.iloc[start:]
        else:
            client_data = df_train.iloc[start:start+n_samples]
        client_data.to_csv(os.path.join(out_dir, f'client_{client_id}.csv'), index=False)
        print(f"Client {client_id} dataset size: {len(client_data)}")
        start += n_samples

if __name__ == '__main__':
    generate()



#//////////////////////////////////////////////////////////////////////////
"""Load real dataset and partition it into local datasets for clients.
Reads data/service_trust_dataset.csv and produces:
- data/global.csv  (full dataset copy)
- data/client_{i}.csv (partition for each client)
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd

# Number of clients (global variable)
NB_CLIENTS = 5

def generate(in_file: str = 'data/service_trust_dataset.csv', out_dir: str = 'data', seed: int = 42) -> None:
    """
    Load real dataset, save global copy, and partition into local datasets for clients.

    Args:
        in_file: Path to the input dataset (CSV)
        out_dir: Directory to save datasets
        seed: Random seed for reproducibility
    """
    np.random.seed(seed)
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # Load the real dataset
    df = pd.read_csv(in_file)

    # Save as global dataset
    global_path = os.path.join(out_dir, 'global.csv')
    df.to_csv(global_path, index=False)

    # Shuffle dataset before partitioning
    df_shuffled = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    # Partition indices to avoid FutureWarning
    indices = np.array_split(df_shuffled.index, NB_CLIENTS)

    for client_id, idx in enumerate(indices):
        part = df_shuffled.loc[idx]
        part_path = os.path.join(out_dir, f'client_{client_id}.csv')
        part.to_csv(part_path, index=False)

    print(f"Loaded dataset with {len(df)} rows from {in_file}")
    print(f"Saved global dataset and {NB_CLIENTS} client partitions in {out_dir}")


if __name__ == '__main__':
    generate(in_file='data/service_trust_dataset.csv', out_dir='data', seed=42)


#/////////////////////////////////////////////////////////////////////////////////////////
"""Generate dataset by replicating sample rows with small perturbations.
Produces data/global.csv and one partition file per client (client_{i}.csv).
"""

import csv
import os
from pathlib import Path
import numpy as np
from typing import List

# Number of clients (global variable)
NB_CLIENTS = 5

SAMPLE = [
    ['S1', 202, 99.41, 0.86, 70.2, 97.4, 'Yes', 56.4, 4, 79.6, 0.838],
    ['S2', 535, 100, 0.83, 69.8, 87, 'Yes', 79.1, 1, 92.3, 0.793],
    ['S3', 960, 99.13, 0.89, 76.1, 75.5, 'No', 65, 5, 99.1, 0.868],
    ['S4', 370, 98.77, 0.82, 95.8, 74.2, 'Yes', 51.7, 2, 83.1, 0.791],
    ['S5', 206, 97.99, 0.9, 60, 87.3, 'Yes', 81.7, 5, 95.8, 0.874],
    ['S6', 171, 99.98, 0.98, 96.1, 92.9, 'Yes', 92.9, 5, 87, 0.959],
    ['S7', 800, 98.87, 0.9, 66.2, 93.4, 'Yes', 67.2, 4, 74.7, 0.839],
    ['S8', 120, 99.27, 0.92, 99.6, 78.3, 'Yes', 69.5, 2, 90.7, 0.85],
    ['S9', 714, 99.95, 0.83, 98.2, 76.1, 'No', 94.6, 5, 85.4, 0.907],
    ['S10', 221, 99.11, 0.81, 79.8, 70.5, 'Yes', 90.7, 2, 89.5, 0.806],
    ['S11', 566, 98.9, 0.82, 80.5, 90.4, 'Yes', 85.7, 3, 96.7, 0.861],
    ['S12', 314, 99.71, 0.92, 79.3, 72.3, 'No', 62.4, 3, 83.2, 0.816],
    ['S13', 430, 99.23, 0.82, 85.4, 82.6, 'Yes', 95.5, 3, 97, 0.867],
    ['S14', 558, 99.29, 0.88, 63.6, 88.1, 'Yes', 57.7, 5, 92.5, 0.852],
    ['S15', 187, 98.24, 0.91, 61.3, 98.1, 'No', 81.3, 3, 84.7, 0.844],
    ['S16', 472, 99.6, 0.88, 62.3, 79.4, 'No', 51.4, 4, 88.1, 0.807],
    ['S17', 199, 99.19, 0.83, 77.8, 77.5, 'Yes', 82.2, 4, 79.8, 0.839],
    ['S18', 971, 98.31, 0.85, 71, 89.6, 'No', 63.9, 5, 99.5, 0.871],
    ['S19', 763, 99.24, 0.99, 84.9, 88.6, 'Yes', 72.5, 2, 90, 0.859],
    ['S20', 230, 97.97, 0.94, 99.7, 80, 'No', 67.3, 1, 71.3, 0.812],
    ['S21', 761, 100, 0.9, 75.5, 97.8, 'Yes', 58.6, 2, 79.8, 0.818],
    ['S22', 408, 98.45, 0.8, 83, 95.3, 'No', 81.1, 2, 88.7, 0.834],
]

HEADER = [
    'ServiceID','ResponseTime(ms)','Availability(%)','ReliabilityScore','Usability(%)',
    'Credibility(%)','Certification','CostSatisfaction(%)','Prestige(1-5)','Security(%)','TrustIndex'
]

def generate(out_dir: str = 'data', records_per_service: int = 200, seed: int = 42) -> None:
    """
    Generate global dataset and partition into local datasets for clients.

    Args:
        out_dir: Directory to save datasets
        records_per_service: Number of rows per service
        seed: Random seed for reproducibility
    """
    np.random.seed(seed)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    rows: List[List] = []

    for s in SAMPLE:
        ServiceID, ResponseTime, Availability, ReliabilityScore, Usability, Credibility, Certification, CostSatisfaction, Prestige, Security, TrustIndex = s
        for _ in range(records_per_service):
            rt = max(1.0, np.random.normal(ResponseTime, ResponseTime * 0.05))
            avail = np.clip(np.random.normal(Availability, 0.5), 0.0, 100.0)
            rel = np.clip(np.random.normal(ReliabilityScore, 0.02), 0.0, 1.0)
            usab = np.clip(np.random.normal(Usability, 2.0), 0.0, 100.0)
            cred = np.clip(np.random.normal(Credibility, 2.0), 0.0, 100.0)
            cert = Certification
            costsat = np.clip(np.random.normal(CostSatisfaction, 3.0), 0.0, 100.0)
            pres = int(np.clip(round(np.random.normal(Prestige, 0.5)), 1, 5))
            sec = np.clip(np.random.normal(Security, 2.0), 0.0, 100.0)
            trust = np.clip(np.random.normal(TrustIndex, 0.02), 0.0, 1.0)
            rows.append([ServiceID, rt, avail, rel, usab, cred, cert, costsat, pres, sec, trust])

    # Save global dataset
    global_path = os.path.join(out_dir, 'global.csv')
    with open(global_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(HEADER)
        writer.writerows(rows)

    # Partition into local datasets for clients
    np.random.shuffle(rows)
    partitions = np.array_split(rows, NB_CLIENTS)

    for client_id, part in enumerate(partitions):
        part_path = os.path.join(out_dir, f'client_{client_id}.csv')
        with open(part_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(HEADER)
            writer.writerows(part)

    print(f'Generated global dataset with {len(rows)} rows and {NB_CLIENTS} client partitions in {out_dir}')


if __name__ == '__main__':
    generate(out_dir='data', records_per_service=200, seed=42)

#//////////////////////////////////////////////////////////////////////////////////////

"""Generate dataset by replicating sample rows with small perturbations.
Produces data/global.csv and one partition file per ServiceID (partition_sp_{i}.csv).
"""

import csv
import os
from pathlib import Path
import numpy as np
from typing import List

SAMPLE = [

        ['S1', 202, 99.41, 0.86, 70.2, 97.4, 'Yes', 56.4, 4, 79.6, 0.838],
        ['S2', 535, 100, 0.83, 69.8, 87, 'Yes', 79.1, 1, 92.3, 0.793],
        ['S3', 960, 99.13, 0.89, 76.1, 75.5, 'No', 65, 5, 99.1, 0.868],
        ['S4', 370, 98.77, 0.82, 95.8, 74.2, 'Yes', 51.7, 2, 83.1, 0.791],
        ['S5', 206, 97.99, 0.9, 60, 87.3, 'Yes', 81.7, 5, 95.8, 0.874],
        ['S6', 171, 99.98, 0.98, 96.1, 92.9, 'Yes', 92.9, 5, 87, 0.959],
        ['S7', 800, 98.87, 0.9, 66.2, 93.4, 'Yes', 67.2, 4, 74.7, 0.839],
        ['S8', 120, 99.27, 0.92, 99.6, 78.3, 'Yes', 69.5, 2, 90.7, 0.85],
        ['S9', 714, 99.95, 0.83, 98.2, 76.1, 'No', 94.6, 5, 85.4, 0.907],
        ['S10', 221, 99.11, 0.81, 79.8, 70.5, 'Yes', 90.7, 2, 89.5, 0.806],
        ['S11', 566, 98.9, 0.82, 80.5, 90.4, 'Yes', 85.7, 3, 96.7, 0.861],
        ['S12', 314, 99.71, 0.92, 79.3, 72.3, 'No', 62.4, 3, 83.2, 0.816],
        ['S13', 430, 99.23, 0.82, 85.4, 82.6, 'Yes', 95.5, 3, 97, 0.867],
        ['S14', 558, 99.29, 0.88, 63.6, 88.1, 'Yes', 57.7, 5, 92.5, 0.852],
        ['S15', 187, 98.24, 0.91, 61.3, 98.1, 'No', 81.3, 3, 84.7, 0.844],
        ['S16', 472, 99.6, 0.88, 62.3, 79.4, 'No', 51.4, 4, 88.1, 0.807],
        ['S17', 199, 99.19, 0.83, 77.8, 77.5, 'Yes', 82.2, 4, 79.8, 0.839],
        ['S18', 971, 98.31, 0.85, 71, 89.6, 'No', 63.9, 5, 99.5, 0.871],
        ['S19', 763, 99.24, 0.99, 84.9, 88.6, 'Yes', 72.5, 2, 90, 0.859],
        ['S20', 230, 97.97, 0.94, 99.7, 80, 'No', 67.3, 1, 71.3, 0.812],
        ['S21', 761, 100, 0.9, 75.5, 97.8, 'Yes', 58.6, 2, 79.8, 0.818],
        ['S22', 408, 98.45, 0.8, 83, 95.3, 'No', 81.1, 2, 88.7, 0.834],
]

HEADER = [
    'ServiceID','ResponseTime(ms)','Availability(%)','ReliabilityScore','Usability(%)',
    'Credibility(%)','Certification','CostSatisfaction(%)','Prestige(1-5)','Security(%)','TrustIndex'
]

def generate(out_dir: str = 'data', records_per_service: int = 200, seed: int = 42) -> None:
    """
    Generate global and per-client partition datasets.

    Args:
        out_dir: Directory to save datasets
        records_per_service: Number of rows per service
        seed: Random seed for reproducibility
    """
    np.random.seed(seed)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    rows: List[List] = []

    for i, s in enumerate(SAMPLE):
        ServiceID, ResponseTime, Availability, ReliabilityScore, Usability, Credibility, Certification, CostSatisfaction, Prestige, Security, TrustIndex = s
        for _ in range(records_per_service):
            rt = max(1.0, np.random.normal(ResponseTime, ResponseTime * 0.05))
            avail = np.clip(np.random.normal(Availability, 0.5), 0.0, 100.0)
            rel = np.clip(np.random.normal(ReliabilityScore, 0.02), 0.0, 1.0)
            usab = np.clip(np.random.normal(Usability, 2.0), 0.0, 100.0)
            cred = np.clip(np.random.normal(Credibility, 2.0), 0.0, 100.0)
            cert = Certification
            costsat = np.clip(np.random.normal(CostSatisfaction, 3.0), 0.0, 100.0)
            pres = int(np.clip(round(np.random.normal(Prestige, 0.5)), 1, 5))
            sec = np.clip(np.random.normal(Security, 2.0), 0.0, 100.0)
            trust = np.clip(np.random.normal(TrustIndex, 0.02), 0.0, 1.0)
            rows.append([i, ServiceID, rt, avail, rel, usab, cred, cert, costsat, pres, sec, trust])

    # Save global dataset
    global_path = os.path.join(out_dir, 'global.csv')
    with open(global_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['sp_id'] + HEADER)
        writer.writerows(rows)

    # Save per-client partitions
    n = len(SAMPLE)
    for sp in range(n):
        part_path = os.path.join(out_dir, f'partition_sp_{sp}.csv')
        with open(part_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['sp_id'] + HEADER)
            writer.writerows([r for r in rows if int(r[0]) == sp])

    print(f'Generated global dataset with {len(rows)} rows and {n} partitions in {out_dir}')


if __name__ == '__main__':
    generate(out_dir='data', records_per_service=200, seed=42)


#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

import csv, os, random
from pathlib import Path

SAMPLE = [

        ['S1', 202, 99.41, 0.86, 70.2, 97.4, 'Yes', 56.4, 4, 79.6, 0.838],
        ['S2', 535, 100, 0.83, 69.8, 87, 'Yes', 79.1, 1, 92.3, 0.793],
        ['S3', 960, 99.13, 0.89, 76.1, 75.5, 'No', 65, 5, 99.1, 0.868],
        ['S4', 370, 98.77, 0.82, 95.8, 74.2, 'Yes', 51.7, 2, 83.1, 0.791],
        ['S5', 206, 97.99, 0.9, 60, 87.3, 'Yes', 81.7, 5, 95.8, 0.874],
        ['S6', 171, 99.98, 0.98, 96.1, 92.9, 'Yes', 92.9, 5, 87, 0.959],
        ['S7', 800, 98.87, 0.9, 66.2, 93.4, 'Yes', 67.2, 4, 74.7, 0.839],
        ['S8', 120, 99.27, 0.92, 99.6, 78.3, 'Yes', 69.5, 2, 90.7, 0.85],
        ['S9', 714, 99.95, 0.83, 98.2, 76.1, 'No', 94.6, 5, 85.4, 0.907],
        ['S10', 221, 99.11, 0.81, 79.8, 70.5, 'Yes', 90.7, 2, 89.5, 0.806],
        ['S11', 566, 98.9, 0.82, 80.5, 90.4, 'Yes', 85.7, 3, 96.7, 0.861],
        ['S12', 314, 99.71, 0.92, 79.3, 72.3, 'No', 62.4, 3, 83.2, 0.816],
        ['S13', 430, 99.23, 0.82, 85.4, 82.6, 'Yes', 95.5, 3, 97, 0.867],
        ['S14', 558, 99.29, 0.88, 63.6, 88.1, 'Yes', 57.7, 5, 92.5, 0.852],
        ['S15', 187, 98.24, 0.91, 61.3, 98.1, 'No', 81.3, 3, 84.7, 0.844],
        ['S16', 472, 99.6, 0.88, 62.3, 79.4, 'No', 51.4, 4, 88.1, 0.807],
        ['S17', 199, 99.19, 0.83, 77.8, 77.5, 'Yes', 82.2, 4, 79.8, 0.839],
        ['S18', 971, 98.31, 0.85, 71, 89.6, 'No', 63.9, 5, 99.5, 0.871],
        ['S19', 763, 99.24, 0.99, 84.9, 88.6, 'Yes', 72.5, 2, 90, 0.859],
        ['S20', 230, 97.97, 0.94, 99.7, 80, 'No', 67.3, 1, 71.3, 0.812],
        ['S21', 761, 100, 0.9, 75.5, 97.8, 'Yes', 58.6, 2, 79.8, 0.818],
        ['S22', 408, 98.45, 0.8, 83, 95.3, 'No', 81.1, 2, 88.7, 0.834],
]

HEADER = ['ServiceID','ResponseTime(ms)','Availability(%)','ReliabilityScore','Usability(%)','Credibility(%)','Certification','CostSatisfaction(%)','Prestige(1-5)','Security(%)','TrustIndex']

def generate(out_dir='data', records_per_service=200, seed=42):
    random.seed(seed)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    rows = []
    for i, s in enumerate(SAMPLE):
        ServiceID, ResponseTime, Availability, ReliabilityScore, Usability, Credibility, Certification, CostSatisfaction, Prestige, Security, TrustIndex = s
        for _ in range(records_per_service):
            rt = max(1.0, random.gauss(ResponseTime, ResponseTime * 0.05))
            avail = min(100.0, max(0.0, random.gauss(Availability, 0.5)))
            rel = min(1.0, max(0.0, random.gauss(ReliabilityScore, 0.02)))
            usab = min(100.0, max(0.0, random.gauss(Usability, 2.0)))
            cred = min(100.0, max(0.0, random.gauss(Credibility, 2.0)))
            cert = Certification
            costsat = min(100.0, max(0.0, random.gauss(CostSatisfaction, 3.0)))
            pres = max(1, min(5, int(round(random.gauss(Prestige, 0.5)))))
            sec = min(100.0, max(0.0, random.gauss(Security, 2.0)))
            trust = max(0.0, min(1.0, random.gauss(TrustIndex, 0.02)))
            rows.append([i, ServiceID, rt, avail, rel, usab, cred, cert, costsat, pres, sec, trust])
    global_path = os.path.join(out_dir, 'global.csv')
    with open(global_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['sp_id'] + HEADER)
        for r in rows:
            writer.writerow(r)
    n = len(SAMPLE)
    for sp in range(n):
        part_path = os.path.join(out_dir, f'partition_sp_{sp}.csv')
        with open(part_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['sp_id'] + HEADER)
            for r in rows:
                if int(r[0]) == sp:
                    writer.writerow(r)
    print(f'Generated global dataset with {len(rows)} rows and {n} partitions in {out_dir}')

if __name__ == '__main__':
    generate(out_dir='data', records_per_service=200, seed=42)

'''
