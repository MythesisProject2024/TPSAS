import random
import pandas as pd

# Dictionnaire des types de VM avec les correspondances entre vCPU() et mémoire (GO)
# 1 VCPU=1 logique core (between 1 and 2 GHZ)
# memory (GO)
'''vm_types = {
    'General-purpose': {
        'e2-small': {'vCPU': 2, 'memory': 2},
        'e2-medium': {'vCPU': 2, 'memory': 4},
        'e2-standard': {'vCPU': 2, 'memory': 8},
        'n2-standard': {'vCPU': 2, 'memory': 8}
    },
    'Compute-optimized': {
        'n2-highcpu': {'vCPU': 4, 'memory': 16},
        'c2-standard': {'vCPU': 4, 'memory': 16},
        'c2-highcpu': {'vCPU': 8, 'memory': 32}
    },
    'Memory-optimized': {
        'm2-ultramem': {'vCPU': 16, 'memory': 64},
        'm2-megamem': {'vCPU': 32, 'memory': 128}
    }
}'''

cloud_vm_types = {
    'General-purpose': {
        'small': {'vCPU': 2, 'memory': 2, 'storage':257, 'bw':5},
        #'medium': {'vCPU': 2, 'memory': 4, 'storage':257, 'bw':5},
        'standard': {'vCPU': 2, 'memory': 8, 'storage':257, 'bw':5}
    },
    'Compute-optimized': {
        'standard': {'vCPU': 4, 'memory': 16, 'storage':257, 'bw':10},
        'highcpu': {'vCPU': 8, 'memory': 32, 'storage':257, 'bw':16}
    },
    'Memory-optimized': {
        'standard': {'vCPU': 16, 'memory': 64, 'storage':257, 'bw':16},
        'megamem': {'vCPU': 32, 'memory': 128, 'storage':257, 'bw':32}
    }
}
edge_vm_types = {
    'General-purpose': {
        'small': {'vCPU': 1, 'memory': 1, 'storage':32, 'bw':0.1},
        #'medium': {'vCPU': 1, 'memory': 2, 'storage':64, 'bw':0.1},
        'standard': {'vCPU': 1, 'memory': 2, 'storage':64, 'bw':0.1}
    },
    'Compute-optimized': {
        'standard': {'vCPU': 2, 'memory': 4, 'storage':257, 'bw':10},
        'highcpu': {'vCPU': 4, 'memory': 4, 'storage':128, 'bw':0.1}
    },
    'Memory-optimized': {
        'standard': {'vCPU': 4, 'memory': 8, 'storage':256, 'bw':0.1},
        'megamem': {'vCPU': 4, 'memory': 16, 'storage':257, 'bw':32}
    }
}
# Liste des familles de machine disponibles
machine_families = ['General-purpose', 'Compute-optimized', 'Memory-optimized']

# Dictionnaire pour stocker les caractéristiques des VM
vm_dataset = []

# Nombre de VM à générer
#edge_num_vms = 50
#cloud_num_vms = 10
num_vms = 1000
nb_nodes = 50 # total nodes edge and cloud

# 1 vCPU=1GH=1000000000 inst/second
#Unit_vCPU=1000000000
Unit_vCPU=4000000000


for _ in range(num_vms):
    # Sélection aléatoire de la famille de machine
    network=random.choice(["edge", "cloud"])
    machine_family = random.choice(machine_families)

    # Sélection aléatoire du type de VM correspondant à la famille
    if network=="cloud":
        vm_types=cloud_vm_types
    else:
        vm_types = edge_vm_types

    vm_type = random.choice(list(vm_types[machine_family].keys()))

    # Récupération des caractéristiques (vCPU et memory) du type de VM
    vcpu = vm_types[machine_family][vm_type]['vCPU']
    frequency = vcpu*Unit_vCPU

    m = vm_types[machine_family][vm_type]['memory']
    memory= m*1000

    storage = vm_types[machine_family][vm_type]['storage']
    bwGb = vm_types[machine_family][vm_type]['bw']
    bw=bwGb*1000


    # number_of_physical_nodes=10 in each network edge and cloud
    node = random.randint(1, nb_nodes)


    # Génération aléatoire de la capacité de stockage (entre 10 et 1000 Go)
    #storage = random.randint(10, 1000)
    # Génération aléatoire the network


    # Ajout des caractéristiques de la VM au dataset

    vm_dataset.append({
        'vm_id':_,
        'family': machine_family,
        'type': vm_type,
        'vCPU': vcpu,
        'frequency': frequency,
        'memory': memory,
        'storage': storage,
        'bw': bw,
        'node': node,
        'network':network
    })


# Affichage du dataset
for vm in vm_dataset:
    print(vm)


#################### save the dictionnaries in CSV file################3333

# Convert the list of dictionaries (vm_dataset) into a DataFrame
df = pd.DataFrame(vm_dataset)

# Specify the path where you want to save the CSV file
csv_file_path = '.\\vm_dataset.csv'

# Save the DataFrame to a CSV file
df.to_csv(csv_file_path, index=False)

print("VM dataset saved to:", csv_file_path)







