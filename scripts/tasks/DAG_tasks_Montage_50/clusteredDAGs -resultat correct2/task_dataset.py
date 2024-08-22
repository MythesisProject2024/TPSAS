import numpy
import random
import pandas as pd

# Define task attributes (task_di, task_size)
'''task_attributes = {
    0: {'task_di': 0, 'task_size': 10000, 'frequencyNeed_claimed': 20000, 'memoryNeed_claimed': 100, 'priority':100},
    1: {'task_di': 1, 'task_size': 10000, 'frequencyNeed_claimed': 200, 'memoryNeed_claimed': 50, 'priority':100},
    2: {'task_di': 2, 'task_size': 5000, 'frequencyNeed_claimed': 80, 'memoryNeed_claimed': 10, 'priority':100},
    3: {'task_di': 3, 'task_size': 2000, 'frequencyNeed_claimed': 10, 'memoryNeed_claimed': 10, 'priority':100},
    4: {'task_di': 4, 'task_size': 1800, 'frequencyNeed_claimed': 150, 'memoryNeed_claimed': 10, 'priority':100},
    5: {'task_di': 5, 'task_size': 1000, 'frequencyNeed_claimed': 300, 'memoryNeed_claimed': 10, 'priority':100},
    6: {'task_di': 6, 'task_size': 1000, 'frequencyNeed_claimed': 50, 'memoryNeed_claimed': 1001, 'priority':100},
    7: {'task_di': 7, 'task_size': 500, 'frequencyNeed_claimed': 50, 'memoryNeed_claimed': 5, 'priority':100},
    8: {'task_di': 8, 'task_size': 500, 'frequencyNeed_claimed': 50, 'memoryNeed_claimed': 5, 'priority':100},
    9: {'task_di': 9, 'task_size': 500, 'frequencyNeed_claimed': 50, 'memoryNeed_claimed': 5, 'priority':100},
}'''

# Dictionnaire pour stocker les caractéristiques des tasks
task_dataset = []
nb_tasks = 1000
min_nb_of_intensive_task=100
min_nb_of_not_intensive_task=100

intensive_task_counter=0
not_int_task_counter=0

for _ in range(nb_tasks):
    # Sélection aléatoire de la famille de machine
    if intensive_task_counter < min_nb_of_intensive_task:
        task_size = numpy.random.randint(6000000000, 10000000000, dtype=numpy.int64)
        #task_size = numpy.random.randint(6000000000, 7000000000, dtype=numpy.int64)
        #memoryNeed_claimed=numpy.random.randint(3000, 8000)
        memoryNeed_claimed=numpy.random.randint(3000, 4000)
        intensive_task_counter+=1
    elif not_int_task_counter < min_nb_of_not_intensive_task:
        task_size = numpy.random.randint(1000000, 2000000000, dtype=numpy.int64)
        memoryNeed_claimed=numpy.random.randint(100, 1000)
        not_int_task_counter+=1
    else:
        #task_size = numpy.random.randint(2000000000, 10000000000, dtype=numpy.int64)
        task_size = numpy.random.randint(2000000000, 7000000000, dtype=numpy.int64)

        #memoryNeed_claimed=numpy.random.randint(1000, 8000)
        memoryNeed_claimed=numpy.random.randint(1000, 4000)

    cpu_instructions_per_second = 1000000000  # 1 GHz
    #frequencyNeed_claimed = task_size / cpu_instructions_per_second *1.1

    #frequency in nb inst/s
    frequencyNeed_claimed = task_size

    priority=numpy.random.randint(1, 100)

   # Ajout des caractéristiques de task au dataset

    task_dataset.append({
        'task_id':_,
        'task_size': task_size,
        'frequencyNeed_claimed': frequencyNeed_claimed,
        'memoryNeed_claimed': memoryNeed_claimed,
        'priority': priority
    })

# Affichage du dataset
for task in task_dataset:
    print(task)
#################### save the dictionnaries in CSV file################3333

# Convert the list of dictionaries (task_dataset) into a DataFrame
df = pd.DataFrame(task_dataset)

# Specify the path where you want to save the CSV file
csv_file_path = '.\\task_dataset.csv'

# Save the DataFrame to a CSV file
df.to_csv(csv_file_path, index=False)

print("task dataset saved to:", csv_file_path)