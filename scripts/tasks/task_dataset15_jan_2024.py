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
min_nb_of_cpu_intensive_task=100
min_nb_of_memory_intensive_task=100
min_nb_of_not_intensive_task=500
#min_nb_of_not_intensive_task=100

cpu_intensive_task_counter=0
memory_intensive_task_counter=0
not_int_task_counter=0


def fun_addInDataSet():
    cpu_instructions_per_second = 1000000000  # 1 GHz
    #frequencyNeed_claimed = task_size / cpu_instructions_per_second *1.1

    #frequency in nb inst/s
    task_size = frequencyNeed_claimed

    #shortest of tasks are prioritized
    shortest = ((10000000000 - task_size)/10000000000 *100)
    importance = numpy.random.randint(1, 50)

     #calculation of timeDeadline
    deltaT = numpy.random.randint(1, 1000)
    timeDeadLinePrefered = task_size/frequencyNeed_claimed * 1000
    timeDeadlineFinal = timeDeadLinePrefered + deltaT

    #calculation of priority
    #priority=numpy.random.randint(1, 100)
    # Calculate the percentage of priority based on the relationship between timeDeadLinePrefered and timeDeadlineFinal
    #priority_percentage = 100 * (timeDeadlineFinal - timeDeadLinePrefered) / (timeDeadlineFinal) + shortest
    priority_percentage = shortest
    priority = round(priority_percentage )

    # Ensure priority_percentage is within the range [0, 100]
    #priority_percentage = max(0, min(100, priority_percentage))
    #priority = round((timeDeadLinePrefered/timeDeadlineFinal)*100)

   # Ajout des caractéristiques de task au dataset

    task_dataset.append({
        'task_id':task_id,
        'task_size': task_size,
        'frequencyNeed_claimed': frequencyNeed_claimed,
        'memoryNeed_claimed': memoryNeed_claimed,
        'priority': priority,
        'timeDeadLinePrefered': timeDeadLinePrefered,
        'timeDeadlineFinal': timeDeadlineFinal

    })

task_id=0
for _ in range(nb_tasks):

    # Sélection aléatoire de la famille de machine
    if cpu_intensive_task_counter < min_nb_of_cpu_intensive_task:
        frequencyNeed_claimed = numpy.random.randint(7000000000, 10000000000, dtype=numpy.int64)
        #frequencyNeed_claimed = numpy.random.randint(6000000000, 10000000000, dtype=numpy.int64)
        #frequencyNeed_claimed = numpy.random.randint(6000000000, 7000000000, dtype=numpy.int64)
        memoryNeed_claimed=numpy.random.randint(500, 3000)
        #memoryNeed_claimed=numpy.random.randint(3000, 8000)
        #memoryNeed_claimed=numpy.random.randint(3000, 4000)

        fun_addInDataSet()
        cpu_intensive_task_counter+=1
        task_id+=1

    if memory_intensive_task_counter < min_nb_of_memory_intensive_task:
        frequencyNeed_claimed = numpy.random.randint(1000000000, 5000000000, dtype=numpy.int64)
        #frequencyNeed_claimed = numpy.random.randint(6000000000, 10000000000, dtype=numpy.int64)
        #frequencyNeed_claimed = numpy.random.randint(6000000000, 7000000000, dtype=numpy.int64)
        memoryNeed_claimed=numpy.random.randint(3000, 6000)
        #memoryNeed_claimed=numpy.random.randint(3000, 8000)
        #memoryNeed_claimed=numpy.random.randint(3000, 4000)

        fun_addInDataSet()
        memory_intensive_task_counter+=1
        task_id+=1

    if not_int_task_counter <= min_nb_of_not_intensive_task:
        frequencyNeed_claimed = numpy.random.randint(1000000, 2000000000, dtype=numpy.int64)
        memoryNeed_claimed=numpy.random.randint(100, 1000)

        fun_addInDataSet()
        not_int_task_counter+=1
        task_id+=1

    elif task_id < 1000:
        #frequencyNeed_claimed = numpy.random.randint(2000000000, 10000000000, dtype=numpy.int64)
        frequencyNeed_claimed = numpy.random.randint(1000000000, 10000000000, dtype=numpy.int64)
        #memoryNeed_claimed=numpy.random.randint(1000, 8000)
        memoryNeed_claimed=numpy.random.randint(100, 3000)
        fun_addInDataSet()
        task_id+=1



# Affichage du dataset
for task in task_dataset:
    print(task)
#################### save the dictionnaries in CSV file################3333

# Convert the list of dictionaries (task_dataset) into a DataFrame
df = pd.DataFrame(task_dataset)

# Specify the path where you want to save the CSV file
csv_file_path = '.\\task_dataset2.csv'

# Save the DataFrame to a CSV file
df.to_csv(csv_file_path, index=False)

print("task dataset saved to:", csv_file_path)