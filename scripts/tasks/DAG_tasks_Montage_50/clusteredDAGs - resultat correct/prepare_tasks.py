import pandas as pd
import random


'''# Read the CSV file
task_dataset = pd.read_csv(".\\tasks\\task_dataset.csv")
nb_task = 50
# Extract the first 50 rows from the DataFrame
task_rows = task_dataset.head(nb_task)
'''
# Load the task dataset
task_dataset = pd.read_csv(".\\tasks\\task_dataset.csv")

# Specify the number of rows to extract
#nb_task = 50

def prepare_tasks(nb_task):

    # Randomly sample 50 rows from the DataFrame
    task_rows = task_dataset.sample(n=nb_task, random_state=random.seed()) # Use a random seed for reproducibility

    # Now, task_rows contains 50 random rows from the task_dataset DataFrame
    #print (task_rows)

    # Initialize an empty dictionary to store the task attributes
    task_attributes = {}

    def initDAG(nb_task):
        for task_id in range(nb_task):
            row = task_rows.iloc[task_id]
            # task attributes
            task_attributes[task_id] = row.to_dict()  # Assign each row as a dictionary
        return task_attributes

    # Call the function to initialize the task attributes dictionary
    task_attributes = initDAG(nb_task)

    return task_attributes

'''# Print the task structure
task_attributes = prepare_tasks(50)
print('*****************TASK ATTRIBUTES structure******************')
print(task_attributes)'''










'''# Read the CSV file
task_dataset = pd.read_csv(".\\task_dataset.csv")

#task_colnames = ['TaskID', 'RAMNeed_Claimed', 'CPUNeed_Claimed', 'PriorityNo', 'SuccessorsImediate']

# Extract the first 50 rows (indices 0 to 20) from the DataFrame based on colnames
task_rows = task_dataset.head(50)

nb_task = 50

# Initialize an empty dictionary to store the DAG structure
#------------------------------------------
dag_structure = {}
task_attributes = {}

def initDAG(nb_task):
    for task_id in range(nb_task):
        row = task_rows.iloc[task_id]
        # task attributes
        task_attributes.append(row)
    return task_attributes

# Number of tasks (50 tasks in this case)

# Call the function to initialize the DAG structure and task dictionneries
task_attributes = initDAG(nb_task)

# Print the task  structure
print('*****************TASK ATTRIBUTES structure******************')
print(task_attributes)
'''