import networkx as nx
import os
import random
import prepare_tasks
import pandas as pd


#from tasks import clusteringDAGs


#read from directory the dag files
#directory_path = "./dag_files_size_1000"
directory_path = "./labeled_dag_files_size_400"

DAG_List = []
DAG_List_clustred = []
#################### Function prepareDAGs
# Iterate through the files and print those with the ".dag" extension
def prepareDAGs(directory_path):
    # List all files in the directory
    files_in_directory = os.listdir(directory_path)
    DAG_List = []
    counter = 0

    for file_name in files_in_directory:
        dag = nx.DiGraph()
        dag.name = counter
        counter+=1
        print(file_name)
        # Read the DAG file and create a directed graph
        dag_file_path = directory_path+'/'+file_name
        with open(dag_file_path, 'r') as file:
            for line in file:
                if line.startswith("EDGE"):
                    data = line.strip().split()
                    source_task = int(data[1])  # Convert to integer
                    target_task = int(data[2])  # Convert to integer

                    dag.add_edge(source_task, target_task)

        # Identify entry and exit nodes
        dag.entry_nodes = [node for node in dag.nodes() if dag.in_degree(node) == 0]
        dag.exit_nodes = [node for node in dag.nodes() if dag.out_degree(node) == 0]

        ################### add dag structure
        # Create a dictionary to store the node neighbors with renamed sequential values
        dag_structure = {}
        # Iterate through the renamed nodes and extract their neighbors
        for node in dag.nodes():
            neighbors = list(dag.successors(node))
            dag_structure[node] = neighbors
        dag.dag_structure=dag_structure

        ################### add edges
        edges = dag.edges()
        edge_weights = {}

        def culculate_edge_weights(edge_weights):
            for edge in edges:
                #weight = 10
                #weight = 20
                #weight = 30
                #weight = 40
                #weight = 50
                #weight = 60
                #weight = 70
                #weight = 80
                #weight = 90
                weight = 100
                #weight = 500
                #weight = 1000
                #weight = 2000
                #weight = 3000
                #weight = 500
                #weight = 700
                #weight = 900
                #weight = random.randint(0, 10) * 100 + 50  # Calculez le poids de l'arête ici en fonction de vos besoins
                edge_weights[edge] = weight
            return edge_weights

        dag.edge_weights=culculate_edge_weights(edge_weights)

        ################### add task_attributes

        #dag50 need 50 tasks
        size = dag.number_of_nodes()
        dag.task_attributes = prepare_tasks.prepare_tasks(size)

        ################### add dags into DAG_List
        DAG_List.append(dag)

    return DAG_List
'''

#################### Function clusterSaveDAGS():save the dictionnaries in CSV file
#task_attributes=prepare_tasks.prepare_tasks(50)
def clusterSaveDAGs (DAG_list):
    nb_DAG = len(DAG_List)
    print(nb_DAG)

    for counter in range (nb_DAG):
        task_attributes=DAG_List[counter].task_attributes

        task_list = []

        for task in task_attributes:
            task_list.append({
                'task_id': task_attributes[task]["task_id"],
                'task_size': task_attributes [task]["task_size"],
                'frequencyNeed_claimed': task_attributes [task]["frequencyNeed_claimed"],
                'memoryNeed_claimed': task_attributes[task]["memoryNeed_claimed"],
                'priority': task_attributes[task]["priority"]
            })

        # Convert the list of dictionaries (task_dataset) into a DataFrame
        df = pd.DataFrame(task_list)

        # Specify the path where you want to save the CSV file
        csv_file_path1 = f'.\\tasks\\DAG_tasks_Montage_50\\tasks_Montage_50_{counter}.csv'

        # Save the DataFrame to a CSV file
        df.to_csv(csv_file_path1, index=False)

        print("tasks montage 50 saved to:", csv_file_path1)

        ######## save the dictionnaries in CSV file normalised for clustering
        task_list_normalised = []
        for task in task_attributes:
            task_list_normalised.append({
                'task_id': task_attributes[task]["task_id"],
                'task_size': task_attributes [task]["task_size"],
                'frequencyNeed_claimed': task_attributes [task]["frequencyNeed_claimed"]/1000000,
                'memoryNeed_claimed': task_attributes[task]["memoryNeed_claimed"],
                'priority': task_attributes[task]["priority"]*100
            })

        # Convert the list of dictionaries (task_dataset) into a DataFrame
        df = pd.DataFrame(task_list_normalised)

        # Specify the path where you want to save the CSV file
        #csv_file_path2 = '.\\tasks\\tasks_Montage_50_normalised.csv'
        csv_file_path2 = f'.\\tasks\\DAG_tasks_Montage_50\\tasks_Montage_50_{counter}_normalised.csv'

        csv_file_path3 = f'.\\tasks\\DAG_tasks_Montage_50\\clusteredDAGs\\tasks_Montage_50_{counter}_clustered.csv'

        # Save the DataFrame to a CSV file
        df.to_csv(csv_file_path2, index=False)

        print("tasks montage 50 normalised saved to:", csv_file_path2)

        clusteringDAGs.clusterDAG(csv_file_path1, csv_file_path2, csv_file_path3 )

    return 0
'''
#################### function clusterDAGList

def clusterDAGList(DAG_List, directory_path):
    # List all files in the directory
    files_in_directory = os.listdir(directory_path)
    #DAG_List = []

    # Iterate through the csv files
    counter=0

    for file_name in files_in_directory:
        path_file_name = directory_path+'/'+file_name
        df = pd.read_csv(path_file_name)
        #task_attributes = {}
        lignes = len(df)
        for task_id in range(lignes):
            row = df.iloc[task_id]
            # task attributes
            DAG_List[counter].task_attributes[task_id] = row.to_dict()  # Assign each row as a dictionary
            #DAG_List[counter].task_attributes = row.to_dict()
        counter+=1
    return DAG_List



############# call the functions: prepareDAGS and saveDAGS


DAG_List = prepareDAGs(directory_path)

#clusterSaveDAGs(DAG_List)

directory_path_clusteredDAGs = ".\\tasks\\DAG_tasks_Montage_400\\clusteredDAGs"
DAG_List_clustred = clusterDAGList(DAG_List, directory_path_clusteredDAGs)

############# print dags
for dag in DAG_List_clustred:
    print('===========================')
    print(dag)
    print('===========================')
    print(dag.task_attributes)

    # Print Entry nodes
    print('DAG: ', dag)
    print('nb of entry tasks: ', len(dag.entry_nodes))
    # Print Exit nodes
    print('nb of exit tasks: ', len(dag.exit_nodes))
    print("Entry nodes:", dag.entry_nodes)
    print('')
    print("Exit nodes:", dag.exit_nodes)
    print('------------------------------------------------------------------')
    # Print nodes, edges, and attributes
    #print("Nodes:", dag.nodes(data=True))
    print("Edges:", dag.edges())
    print("dag_structure:", dag.dag_structure)
    print("edge weights:", dag.edge_weights)



#f1.close()
