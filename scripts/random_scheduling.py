import numpy as np
import numpy
import math
from mealpy.swarm_based.GSA import GSA


####################################################3

import networkx as nx
import pandas as pd
import os

import random
import copy
import numpy
import math
import time
import move
import csv

#-------------------to save output messages
import sys
f=open('output1.txt', 'w')
# We redirect the 'sys.stdout' command towards the descriptor file
sys.stdout = f
#-------------------End

#dag files size 50 to 700 clusterd
import readClusteredDAGfile500
DAG_List_clustred = readClusteredDAGfile500.DAG_List_clustred
dag = DAG_List_clustred[19]

csv_file_path_of_vms = '.\\VM\\cloud200_edge100_vms_SameConfiguration.csv'
vm_dataset = pd.read_csv(csv_file_path_of_vms)
# Convert the DataFrame to a dictionary
vm_attributes = vm_dataset.to_dict(orient='records')

#copy vm_attributes to save initial values because in the functions the VM capacities reduce
vm_attributes_copy = copy.deepcopy(vm_attributes)

#########################


####################### define other functons

# Function to calculate the memory sum for an agent's position
def calculate_memory_sum(x, vm_id):
    memory_sum = 0
    frequency_sum = 0
    #dim=10
    for task_id in range(0,len(x),1):
        if x[task_id]==vm_id:
            memory_sum += task_attributes[task_id]['memoryNeed_claimed']
            frequency_sum += task_attributes[task_id]['frequencyNeed_claimed']
    return memory_sum, frequency_sum

# Function display results best solution nd best fitness
def fun_save_metrics_result(param0 = None, param1 = None, param2 = None, param3 = None, param4 = None, param5 = None ):
    # Your code to calculate metrics here...

    # Organize metrics into a dictionary
    metrics = {
        "Algorithm": param0,
        "Run":param1,
        "DAG": param2,
        "ExecutionTime": param3,
        "Fitness":param4 ,
        "Agent":param5,
    }

    # Define the CSV file path
    csv_file_path = "Random_results.csv"
    #csv_file_path = "test2.csv"

    # Write metrics to CSV file in append mode
    with open(csv_file_path, "a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=metrics.keys())
        # If the file is empty, write the header
        if csvfile.tell() == 0:
            writer.writeheader()
        writer.writerow(metrics)
    print("Metrics appended to", csv_file_path)

######## FUNCTION OF INITIAL SOLUTION

def fun_task_queues(dag):
    task_attributes = dag.task_attributes
    task_queues = {f"queue_{i}": [] for i in range(8)}  # Initialize task_queues as a dictionary with empty lists

    for task_id, task_info in task_attributes.items():
        cluster = int(task_info["cluster"])  # Convert cluster to an integer
        queue_name = f"queue_{cluster}"
        task_queues[queue_name].append(task_id)

    return task_queues

##
'''def fun_init_solution(dag):
    task_queues = fun_task_queues(dag)
    return task_queues'''

##
def fun_display_task_queues(queues):
    for queue_name, task_list in queues.items():
        print('queue_cluster', queue_name, 'list of tasks', task_list)
    return 0

##

def fun_vm_queues(vm_attributes):
    # Define a dictionary to store the queues
    #queues = {}
    cloud_queues = {}
    edge_queues = {}

    # Your list of VMs
    vm_list = vm_attributes

    # Classify VMs into queues by "network" and then by a combination of "family" and "type"
    for vm in vm_list:
        network = vm['network']
        family = vm['family']
        vm_id = vm['vm_id']

        if network not in edge_queues:
            edge_queues[network] = {}

        if network not in cloud_queues:
            cloud_queues[network] = {}

        # Get the "type" attribute with a default value of "Unknown" if it's missing
        vm_type = vm['type']
        # Create a queue for the family and type combination within the network if it doesn't exist
        family_type = f'{family} - {vm_type}'
        '''if family_type not in queues[network]:
            queues[network][family_type] = []'''

        if family_type not in edge_queues[network]:
            edge_queues[network][family_type] = []

        if family_type not in cloud_queues[network]:
            cloud_queues[network][family_type] = []

        # Add the VM to the appropriate queue
        if network=="edge":
            edge_queues[network][family_type].append(vm_id)
        else:
            cloud_queues[network][family_type].append(vm_id)
            #queues[network][family_type].append(vm_id)

    return edge_queues, cloud_queues
    # Print the queues

##
def fun_display_vm_queues(queues):
    # Print the queues
    for network, network_queues in queues.items():
        print(f'Network: {network}')
        for family_type, vm_ids in network_queues.items():
            print(f'  Family/Type: {family_type}, VMs: {vm_ids}')
    return 0

######################################
def check_constraints(vm_id, task_id):
    # Check memory and frequency constraints for the current VM and task
    print("vm id inside check_constraints function= ", vm_id)
    print ("task_id inside check_constraints function", task_id)
    #if vm_attributes['vm_id'] == vm_id:
    # Access the 'memory' attribute
    remaining_memory = vm_attributes[vm_id]['memory'] - task_attributes[task_id]['memoryNeed_claimed']
    print ('remaining memory', remaining_memory )

    remaining_frequency = vm_attributes[vm_id]['frequency'] - task_attributes[task_id]['frequencyNeed_claimed']
    print ('remaining frequncy', remaining_frequency )

    # Adjust the conditions based on your constraints
    if remaining_memory >= 0 and remaining_frequency >= 0:
        # Update the remaining sizes after the task assignment
        vm_attributes[vm_id]['memory'] = remaining_memory
        vm_attributes[vm_id]['frequency'] = remaining_frequency
        return True
    else:
        return False

###################################

########################

def selectVmFromEdge_with_load_balancing(task_id, vm_allocation_count):
    edge_queues, cloud_queues = fun_vm_queues(vm_attributes)

    #list of edge queues
    edge_queue_MO_megamem = edge_queues["edge"].get("Memory-optimized - megamem", [])
    edge_queue_MO_standard = edge_queues["edge"].get("Memory-optimized - standard", [])
    edge_queue_CO_highcpu = edge_queues["edge"].get("Compute-optimized - highcpu", [])
    edge_queue_CO_standard = edge_queues["edge"].get("Compute-optimized - standard", [])
    edge_queue_small = edge_queues["edge"].get("General-purpose - small", [])
    edge_queue_standard = edge_queues["edge"].get("General-purpose - standard", [])

    # Initialize an empty list
    edge_queue_union = []

    # Append lists to edge_queue_union
    edge_queue_union.extend(edge_queue_MO_megamem)
    edge_queue_union.extend(edge_queue_MO_standard)
    edge_queue_union.extend(edge_queue_CO_highcpu)
    edge_queue_union.extend(edge_queue_CO_standard)
    edge_queue_union.extend(edge_queue_small)
    edge_queue_union.extend(edge_queue_standard)

    print('edge_queue_union========================')
    print(edge_queue_union)



    for vm_id in edge_queue_union:
        if vm_allocation_count[vm_id] <= nb_task_per_vm:
            if check_constraints(vm_id, task_id):
                return vm_id

    print(f'No available edge VM with enough resources.')


def selectVmFromCloud(task_id):
    edge_queues, cloud_queues = fun_vm_queues(vm_attributes)

    #list of edge queues
    cloud_queue_MO_megamem = cloud_queues["cloud"].get("Memory-optimized - megamem", [])
    cloud_queue_MO_standard = cloud_queues["cloud"].get("Memory-optimized - standard", [])
    cloud_queue_CO_highcpu = cloud_queues["cloud"].get("Compute-optimized - highcpu", [])
    cloud_queue_CO_standard = cloud_queues["cloud"].get("Compute-optimized - standard", [])
    cloud_queue_small = cloud_queues["cloud"].get("General-purpose - small", [])
    cloud_queue_standard = cloud_queues["cloud"].get("General-purpose - standard", [])

    # Initialize an empty list
    cloud_queue_union = []

    # Append lists to edge_queue_union
    cloud_queue_union.extend(cloud_queue_MO_megamem)
    cloud_queue_union.extend(cloud_queue_MO_standard)
    cloud_queue_union.extend(cloud_queue_CO_highcpu)
    cloud_queue_union.extend(cloud_queue_CO_standard)
    cloud_queue_union.extend(cloud_queue_small)
    cloud_queue_union.extend(cloud_queue_standard)

    print('cloud_queue_union ========================')
    print(cloud_queue_union)

    for vm_id in cloud_queue_union:
        if check_constraints(vm_id, task_id):
            return vm_id

    print(f'No available cloud VM with enough resources.')





#################### Rondom scheduling
def random_scheduling(task_queues, vms, vm_allocation_count):
    waiting_times = {}
    response_times = {}
    ready_queue = []  # Queue for tasks ready to be scheduled

    # Initialize ready queue with all tasks
    # Append lists to ready_queue
    ready_queue.extend(task_queues['queue_0'])
    ready_queue.extend(task_queues['queue_1'])
    ready_queue.extend(task_queues['queue_2'])
    ready_queue.extend(task_queues['queue_3'])
    ready_queue.extend(task_queues['queue_4'])
    ready_queue.extend(task_queues['queue_5'])
    ready_queue.extend(task_queues['queue_6'])
    ready_queue.extend(task_queues['queue_7'])
    print("================   READY QUEUE =========")
    print(ready_queue)


    #nb_tasks = dag.number_of_nodes
    nb_tasks = len(ready_queue)
    task_allocation = [-1 for _ in range(nb_tasks)]

    current_vm_index = 0  # Index of the current VM in the circular order

    #------- load balancing:  init the list of vm  and t nb tasks in each vm
    nb_vms = len(vms)
    vm_allocation_count = [0] * ( nb_vms)

    #====================================== DEPLOYEMENT OF ENTRY AND EXIT TASKS
    print("============== ENTRY TASKS =====================")
    print(entry_tasks)
    print("============== EXIT TASKS =====================")
    print(exit_tasks)

    vm_id = None
    #================Deploy Entry tasks on the EDGE
    for task_id in entry_tasks :
        vm_id = selectVmFromEdge_with_load_balancing(task_id, vm_allocation_count)
        if vm_id is not None:
            task_allocation[task_id] = vm_id
            vm_allocation_count [vm_id] += 1
        else:
            print('---No Edge VM for entry tasks')

    #=================Deploy exit tasks on the CLOUD
    for task_id in exit_tasks :
        vm_id = selectVmFromCloud(task_id)
        #vm_id = selectVmFromCloud_with_load_balancing(task_id, vm_allocation_count)
        if vm_id is not None:
            task_allocation[task_id] = vm_id
            vm_allocation_count [vm_id] += 1
        else:
            print('---No Cloud VM for entry tasks')


    # Use a list comprehension to filter out entry tasks from ready_queue
    ready_queue = [task_id for task_id in ready_queue if task_id not in entry_tasks]
    ready_queue = [task_id for task_id in ready_queue if task_id not in exit_tasks]
    #===================================== END =====================================

    # Print the updated ready_queue
    print("================ UPDATED READY QUEUE =========")
    print(ready_queue)
    print ("===============TASK ALLOCATION==============")
    print(task_allocation)

    for task_id in ready_queue:
        #task_id = ready_queue.pop(0)  # Dequeue the first task
        print ("select vm inside random_scheduling to task: ", task_id)

        selected_vm = None
        counter = 0
        Max_iteration = len(vms)
        while True:
            vm = random.choice(vms)
            vm_id = vm['vm_id']
            counter += 1
            if check_constraints(vm_id, task_id):
                task_allocation[task_id] = vm_id
                vm_allocation_count [vm_id] += 1
                break

            if counter > Max_iteration:
                break
    return task_allocation, vm_allocation_count
###################################


########### DATA STRUCTURE

def get_transmission_rate(transmission_type):
    rates = {
        "iot_to_edge": 50,  # Mega bits per second
        "edge_to_edge": 1000,  # Mega bits per second
        "edge_to_cloud": 100,   # Mega bits per second
        "cloud_to_edge": 1000,   # Mega bits per second
        "cloud_to_cloud": 10000,   # Mega bits per second
    }
    return rates.get(transmission_type, 0)

def get_vm_pricing(vm_type, vm_subtype, location):
    cloud_vm_types = {
        'General-purpose': {
            'small': {'price': 0.50},
            'standard': {'price': 1.00}
        },
        'Compute-optimized': {
            'standard': {'price': 0.75},
            'highcpu': {'price': 1.50}
        },
        'Memory-optimized': {
            'standard': {'price': 1.20},
            'megamem': {'price': 2.00}
        }
    }


    edge_vm_types = {
        'General-purpose': {
            'small': {'price': 0.0},
            'standard': {'price': 0.0}
        },
        'Compute-optimized': {
            'standard': {'price': 0.0},
            'highcpu': {'price': 0.0}
        },
        'Memory-optimized': {
            'standard': {'price': 0.0},
            'megamem': {'price': 0.0}
        }

    }

    if location.lower() == 'cloud' and vm_type in cloud_vm_types and vm_subtype in cloud_vm_types[vm_type]:
        return cloud_vm_types[vm_type][vm_subtype]['price']
    elif location.lower() == 'edge' and vm_type in edge_vm_types and vm_subtype in edge_vm_types[vm_type]:
        return edge_vm_types[vm_type][vm_subtype]['price']
    else:
        return None

########################################   CALL FUNCTIONS
task_allocation_RO = []
vm_allocation_count = []
#vm_attributes = vm_attributes_copy

nb_vms = len(vm_attributes)
nb_tasks = dag.number_of_nodes()
nb_task_per_vm = round (nb_tasks/nb_vms) + 1

task_queues = fun_task_queues(dag)
edge_queues, cloud_queues = fun_vm_queues(vm_attributes)

#define data structure
task_attributes = dag.task_attributes
entry_tasks = dag.entry_nodes
exit_tasks = dag.exit_nodes
exit_task= exit_tasks
dag_structure = dag.dag_structure
edge_weights = dag.edge_weights



########################## OBJECTIVE FUNCTION

#====================================================
def penalty_Critical_Task(x, task_attributes, vm_attributes):
    NB_Tasks = len(x)
    penaltyCriticalTask = 0
    for task_id in range(NB_Tasks):
        #Constraint : Critical tasks should be placed close to user (on Edge)
        vm_id = x[task_id]
        if (vm_attributes[vm_id]['network'] == "cloud" and task_attributes[task_id]['cluster'] in {1, 5, 6, 7}):
            penaltyCriticalTask += 1
    return penaltyCriticalTask
#====================================================

def penalty_Cap_VM(allocation, task_attributes, vm_attributes):
    liste_of_used_vm = list(set(allocation))
    penaltyCapVM = 0

    for vm_id in liste_of_used_vm:
        # Initialize the sum variables for CPU and memory
        Sum_CPU = 0
        Sum_memory = 0

        # Calculate the sum of CPU and memory needs for tasks allocated to this VM
        for i in range(len(allocation)):
            if allocation[i] == vm_id:
                Sum_CPU += task_attributes[i]['task_size']
                Sum_memory += task_attributes[i]['memoryNeed_claimed']

        # Check if the summed requirements exceed the VM's capacity
        if Sum_CPU > vm_attributes[vm_id]['frequency'] or Sum_memory > vm_attributes[vm_id]['memory']:
            penaltyCapVM += 1
            print('CONST_VM', vm_id)

    return penaltyCapVM

#====================================================
def longest_path_dag(x, dag, dag_structure, task_attributes, edge_weights, vm_attributes, entry_tasks, exit_task):
    memo = {}
    '''x = np.vectorize(int)(x)'''
    # Depth-First Search function
    def dfs(node):
        if node in memo:
            return memo[node]

        max_path = 0
        print('---------- node :')
        print(node)

        for neighbor in dag_structure[node]:
            # Ensure the indices are integers

            # Calculate computing latency (task execution time) and communication latencies (edge: data transmission time)
            computing_latency_neighbor = task_attributes[neighbor]['task_size'] / vm_attributes[int(x[neighbor])]['frequency']

            # assignement of network rates is corrected date: 08/12/2024
            # determine the rate
            if (vm_attributes[x[node]]['network'] == "edge" and vm_attributes[x[neighbor]]['network'] == "edge"):
                rate=get_transmission_rate("edge_to_edge")

            elif (vm_attributes[x[node]]['network'] == "cloud" and vm_attributes[x[neighbor]]['network'] == "edge"):
                rate=get_transmission_rate("cloud_to_edge")

            elif (vm_attributes[x[node]]['network'] == "edge" and vm_attributes[x[neighbor]]['network'] == "cloud"):
                rate=get_transmission_rate("edge_to_cloud")

            elif (vm_attributes[x[node]]['network'] == "cloud" and vm_attributes[x[neighbor]]['network'] == "cloud"):
                rate=get_transmission_rate("cloud_to_cloud")

            elif (vm_attributes[x[node]]['network'] == "iot" and vm_attributes[x[neighbor]]['network'] == "edge"):
                rate=get_transmission_rate("iot_to_edge")
            else:
                rate=0
                print('rate not valid')


            #Calculate communication latency of the current node
            transmitted_data = edge_weights.get((node, neighbor), 0)
            communication_latencie_neighbor = transmitted_data/rate


            #*** BEGIN CODE ADDED 12 JUIN 2024 TO CONSIDER TRANSMITTED DATA FROM SOURCE (ENTRY TASKS) TO NEIGHBORS
            #*** COMMUNICATION LATENCY OF NEIGHBOR WILL BE UPDATED

            if (node in dag.entry_nodes and vm_attributes[x[neighbor]]['network'] == "edge"):
                rate = get_transmission_rate("iot_to_edge")
                communication_latencie_neighbor = transmitted_data/rate

            if (node in dag.entry_nodes and vm_attributes[x[neighbor]]['network'] == "cloud"):
                rateIoTEdge = get_transmission_rate("iot_to_edge")
                rateEdgeCloud = get_transmission_rate("edge_to_cloud")
                communication_latencie_neighbor = transmitted_data/rateIoTEdge + transmitted_data/rateEdgeCloud

            #*** END CODE ADDED 12 JUIN 2024

            #================================================================= CONSTRAINT AND PINALITY

            # constraint : two tasks allocated in the same network and same node(vm), communication latency=0
            if (communication_latencie_neighbor != 0):
                #updated the date of 18 january 2024
                if (vm_attributes[x[neighbor]]['network'] == vm_attributes[x[node]]['network'] and vm_attributes[x[neighbor]]['node'] == vm_attributes[x[node]]['node']):
                    communication_latencie_neighbor = 0

            # total latency_neighbor
            weight = computing_latency_neighbor + communication_latencie_neighbor

            max_path = max(max_path, dfs(neighbor) + weight)
            print('max path', max_path)

        memo[node] = max_path
        return max_path

    '''longest_path = dfs(entry_task)'''

    # if we want to add the weight of entry task, we desactivate the code below

    '''weight_task_entry_task = task_attributes[entry_task]['task_size'] / vm_attributes[x[entry_task]]['frequency']
    print('Entry task:: ',entry_task, 'weight:: ', weight_task_entry_task)

    latency = longest_path + weight_task_entry_task

    print("Longest path from entry task", entry_task, "to exit task", exit_task, ":", longest_path)
    print('latency of the DAG of the agent :',x,' = ', latency)
    print('*********************')'''

    #***CODE ADDED 12 JUIN 2024
    # i update the code to iterate calculating fitness for N entry_tasks done 08 sep 2023
    max_longest_path = 0
    for entry_task in entry_tasks:
        longest_path = dfs(entry_task)
        if longest_path > max_longest_path:
            max_longest_path = longest_path

    latency = max_longest_path
    # end update the code to iterate calculating fitness for N entry_tasks done 08 sep 2023


    ####### adding code 02 juin 2024 to calculate cost
    def cost():
        cost = 0
        costEdge = 0
        costCloud = 0

        size=len(x)

        for counter in range(size):
            computing_time = task_attributes[counter]['task_size'] / vm_attributes[x[counter]]['frequency']
            '''print('Task size', task_attributes[counter]['task_size'])
            vm = vm_attributes[x[counter]]
            print(f'VM frequency: {vm["frequency"]} of VM={vm}')
            print('computing time', computing_time )'''

            if (vm_attributes[x[counter]]['network'] == 'edge'):
                vm_family = vm_attributes[x[counter]]['family']
                vm_type = vm_attributes[x[counter]]['type']
                costEdge = get_vm_pricing(vm_family, vm_type, 'edge') * computing_time

                piEdge=1
                piCloud=0
            else:
                vm_family = vm_attributes[x[counter]]['family']
                vm_type = vm_attributes[x[counter]]['type']
                costCloud = get_vm_pricing(vm_family, vm_type, 'cloud') * computing_time

                piCloud=1
                piEdge=0

            #cost = cost + (costEdge*piEdge + costCloud*piCloud) * computing_time
            cost = cost + costEdge*piEdge + costCloud*piCloud
            print('COST=', cost)

        return cost

    totalCost = cost()

    biObjectives = 0.5 * latency + 0.5 * totalCost
    #penality P1 of VM's size
    penaltyCapVM = penalty_Cap_VM(x, task_attributes, vm_attributes )
    #penality P2 related to critical tasks
    penaltyCriticalTask = penalty_Critical_Task(x, task_attributes, vm_attributes)

    #****************** Return objective parameters
    #fitness calculated without applying penality
    #fitness = latency

    '''fitness calculated and applying penality'''
    #fitness = latency + penaltyCapVM + penaltyCriticalTask
    #fitness = totalCost + penaltyCapVM + penaltyCriticalTask
    fitness = biObjectives + penaltyCapVM + penaltyCriticalTask

    return(fitness)


#################   CALL OBJECTIVE FUNCTION AND GSA


#Allocation of tasks of DAGs on VMs using ML_WPStreamCloud method
vm_attributes = vm_attributes_copy
algo_name = "RAND"
task_allocation_RO, vm_allocation_count = random_scheduling(task_queues, vm_attributes, vm_allocation_count)
x = task_allocation_RO

Fitness = longest_path_dag(x, dag, dag_structure, task_attributes, edge_weights, vm_attributes, entry_tasks, exit_task)

fun_save_metrics_result("RAND_DAG500_FitnessWithP1andP2",None, dag.name,  None, Fitness, x)
fun_save_metrics_result("###########","###########", "###########",  "###########", "###########", "###########")
f.close()