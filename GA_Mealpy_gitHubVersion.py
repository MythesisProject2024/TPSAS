import numpy as np
import numpy
import math
from mealpy.swarm_based.GSA import GSA
#from mealpy.evolutionary_based.GA import BaseGA

from mealpy import FloatVar, GA


####################################################3

import networkx as nx
import pandas as pd
import os

import random
import copy
import numpy
import math
#from solution import solution
import time
#import massCalculation
#import gConstant
#import gField
import move
import csv

#---Sabeur updating
import sys
f=open('output1.txt', 'w')
# We redirect the 'sys.stdout' command towards the descriptor file
sys.stdout = f
#---End Sabeur updating



'''
#dag files size 100
import readDAGfiles100'''

import readClusteredDAGfile50

DAG_List_clustred = readClusteredDAGfile50.DAG_List_clustred
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
def fun_save_metrics_result(param0 = None, param1 = None, param2 = None, param3 = None, param4 = None ):
    # Your code to calculate metrics here...

    # Organize metrics into a dictionary
    metrics = {
        "Algorithm": param0,
        "DAG": param1,
        "ExecutionTime": param2,
        "Fitness":param3 ,
        "Agent":param4
    }

    # Define the CSV file path
    csv_file_path = "GA_results.csv"
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

def selectVmFromEdge(task_id):
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

    #if vm_allocation_count[vm_id] < nb_task_per_vm:

    for vm_id in edge_queue_union:
        if check_constraints(vm_id, task_id):
            return vm_id

    print(f'No available edge VM with enough resources.')
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


def fun_select_vm(task_id, queue, edge_queues, cloud_queues):
    selected_edge_vm_id = None
    selected_cloud_vm_id = None
    selected_edge_standard_vm_id = None
    selected_edge_small_vm_id = None
    selected_cloud_small_vm_id = None
    selected_cloud_standard_vm_id = None

    #list of edge queues

    '''
    edge_queue_MO_standard = edge_queues["edge"].get("Memory-optimized - standard", [])
    edge_queue_CO_highcpu = edge_queues["edge"].get("Compute-optimized - highcpu", [])
    edge_queue_CO_standard = edge_queues["edge"].get("Compute-optimized - standard", [])
    edge_queue_small = edge_queues["edge"].get("General-purpose - small", [])
    edge_queue_standard = edge_queues["edge"].get("General-purpose - standard", [])'''

    print("Select vm to task :: ", task_id)
    if queue == 'queue_7' or queue == 'queue_2':
        #edge_queue = edge_queues["edge"].get("Memory-optimized - megamem", [])
        edge_queue_MO_megamem = edge_queues["edge"].get("Memory-optimized - megamem", [])
        if queue == 'queue_7':
            for vm_id in edge_queue_MO_megamem:
                if check_constraints(vm_id, task_id):
                    selected_edge_vm_id = vm_id
                    return selected_edge_vm_id
                    #break  # Exit the loop when a suitable VM is found
            if selected_edge_vm_id is None:
                print(f'No available edge VM with enough resources in megamem type.')

        # Continue with cloud queue
        #cloud_queue = cloud_queues["cloud"].get("Memory-optimized - megamem", [])
        cloud_queue_MO_megamem = cloud_queues["cloud"].get("Memory-optimized - megamem", [])
        for vm_id in cloud_queue_MO_megamem:
            if check_constraints(vm_id, task_id):
                selected_cloud_vm_id = vm_id
                return selected_cloud_vm_id
                #break  # Exit the loop when a suitable VM is found
        if selected_cloud_vm_id is None:
            print(f'No available cloud VM with enough resources in megamem type.')

        #return selected_edge_vm_id if selected_edge_vm_id is not None and queue == 'queue_7' else selected_cloud_vm_id

    #---------------------------------------------------
    elif queue == 'queue_5' or queue == 'queue_3':
        if queue == 'queue_5':
            #edge_queue = edge_queues["edge"].get("Memory-optimized - standard", [])
            edge_queue_MO_standard = edge_queues["edge"].get("Memory-optimized - standard", [])
            for vm_id in edge_queue_MO_standard:
                if check_constraints(vm_id, task_id):
                    selected_edge_vm_id = vm_id
                    return selected_edge_vm_id
                    #break  # Exit the loop when a suitable VM is found
            if selected_edge_vm_id is None:
                print(f'No available edge VM with enough resources Memory optimized in standard type.')

        # Continue with cloud queue
        cloud_queue_MO_standard = cloud_queues["cloud"].get("Memory-optimized - standard", [])
        for vm_id in cloud_queue_MO_standard:
            if check_constraints(vm_id, task_id):
                selected_cloud_vm_id = vm_id
                return selected_cloud_vm_id
                #break  # Exit the loop when a suitable VM is found

        if selected_cloud_vm_id is None:
            print(f'No available cloud VM with enough resources Memory optimized in standard type.')

        #return selected_edge_vm_id if selected_edge_vm_id is not None and queue == 'queue_5' else selected_cloud_vm_id
    #----------------------------------------------------------
    elif queue == 'queue_1' or queue == 'queue_4':
        if queue == 'queue_1':
            edge_queue_CO_highcpu = edge_queues["edge"].get("Compute-optimized - highcpu", [])
            for vm_id in edge_queue_CO_highcpu:
                if check_constraints(vm_id, task_id):
                    selected_edge_vm_id = vm_id
                    return selected_edge_vm_id

            if selected_edge_vm_id is None:
                print(f'No available edge VM with enough resources in Compute-optimized - highcpu type.')

            edge_queue_CO_standard = edge_queues["edge"].get("Compute-optimized - standard", [])
            for vm_id in edge_queue_CO_standard:
                if check_constraints(vm_id, task_id):
                    selected_edge_vm_id = vm_id
                    return selected_edge_vm_id

            if selected_edge_vm_id is None:
                print(f'No available edge VM with enough resources in compute optimized - standard type.')

        # Continue with cloud queue
        cloud_queue_CO_highcpu = cloud_queues["cloud"].get("Compute-optimized - highcpu", [])
        for vm_id in cloud_queue_CO_highcpu:
            if check_constraints(vm_id, task_id):
                selected_cloud_vm_id = vm_id
                return selected_cloud_vm_id

        if selected_cloud_vm_id is None:
            print(f'No available cloud VM with enough resources in compute optimized -highcpu type.')

        cloud_queue_CO_standard = cloud_queues["cloud"].get("Compute-optimized - standard", [])
        for vm_id in cloud_queue_CO_standard:
            if check_constraints(vm_id, task_id):
                selected_cloud_vm_id = vm_id
                return selected_cloud_vm_id

        if selected_cloud_vm_id is None:
            print(f'No available cloud VM with enough resources in compute optimized -standard type.')

        #return selected_edge_vm_id if selected_edge_vm_id is not None and queue == 'queue_1' else selected_cloud_vm_id

    #-----------------------------------------
    elif queue == 'queue_6' or queue == 'queue_0':
        edge_queue_small = edge_queues["edge"].get("General-purpose - small", [])
        for vm_id in edge_queue_small:
            if check_constraints(vm_id, task_id) == True:
                print("VM -small found", vm_id)
                selected_edge_small_vm_id = vm_id
                return selected_edge_small_vm_id
        if selected_edge_small_vm_id is None:
            print(f'No available edge VM with enough resources in small type.')

        edge_queue_standard = edge_queues["edge"].get("General-purpose - standard", [])
        for vm_id in edge_queue_standard:
            if check_constraints(vm_id, task_id) == True:
                print("VM -standard found", vm_id)
                selected_edge_standard_vm_id = vm_id
                return selected_edge_standard_vm_id
        if selected_edge_standard_vm_id is None:
            print(f'No available edge VM with enough resources in standard type.')

        #---------------------------------------------- cloud
        cloud_queue_small = cloud_queues["cloud"].get("General-purpose - small", [])
        for vm_id in cloud_queue_small:
            if check_constraints(vm_id, task_id) == True:
                print("VM -small found", vm_id)
                selected_cloud_small_vm_id = vm_id
                return selected_cloud_small_vm_id
        if selected_cloud_small_vm_id is None:
            print(f'No available cloud VM with enough resources in small type.')

        cloud_queue_standard = cloud_queues["cloud"].get("General-purpose - standard", [])
        for vm_id in cloud_queue_standard:
            if check_constraints(vm_id, task_id) == True:
                print("VM -standard found", vm_id)
                selected_cloud_standard_vm_id = vm_id
                return selected_cloud_standard_vm_id
        if selected_cloud_standard_vm_id is None:
            print(f'No available cloud VM with enough resources in standard type.')

    # in all other case when the apropriate VM not found:
    # select vm from edge
    selected_edge_vm_id = selectVmFromEdge(task_id)
    if selected_edge_vm_id is not None:
        return selected_edge_vm_id
    #select vm from cloud
    selected_cloud_vm_id = selectVmFromCloud(task_id)
    if selected_cloud_vm_id is not None:
        return selected_cloud_vm_id

def fun_resource_allocator_with_profiling(task_queues, edge_queues, cloud_queues, vm_allocation_count):
    critical_task_queues = ['queue_6', 'queue_1', 'queue_5', 'queue_7']
    not_critical_task_queues = ['queue_0', 'queue_4', 'queue_3', 'queue_2']
    nb_tasks = dag.number_of_nodes()
    task_allocation = [-1] * nb_tasks  # Initialize a list of size 50 with -1 values

    vm_allocation_count = [0] * ( nb_vms)

    #====================================== DEPLOYEMENT OF ENTRY AND EXIT TASKS
    print("============== ENTRY TASKS =====================")
    print(entry_tasks)
    print("============== EXIT TASKS =====================")
    print(exit_tasks)

    vm_id = None
    #================Deploy Entry tasks on the EDGE

    for task_id in entry_tasks :
        #vm_id = selectVmFromEdge_with_load_balancing(task_id, vm_allocation_count)
        ''' code ipdated on 04 juin 2024 to let entry tasks loaded to edge and to suitable VM'''
        cluster = task_attributes[task_id]['cluster']
        cluster = int(cluster) # Convert cluster to an integer
        queue_name = f"queue_{cluster}"
        print("queue_name", queue_name)
        vm_id = fun_select_vm(task_id, queue_name , edge_queues, cloud_queues)
        ''' end modification '''
        if vm_id is not None:
            task_allocation[task_id] = vm_id
            vm_allocation_count [vm_id] += 1
        else:
            print('---No Edge VM for entry tasks')

    #=================Deploy exit tasks on the CLOUD
    for task_id in exit_tasks :
        vm_id = selectVmFromCloud(task_id)
        if vm_id is not None:
            task_allocation[task_id] = vm_id
            vm_allocation_count [vm_id] += 1
        else:
            print('---No Cloud VM for entry tasks')

    #===================================== END =====================================

    print ("===============TASK ALLOCATION==============")
    print(task_allocation)

    # assigning the critical tasks first
    for queue in critical_task_queues:
        if len(task_queues[queue]) != 0:
            for task_id in task_queues[queue]:
                if task_id not in entry_tasks and task_id not in exit_tasks:
                    vm_id = fun_select_vm(task_id, queue, edge_queues, cloud_queues)
                    print('task id before assignement', task_id)
                    print('vm id before allocation: ', vm_id)
                    if vm_id is not None:
                        task_allocation[task_id] = vm_id
                        vm_allocation_count [vm_id] += 1
                    else:
                        print(f'No available VM for critical task {task_id} in queue {queue}.')


    for queue in not_critical_task_queues:
        if len(task_queues[queue]) != 0:
            for task_id in task_queues[queue]:
                if task_id not in entry_tasks and task_id not in exit_tasks:
                    vm_id = fun_select_vm(task_id, queue, edge_queues, cloud_queues)
                    if vm_id is not None:
                        task_allocation[task_id] = vm_id
                        vm_allocation_count [vm_id] += 1
                    else:
                        print(f'No available VM for non-critical task {task_id} in queue {queue}.')
    return task_allocation, vm_allocation_count


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

    '''edge_vm_types = {
        'General-purpose': {
            'small': {'price': 0.30},
            'standard': {'price': 0.60}
        },
        'Compute-optimized': {
            'standard': {'price': 0.45},
            'highcpu': {'price': 0.90}
        },
        'Memory-optimized': {
            'standard': {'price': 0.60},
            'megamem': {'price': 1.00}
        }
        '''
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
task_allocation_with_profiling = []
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

#define cost cloud and edge resource utlisation unit.  code added 02 juin 2024

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
    x = np.vectorize(int)(x)
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

    penaltyCapVM = penalty_Cap_VM(x, task_attributes, vm_attributes )

    penaltyCriticalTask = penalty_Critical_Task(x, task_attributes, vm_attributes)

    #return objective parameters
    fitness = latency
    fitness = fitness  +  penaltyCapVM + penaltyCriticalTask

    return(fitness)


#################   CALL OBJECTIVE FUNCTION AND GSA


# Define the problem dimensions and boundaries
problem_size = 50  # Dimension nb tasks
lower_bound = 0    # nb VMs from 0 to 299
upper_bound = 299
domain_range = [lower_bound, upper_bound]
epoch = 1000         #nb iteration
pop_size = 50        #population
num_runs = 1

# Initialize a list to store the results
results = []

#Allocation of tasks of DAGs on VMs using ML_WPStreamCloud method
algo_name = "ML_WPStreamCloud"
task_allocation_with_profiling, vm_allocation_count = fun_resource_allocator_with_profiling(task_queues, edge_queues, cloud_queues, vm_allocation_count)

vm_attributes = vm_attributes_copy

def objective_function(solution):
    # Define your DAG, attributes, etc. Replace with actual data structures.
    # .....
    # Call longest_path_dag with all required arguments
    return longest_path_dag(solution, dag, dag_structure, task_attributes, edge_weights, vm_attributes, entry_tasks, exit_task)


# Perform multiple runs of the optimization process
for run in range(num_runs):
    X = task_allocation_with_profiling
    print('X=', X)

    timerStart=time.time()
    #------------------------------------CALL OF GSA
    '''#gsa = GSA(problem_size, domain_range, longest_path_dag, epoch, pop_size, X, dag, dag_structure, task_attributes, edge_weights, vm_attributes, entry_tasks, exit_task)
    gsa = GSA(problem_size, domain_range, longest_path_dag, epoch, pop_size, None, dag, dag_structure, task_attributes, edge_weights, vm_attributes, entry_tasks, exit_task)
    best_solution, best_fitness = gsa.solve()'''

    #------------------------------------CALL OF GA

    # Instantiate and run the GA algorithm
    #model = BaseGA(epoch=1000, pop_size=50, pc=0.9, pm=0.05)
    #best_position, best_fitness = model.solve(problem_dict)

    #ga = BaseGA(problem_dict)
    #best_position, best_fitness = ga.solve()

    # Example usage of genetic algorithm with your objective function
    problem_dict = {
        "bounds": [FloatVar(lb=0, ub=299, name=f"VM_{i}") for i in range(len(task_attributes))],  # Bounds for population (nb tasks)
        "obj_func": objective_function,
        "minmax": "min"
    }

    model = GA.BaseGA(epoch=1000, pop_size=50, pc=0.9, pm=0.05)
    g_best = model.solve(problem_dict)

    best_fitness, best_solution = g_best.target.fitness, g_best.solution
    best_solution= np.round(best_solution).astype(int)

    print(f"Run {run + 1}/{num_runs} - Best Solution: {best_solution}, Best Fitness: {best_fitness}")
    executionTime = time.time()-timerStart
    fun_save_metrics_result("GA_DAG50_withP1andP2",dag.name,  executionTime, best_fitness, best_solution)

f.close()