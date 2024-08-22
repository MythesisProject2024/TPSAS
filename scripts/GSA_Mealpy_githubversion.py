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

#---Sabeur updating
import sys
f=open('output1.txt', 'w')
# We redirect the 'sys.stdout' command towards the descriptor file
sys.stdout = f
#---End Sabeur updating

#dag files size 50 to 700 clusterd
import readClusteredDAGfile50

#dag files size 100 not clustered
#import readDAGfiles100
#DAG_List_clustred = readDAGfiles100.DAG_List_clustred

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


# Function to save results == best solution and best fitness
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
    csv_file_path = "GSA_resultsDAG50.csv"
    #csv_file_path = "test2.csv"

    # Write metrics to CSV file in append mode
    with open(csv_file_path, "a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=metrics.keys())
        # If the file is empty, write the header
        if csvfile.tell() == 0:
            writer.writeheader()
        writer.writerow(metrics)
    print("Metrics appended to", csv_file_path)




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

########################################  other parameters
#task_allocation_with_profiling = []
vm_allocation_count = []
#vm_attributes = vm_attributes_copy

nb_vms = len(vm_attributes)
nb_tasks = dag.number_of_nodes()
#nb_task_per_vm = round (nb_tasks/nb_vms) + 1

#task_queues = fun_task_queues(dag)
#edge_queues, cloud_queues = fun_vm_queues(vm_attributes)

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


# Define the problem dimensions and boundaries
problem_size = 50  # Dimension nb tasks
lower_bound = 0    # nb VMs from 0 to 299
upper_bound = 299
domain_range = [lower_bound, upper_bound]
epoch = 1000          #nb iteration
pop_size = 50        #population
num_runs = 1

# Initialize a list to store the results
results = []

vm_attributes = vm_attributes_copy

# Perform multiple runs of the optimization process
SumExecutionTime = 0
SumFitness = 0
for run in range(num_runs):

    timerStart=time.time()
    #CALL OF GSA
    gsa = GSA(problem_size, domain_range, longest_path_dag, epoch, pop_size, None, dag, dag_structure, task_attributes, edge_weights, vm_attributes, entry_tasks, exit_task)

    best_solution, best_fitness = gsa.solve()
    executionTime = time.time()-timerStart

    fun_save_metrics_result("GSA_DAG50_10MB_FitnessWithP1andP2",run, dag.name,  executionTime, best_fitness, best_solution)

    SumFitness = SumFitness + best_fitness
    SumExecutionTime = SumExecutionTime + executionTime

#print(f"fitness Average of {num_runs} : {averageFitness}")
AvgFitness = SumFitness/num_runs
AvgExecutiontime =   SumExecutionTime/num_runs
fun_save_metrics_result("GSA_DAG50_AverageFitnessWithP1andP2",num_runs, dag.name,  AvgExecutiontime, AvgFitness, None)
fun_save_metrics_result("###########","###########", "###########",  "###########", "###########", "###########")
f.close()