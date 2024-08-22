# TPSAS

## Description
This repository contains the Python code for experimentation of the proposed TPSAS approach of our paper, which has been submitted to a conference and is currently under review.

## Getting Started
This python project includes python scripts, task dataset and VM data set. 
The scripts code includes four main programs:  
GA_Mealpy_gitHubVersion.py, GSA_Mealpy_gitHubVersion.py, GSAP_Mealpy_gitHubVersion.py, random_scheduling.py
Datasets are: task_dataset2.csv , cloud200_edge100_vms_SameConfiguration.csv

### Dependencies
- You need to download a new version of a Python IDE or Pyzo IDE.
- Required libraries: pandas, networkx, mealpy, etc.
- Use the command `pip` to install any necessary library from the internet. 
  Example: `pip install mealpy`

### OS Version
- Example: Windows 10

### Installing
- Install only the required Python libraries.
  Example: mealpy - An open-source library for the latest meta-heuristic algorithms in Python.

### Configuration of Mealpy: adding of GSA algorithm
In the currect version of Mealpy library, GSA algorithm is not included, we need to addit as following:
- add GSA.py to the folder "...\mealpy\swarm_based\".
- add this line "from .GSA import GSA" in  "...\mealpy\swarm_based\_init_.py"
- Import GSA.py in any python script: from mealpy.swarm_based.GSA  import GSA
- Import GA.py in any python script: from mealpy import FloatVar, GA
  
### How to Run the Program
1. Decompress the folder containing all programs and files.
2. To run GA algorithm, Open the Python file `GA_Mealpy_gitHubVersion.py` in python IDE, and then run it
3. To run GSA algorithm, Open the Python file `GSA_Mealpy_gitHubVersion.py` in python IDE, and then run it
4. To run GSAP algorithm, Open the Python file `GSAP_Mealpy_gitHubVersion.py` in python IDE, and then run it
5. To run RAND algorithm, Open the Python file `random_scheduling.py` in python IDE, and then run it.

### To run GSA and GA algorithm

1. Import the appropriate dataset.
   Example: To process DAG with size=50, import its dataset like this :   `import readClusteredDAGfile50`
   and the same for all DAGs starting from 50 to 700.   
3. Initialize the list of clustered DAGs.
   Example: `DAG_List_clustred = readClusteredDAGfile50.DAG_List_clustred`
4. Update the problem dimension, number of iteration and runs variables.
   Example: 
          lower_bound = 0       #nb_of_VMs =300 from 0 to 299
          upper_bound = 299
          domain_range = [lower_bound, upper_bound]
          problem_size = 50     #Dimension is the nb tasks          
          epoch = 1000          #nb iteration
          pop_size = 50         #population size
          num_runs = 20         #Dimension: number of runs`

## Output and Results

After running the program, the results are saved in an Excel files named `GSA_results.xls` and GA_results.xls.
