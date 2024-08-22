# TPSAS

## Description
This repository contains the Python code for experimentation of the proposed TPSAS approach of our paper, which has been submitted to a conference and is currently under review.

## Getting Started
- This python project includes python scripts, task dataset and VM dataset. 
- The scripts code includes four main programs:  GA_Mealpy_gitHubVersion.py, GSA_Mealpy_gitHubVersion.py, GSAP_Mealpy_gitHubVersion.py, random_scheduling.py
- Datasets are: task_dataset2.csv , cloud200_edge100_vms_SameConfiguration.csv

### OS Version
- Example: Windows 10

### Installing
Download the required Python libraries and components:

- A recent version of a Python IDE or Pyzo IDE.
- Mealpy: an open-source library for the latest meta-heuristic algorithms in Python.
- Other required libraries: pandas, networkx, numpy, etc.
- Use the pip command to install any necessary libraries from the internet.
  Example: pip install mealpy

### Configuration of Mealpy: adding of GSA algorithm
In the current version of the Mealpy library, the GSA algorithm is not included. We need to add it as follows:

- Copy the Python file "scripts/GSA.py" into your local folder at "...\mealpy\swarm_based\".
- Add the line "from .GSA import GSA" to ...\mealpy\swarm_based\__init__.py.
- Import GSA.py in any Python script using: from mealpy.swarm_based.GSA import GSA.
- Import GA.py in any Python script using: from mealpy import GA.
  
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
