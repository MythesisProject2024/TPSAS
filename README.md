# Thesis_Project_Code

## Description

This repository includes the Python code for my Stream Application Scheduling approach.

## Getting Started

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

## How to Download Your Program

- Download the zipped file "projet_these_Saber_github.7z" located in the repository on my GitHub account. [URL:  https://github.com/SabeurLajili/Thesis_Project_Code]

## How to Run the Program

1. Decompress the folder containing all programs and files.
2. To run GSA algorithm, Open the Python file: `GSA_Mealpy_gitHubVersion.py` in python IDE, and then run it
3. To run GA algorithm, Open the Python file: `GA_Mealpy_gitHubVersion.py` in python IDE, and then run it

### To run GSA and GA algorithm

1. Import the appropriate dataset.
   Example: To process DAG with size=50, import its dataset like this :   `import readClusteredDAGfile50`
   and the same for all DAGs starting from 50 to 700.   
3. Initialize the list of clustered DAGs.
   Example: `DAG_List_clustred = readClusteredDAGfile50.DAG_List_clustred`
4. Update the population variable.
   Example: `problem_size = 50  # Dimension: number of tasks`

## Output and Results

After running the program, the results are saved in an Excel files named `GSA_results.xls` and GA_results.xls.

