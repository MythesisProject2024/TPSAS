# Fl_trust_composition project : ##cleaned_project.zip##

This project is an update of the FL\_Fl_trust_composition adapted to a regression task.
The target is a continuous TrustIndex (float). Models are simple linear
regression (MSE loss) implemented with NumPy for easy running.
You can download the hole project in zipped form titled: 


Key files:

* model.py           : LinearRegressionModel (NumPy)
* client.py          : FLClient (per-SP local training)
* server.py          : FLServer (FedAvg aggregation)
* generate.py        : generates synthetic dataset based on provided sample
* graph.py, vertex.py: Service Provider graph utilities
* train\_federated.py : orchestrator for federated rounds (evaluates RMSE)
* test.py            : evaluate saved global model (RMSE, MAE, R2)
* data/               : contains generated dataset files (global.csv \& partitions)

Requirements:

* Python 3.8+
* numpy (pip install numpy)

Quick start:

1. Client Data set with equal sizes 1000: 

&nbsp;  RUN python files in this order:

* syhntheticDataSet.py	#generate synthetic dataset for trust client
* generate.py		#generate clients datasets of equal sizes
* train\_federated.py	#training aggregated model
* test.py   		# evaluates saved aggregated model



2\. Client Data set with different sizes 400, 1000, 5000, 12500, 37500:   

RUN python files in this order:

* syhntheticDataSet\_withSizeVariation.py   

* train\_federated.py	#training aggregated model

* test.py   		# evaluates saved aggregated model
  

3\. Client Data set with outliers (feature reponse time (RT)): sizes 1000, 5000, 12500, 20000, 37500:

RUN python files in this order:

* syhntheticDataSet\_withSizeVariation.py
* train\_federated.py	#training aggregated model
* syhntheticDataSet\_withOutliers\_RT.py

* test.py   		# evaluates saved aggregated model  (enter the number 4 to test how the global model behaves with outliers)



4\. Detection and elimination of bad client that dont improve the global model :

RUN python file : 

* scenario3\_7.py


5)To download the hole project :
  download : ##cleaned_project.zip##

&nbsp;

