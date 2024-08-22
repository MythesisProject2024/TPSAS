# import the libraries
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objs as go
from plotly import tools
from plotly.subplots import make_subplots
import plotly.offline as py


from sklearn.metrics.pairwise import euclidean_distances
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import scale

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
scaler = StandardScaler()


trainingDataset = "D:\\python\\projet_these_sabeur_experimentation\\GSAmaster_my_solution\\tasks\\minidatasetlast5000.csv"
    #########
def clusterDAG(inputDataset, inputDatasetNormalised, outputDataset):
    #df = pd.read_csv(".\\minidatasetlast5000.csv")
    df = pd.read_csv(trainingDataset)

    #pred= pd.read_csv(".\\tasks_Montage_50_normalised.csv")
    pred= pd.read_csv(inputDatasetNormalised)

    #df.describe()
    #print(df.head())
    #df.info()

    x = df[['RAMNeed_Claimed', 'CPUNeed_Claimed', 'PriorityNo']].values
    xpred = pred[['memoryNeed_claimed', 'frequencyNeed_claimed', 'priority']].values

    # finding the clusters based on input matrix "x"
    model = KMeans(n_clusters = 8, init = "k-means++", max_iter = 300, n_init = 10, random_state = 0)

    #==========================MinMaxScaler for training dataset
    from sklearn.preprocessing import MinMaxScaler

    # Create a MinMaxScaler object
    scaler = MinMaxScaler()
    # Fit the scaler on the training data
    scaler.fit(x)
    # Apply the scaler to the training data
    scaled_training_data = scaler.transform(x)
    x=scaled_training_data
    # Apply the same scaler to the new data
    scaled_new_data = scaler.transform(xpred)
    xpred=scaled_new_data

    #=========================== training model
    model.fit(x)
    #predict clusters of x dataset
    y_clusters = model.predict(x)
    '''
    print("========  Training Dataset =============")
    print(x)
    print("========  Clusters of Traing Dataset =============")
    for cluster in y_clusters:
        print(cluster)
    '''
    #predict clusters of inpudataset=xpred
    print("========  predicted Dataset =============")
    clusters_of_xpred = model.predict(xpred)
    for i in range(0, len(clusters_of_xpred), 10):
        print('\t'.join(map(str, clusters_of_xpred[i:i+10])))

    # get centroids
    #print("------centroids------ ")
    centroids = model.cluster_centers_
    #print(centroids)

    def euclidean_distance(point, centroid):
        return np.sqrt(np.sum((point - centroid)**2))

    #================determine cluster of one point
    #print ('===determine cluster of one point===')
    data_point = np.array([1086, 8432.186, 0])
    centroids = np.array([[3073.66220238, 7896.7030253, 122.43154762],
                        [2066.83283582, 6781.86623433, 8621.93134328],
                        [998.53813559, 7760.57925565, 122.04943503],
                        [1933.41796875, 6924.41983789, 3214.18359375],
                        [3063.54931715, 5500.30636874, 200.60242792],
                        [991.925, 5535.08563281, 176.1875],
                        [1978.02653631, 6607.97854609, 5975.5698324]])

    distances = []
    for centroid in centroids:
        distance = euclidean_distance(data_point, centroid)
        distances.append(distance)

    closest_cluster = np.argmin(distances)

    #print("Data point:", data_point)
    #print("Closest cluster:", closest_cluster)
    #print("=================================================")

    #=====determine cluster of inpuDataset nornalised
    #print ('===determine cluster of inpuDataset nornalised ===')

    #===========================determine cluster of each point of dataset
    '''def clustering (xpred):
        xtest=xpred
        clusters = []
        for data_point in xtest:
            distances = []
            for centroid in centroids:
                distance = euclidean_distance(data_point, centroid)
                distances.append(distance)
            closest_cluster = np.argmin(distances)
            clusters.append(closest_cluster)
        return clusters

    clusters_of_xpred = []
    clusters_of_xpred = clustering (xpred)

    print datapoints and clusters of predicted dataset (disable print now)
    print("Data points:", xpred)
    print("Clusters:", clusters_of_xpred)'''


    # save physically the dataset tasks_Montage_50.csv
    # to have  clusterd dataset with original values (not normalised)
    #outputdataset= pd.read_csv(".\\tasks_Montage_50.csv")
    inputdatasetDF= pd.read_csv(inputDataset)

    #insert culumn class in outputdatasetDF pred
    inputdatasetDF.insert(loc=5,
            column='cluster',
            value=1)
    inputdatasetDF['cluster']= clusters_of_xpred

    #print ('===display the head of outputdatasetDF===')
    #print(inputdatasetDF.head())

    inputdatasetDF.to_csv(outputDataset, index=False)
    return 0




'''
#Dataset = ".\\tasks_Montage_50.csv"
Dataset = "D:\\python\\projet_these_sabeur_experimentation\\GSAmaster_my_solution\\tasks\\tasks_Montage_50.csv"

#DatasetNormalised = ".\\tasks_Montage_50_normalised.csv"
DatasetNormalised = "D:\\python\\projet_these_sabeur_experimentation\\GSAmaster_my_solution\\tasks\\tasks_Montage_50_normalised.csv"

clusterd_Dataset = "D:\\python\\projet_these_sabeur_experimentation\\GSAmaster_my_solution\\tasks\\tasks_Montage_50_clustered.csv"

############# call function clusteringDAG
clusterDAG(Dataset, DatasetNormalised, clusterd_Dataset)'''
























