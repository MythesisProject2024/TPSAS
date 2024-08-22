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

import sys

import prepare_tasks
task_attributes=prepare_tasks.prepare_tasks()
task_list_normalised = []

######## save the dictionnaries in CSV file normalised for clustering##########
for task in task_attributes:
    task_list_normalised.append({
        'task_id': int(task_attributes[task]["task_id"]),
        'task_size': int(task_attributes [task]["task_size"]),
        'frequencyNeed_claimed': int(task_attributes [task]["frequencyNeed_claimed"]/1000000),
        'memoryNeed_claimed': int(task_attributes[task]["memoryNeed_claimed"]),
        'priority': int(task_attributes[task]["priority"]*100)
    })

# Convert the list of dictionaries (task_dataset) into a DataFrame
df_DAG_tasks = pd.DataFrame(task_list_normalised)

'''# Specify the path where you want to save the CSV file
csv_file_path2 = '.\\tasks\\tasks_Montage_50_normalised.csv'

# Save the DataFrame to a CSV file
df.to_csv(csv_file_path2, index=False)

print("tasks montage 50 normalised saved to:", csv_file_path2)'''




#########
df = pd.read_csv(".\\minidatasetlast5000.csv")
#pred= pd.read_csv(".\\tasks_Montage_50_normalised.csv")

colnames = ['CPUNeed_Claimed', 'RAMNeed_Claimed', 'PriorityNo']

#df.describe()
#print(df.head())
#df.info()

x = df[['RAMNeed_Claimed', 'CPUNeed_Claimed', 'PriorityNo']].values
#xpred = pred[['memoryNeed_claimed', 'frequencyNeed_claimed', 'priority']].values
xpred = task_list_normalised

'''
# find the optimal number of clusters using elbow method  -- >This is for 3 features = [ram,cpu ,priority]

WCSS = []
for i in range(1,11):
    model = KMeans(n_clusters = i,init = 'k-means++')
    model.fit(x)
    WCSS.append(model.inertia_)
fig = plt.figure(figsize = (5,4))
plt.plot(range(1,11),WCSS, linewidth=4, markersize=12,marker='o',color = 'red')
plt.xticks(np.arange(11))
plt.xlabel("Number of clusters")
plt.ylabel("WCSS")
plt.show()
'''
# finding the clusters based on input matrix "x"
model = KMeans(n_clusters = 8, init = "k-means++", max_iter = 300, n_init = 10, random_state = 0)
#============================================================================
from sklearn.preprocessing import MinMaxScaler

# Create a MinMaxScaler object
scaler = MinMaxScaler()
# Fit the scaler on the training data
scaler.fit(x)
# Apply the scaler to the training data
scaled_training_data = scaler.transform(x)
x=scaled_training_data
# Apply the same scaler to the new data

#scaled_new_data = scaler.transform(xpred)
#xpred=scaled_new_data'''

#============================================================================
#training model
model.fit(x)
#predict clusters of x dataset
y_clusters = model.predict(x)

# get centroids
print("centroids----------------------------------------------- ")
centroids = model.cluster_centers_
print(centroids)

def euclidean_distance(point, centroid):
    return np.sqrt(np.sum((point - centroid)**2))

#============================================================================
print ('====================determine cluster of one point======================')


# determine cluster of a point
data_point = np.array([8432.186, 1086, 0])
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

print("Data point:", data_point)
print("Closest cluster:", closest_cluster)
#===============================================================================
print ('====================determine cluster of inpuDataset======================')

#determine cluster of each point of dataset
def clustering (xpred, centroids):
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
clusters_of_xpred = clustering (xpred, centroids)

print("Data points:", xpred)
print("Clusters:", clusters_of_xpred)

#insert culumn class in inputdataset pred
#insert culumn class in inputdataset pred
'''pred.insert(loc=5,
        column='cluster',
        value=1)
pred['cluster']= clusters_of_xpred
print(pred.head())
'''

# save physically the dataset tasks_Montage_50.csv
# to have  clusterd dataset with original values (not normalised)
inputdataset= pd.read_csv(".\\tasks_Montage_50.csv")

#insert culumn class in inputdataset pred
inputdataset.insert(loc=5,
        column='cluster',
        value=1)
inputdataset['cluster']= clusters_of_xpred
print(inputdataset.head())

inputdataset.to_csv(".\\tasks_Montage_50_clustered.csv", index=False)




'''

#==============================================================================
#y_clusters = model.fit_predict(x)

#pred_clusters=model.predict(xpred)
#y_clusters=model.predict(xpred)
#y_clusters=clusters
#y_clusters=model.predict(x)
#pred_clusters = model.fit_predict(xpred)

#print(pred_clusters)
#pred.insert(loc=10, column='y_clusters',value=pred_clusters)
#print ("===================prediction=============================")
#print(pred.head())



# 3d scatterplot using matplotlib
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x[y_clusters == 0,0],x[y_clusters == 0,1],x[y_clusters == 0,2], s = 20 , color = 'blue', label = "cluster 1")
ax.scatter(x[y_clusters == 1,0],x[y_clusters == 1,1],x[y_clusters == 1,2], s = 20 , color = 'orange', label = "cluster 2")
ax.scatter(x[y_clusters == 2,0],x[y_clusters == 2,1],x[y_clusters == 2,2], s = 20 , color = 'green', label = "cluster 3")
ax.scatter(x[y_clusters == 3,0],x[y_clusters == 3,1],x[y_clusters == 3,2], s = 20 , color = '#D12B60', label = "cluster 4")
ax.scatter(x[y_clusters == 4,0],x[y_clusters == 4,1],x[y_clusters == 4,2], s = 20 , color = 'purple', label = "cluster 5")
ax.scatter(x[y_clusters == 5,0],x[y_clusters == 5,1],x[y_clusters == 5,2], s = 20 , color = 'black', label = "cluster 6")
ax.scatter(x[y_clusters == 6,0],x[y_clusters == 6,1],x[y_clusters == 6,2], s = 20 , color = 'red', label = "cluster 7")
ax.scatter(x[y_clusters == 7,0],x[y_clusters == 7,1],x[y_clusters == 7,2], s = 20 , color = 'yellow', label = "cluster 8")
ax.set_xlabel('Memory-->')
ax.set_ylabel('CPU-->')
ax.set_zlabel('Priority-->')
ax.legend()
plt.show()


# 3d scatterplot using plotly
Scene = dict(xaxis = dict(title  = 'Memory -->'),yaxis = dict(title  = 'CPU--->'),zaxis = dict(title  = 'Priority-->'))

# model.labels_ is nothing but the predicted clusters i.e y_clusters
labels = model.labels_
trace = go.Scatter3d(x=x[:, 0], y=x[:, 1], z=x[:, 2], mode='markers',marker=dict(color = labels, size= 5, line=dict(color= 'black',width = 5)))
layout = go.Layout(margin=dict(l=0,r=0),scene = Scene,height = 800,width = 800)
data = [trace]
fig = go.Figure(data = data, layout = layout)
fig.show()

'''
