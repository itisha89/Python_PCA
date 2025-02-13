# %%
# loading packages

import os

import pandas as pd
import numpy as np

# plotting packages
%matplotlib inline
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as clrs
import seaborn as sns 
sns.set()
# Kmeans algorithm from scikit-learn
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score


# %% [markdown]
# ## Load raw data

# %%
# load raw data
DATA_FOLDER = './'
raw = pd.read_csv(os.path.join(DATA_FOLDER, 'EUEmployment.csv'))
# check the raw data
columns=[str(i).replace('(','').replace(')','') for i in raw.loc[0,:].values]
columns[1]='Group'
raw.columns=columns
Data=raw.loc[1:]
Data.head(31)
for col in Data.columns:
    if col not in ['Country','Group']:
        Data[col] = Data[col].astype(float)

# %%
Data.shape

# %%
import numpy as np
p=np.arange(.1,1,.1)
Data.describe(p)

# %%
Data['Country'].nunique()
Data['Country'].unique()


# %%
Data['Group'].nunique()
Data['Group'].unique()


# %%
Data.columns

# %%
correlation_matrix = Data[['MAG', 'PS', 'CON', 'SER', 'FIN','SPS', 'TC']].corr().round(2)
# annot = True to print the values inside the square
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(data=correlation_matrix, annot=True, ax = ax)

print(correlation_matrix)


# %%
Data2 = Data.drop(columns=['Country','Group','AGR',"MIN"]).copy()

Data2=(Data2-Data2.mean())/Data2.std()

# %%
Data2

# %%
# https://stackoverflow.com/questions/41540751/sklearn-kmeans-equivalent-of-elbow-method

Ks = range(1, 10)
inertia = [KMeans(i).fit(Data2).inertia_ for i in Ks]

fig = plt.figure()
plt.plot(Ks, inertia, '-bo')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia (within-cluster sum of squares)')
plt.show()

# %%
# Silhouette Analysis
range_n_clusters=[2,3,4,5,6]
for n_clusters in range_n_clusters:
    clusterer=KMeans(n_clusters=n_clusters, random_state=1)
    cluster_labels=clusterer.fit_predict(Data2)
    silhouette_avg=silhouette_score(Data2,cluster_labels)
    print("For n_clusters=", n_clusters,
          "The average silhouette_score is :", silhouette_avg)

# %%
k =2
kmeans = KMeans(n_clusters=k, random_state=1,n_init=20)
kmeans.fit(Data2)
y2= kmeans.labels_


# %%
label2 = pd.DataFrame({'Country':Data['Country'],'Label':y2})
label2.groupby('Label').count()

# %%
k =3
kmeans = KMeans(n_clusters=k, random_state=1,n_init=50)
kmeans.fit(Data2)
y3= kmeans.labels_


# %%
label3 = pd.DataFrame({'Country':Data['Country'],'Label':y3})
label3.groupby('Label').count()

# %%
k =4
kmeans = KMeans(n_clusters=k, random_state=1,n_init=50)
kmeans.fit(Data2)
y4=kmeans.labels_


# %%
label4 = pd.DataFrame({'Country':Data['Country'],'Label':y4})
label4.groupby('Label').count()

# %%
cluster_centers  = pd.DataFrame(kmeans.cluster_centers_,columns=['MAG', 'PS', 'CON','SER', 'FIN','SPS', 'TC'],index=['center1','center2','center3','center4'])
cluster_centers

# %%
result = pd.DataFrame({'Country':Data['Country'], 'Label':y4})
with pd.option_context('display.max_rows', None, 'display.max_columns', 3):
    print(result.sort_values('Label'))



