# %%

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
from sklearn.preprocessing import StandardScaler
X = Data.drop(columns=['Country','Group']).copy()
Scaler = StandardScaler()
Scaler.fit(X)
X = Scaler.transform(X)

# %%
from sklearn.decomposition import PCA
PCA_0 = PCA(n_components = .80)
PCA_0.fit(X)
print(PCA_0.n_components_)


# %%
from sklearn.decomposition import PCA
PCA_1 = PCA(n_components = 2)
PCA_1.fit(X)
print(PCA_1.n_components_)
Data1_Scaled_PCA_1 = PCA_1.transform(X)
Data1_Scaled_PCA_1

# %%
Data1_Scaled_PCA_1.shape
# Plot the variances
features = range(PCA_1.n_components_)
plt.bar(features, PCA_1.explained_variance_ratio_, color='black')
plt.xlabel('PCA features')
plt.ylabel('variance %')
plt.xticks(features)
# Save components to a DataFrame
PCA_components = pd.DataFrame(Data1_Scaled_PCA_1, columns=['PCA_1','PCA_2'])
PCA_components_1 = pd.concat([PCA_components,Data['Group']],axis=1)
PCA_components_1.head()
colors = {'EU':'red', 'EFTA':'green', 'Eastern':'blue', 'Other':'yellow'}

sns.pairplot(PCA_components_1, vars = PCA_components_1.columns[:-1], hue='Group')
plt.show()
# Plot the variances
plt.style.use("ggplot")
plt.plot(PCA_1.explained_variance_, marker='o')
plt.xlabel("Eigenvalue number")
plt.ylabel("Eigenvalue size")
plt.title("Scree Plot")
plt.show()


# %%
PCA_1.explained_variance_ratio_

# %%
PCA_2 = PCA(n_components = 3)
PCA_2.fit(X)
Data1_Scaled_PCA_2 = PCA_2.transform(X)

# %%
# Plot the variances
plt.style.use("ggplot")
plt.plot(PCA_2.explained_variance_, marker='o')
plt.xlabel("Eigenvalue number")
plt.ylabel("Eigenvalue size")
plt.title("Scree Plot")
plt.show()

# Plot the variances Ratio
features = range(PCA_2.n_components_)
plt.bar(features, (PCA_2.explained_variance_ratio_)*100, color='black')
plt.xlabel('PCA features')
plt.ylabel('variance %')
plt.xticks(features)
plt.show()

PCA_components = pd.DataFrame(Data1_Scaled_PCA_2, columns=['PCA_1','PCA_2', 'PCA_3'])
PCA_components_1 = pd.concat([PCA_components,Data['Group']],axis=1)
PCA_components_1.head()
sns.pairplot(PCA_components_1, vars = PCA_components_1.columns[:-1], hue='Group')
plt.show()

# %%
sum(PCA_2.explained_variance_ratio_)

# %%



