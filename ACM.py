import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load unlabelled dataset:
data = pd.read_csv("C:/Users/vijes/Downloads/archive/Unlabelled_dataset.csv")
print("The data of the file : \n",data)
print("\n")

## Data Preparation:
# Drop the 'Unnamed: 4' column
data = data.drop(columns=['Unnamed: 4'])
# Rename columns to Feature_1, Feature_2, Feature_3 and Feature_4 :
data.columns = ['Feature_1', 'Feature_2', 'Feature_3', 'Feature_4']
print("The data of the file after Data preparation : \n",data)
print("\n")

# Number of null values :
a1 = data.isnull().sum()
print("The number of null values :\n",a1)
print("\n")

# Total Number of null values :
a2 = data.isnull().sum().sum()
print("Total number of null values :\n",a2)
print("\n")

# Filling the null values as mean of the column :
data = data.fillna(value = data['Feature_1'].mean())
data = data.fillna(value = data['Feature_2'].mean())
data = data.fillna(value = data['Feature_3'].mean())
data = data.fillna(value = data['Feature_4'].mean())

# After filling the null values, Number of null values :
a3 = data.isnull().sum()
print("After filling the null values, Number of null values :\n",a3)
print("\n")
print("After Handling the missing data :\n",data)
print("\n")

# Scale the features
# We need to normalize or standardize this dataset.This is important for algorithms like K-Means.
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)
print("After Scaling the data , the data will be :\n",scaled_data)
print("\n")

# Reduce dimensions by using PCA:
pca = PCA(n_components=2)
data_pca = pca.fit_transform(scaled_data)
print("After reducing the dimensions of data by using PCA method , the data will be :\n",data_pca)
print("\n")

## Data modeling :
# By using Elbow Method for clustering :
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(scaled_data)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title("Elbow Method")
plt.xlabel("Number of clusters")
plt.ylabel("WCSS")
plt.show()


# Define the model for K-Means:
kmeans = KMeans(n_clusters = 4) # I selected 4 clusters based on the Elbow Method.
# Fit the model to the scaled data:
kmeans.fit(scaled_data)
# Get cluster labels
labels = kmeans.labels_


# Plotting clusters:
plt.scatter(scaled_data[:, 0], scaled_data[:, 1], c=labels)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('K-Means Clustering Results')
plt.show()


# Plotting clusters:
plt.scatter(scaled_data[:, 2], scaled_data[:, 3], c=labels)
plt.xlabel('Feature 3')
plt.ylabel('Feature 4')
plt.title('K-Means Clustering Results')
plt.show()


# Plotting clusters:
plt.scatter(scaled_data[:, 2], scaled_data[:, 3], c=labels)
plt.xlabel('Feature 3')
plt.ylabel('Feature 4')
plt.title('K-Mea Clustering Results')
plt.show()


# Plotting clusters:
plt.scatter(data_pca[:, 0], data_pca[:, 1], c=labels)
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('Visualizing Clusters with PCA')
plt.show()


from sklearn.metrics import silhouette_score

score = silhouette_score(scaled_data, labels)
print("Silhouette Score : ",score)


#Results :
""" 
Data Overview :
* The dataset contains several numerical features, likely representing various attributes of a biological or environmental dataset .
* The data seems like petal length, petal width, sepal length, and sepal width .
*Feature Ranges: Each feature likely exhibits a range of values, with some features showing greater variability than others. For instance, petal length might have a wider range compared to petal width.
*Central Tendencies: The means and medians of each feature can indicate typical values within the dataset. For example, if most petal widths cluster around a specific value, this could suggest a common trait among the samples.

Choosing a Clustering Method :
* K-means Clustering: This method partitions the data into K clusters by minimizing the variance within each cluster. It is efficient for large datasets .

Determining the Number of Clusters :
* Using methods like the Elbow Method or Silhouette Score can help determine the optimal number of clusters. The Elbow Method involves plotting the explained variance against the number of clusters and identifying the "elbow" point where adding more clusters yields diminishing returns .
* Based on Elbow Method we had took 4 clusters .

Clustering Patterns :
*By using K-Means Method ;
*Each cluster can represent a group of samples with similar characteristics.
*Like Smaller measurements in all attributes. Larger measurements across the board.
*Overlap and Separation: The degree of overlap between clusters can indicate how distinct the groups are. A clear separation suggests well-defined categories, while significant overlap may imply more variability within species or measurement error.
"""