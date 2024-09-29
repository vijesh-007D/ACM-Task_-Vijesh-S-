Results :
Data Overview :
* The dataset contains several numerical features, likely representing various attributes of a biological or environmental dataset .
* The data seems like petal length, petal width, sepal length, and sepal width .
*Feature Ranges: Each feature likely exhibits a range of values, with some features showing greater variability than others. For instance, petal length might have a 
 wider range compared to petal width.
*Central Tendencies: The means and medians of each feature can indicate typical values within the dataset. For example, if most petal widths cluster around a 
 specific value, this could suggest a common trait among the samples.

Choosing a Clustering Method :
* K-means Clustering: This method partitions the data into K clusters by minimizing the variance within each cluster. It is efficient for large datasets .

Determining the Number of Clusters :
* Using methods like the Elbow Method or Silhouette Score can help determine the optimal number of clusters. The Elbow Method involves plotting the explained 
 variance against the number of clusters and identifying the "elbow" point where adding more clusters yields diminishing returns .
* Based on Elbow Method we had took 4 clusters .

Clustering Patterns :
*By using K-Means Method ;
*Each cluster can represent a group of samples with similar characteristics.
*Like Smaller measurements in all attributes. Larger measurements across the board.
*Overlap and Separation: The degree of overlap between clusters can indicate how distinct the groups are. A clear separation suggests well-defined categories, while 
 significant overlap may imply more variability within species or measurement error.
