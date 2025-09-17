#!/usr/bin/env python
# coding: utf-8

# # **Machine Learning Homework: Clustering Analysis with K-Means and DBSCAN**
# 
# ---
# 
# ## **Objective**  
# Students will explore unsupervised learning through clustering. The assignment includes manual calculations for k-means clustering and Python implementation of k-means, hierarchical clustering, and DBSCAN. Students will learn how to evaluate clustering quality using silhouette scores and visualize cluster structures.  
# 

# ---
# ## **Part 1: Manual Clustering Calculations**  
# ### **Step 1: K-Means Clustering (2D Data)**  
# Given the following 2D data points:
# 
# | Point | x1 | x2 |
# |-------|----|----|
# | A     | 1  | 2  |
# | B     | 1  | 4  |
# | C     | 3  | 2  |
# | D     | 5  | 8  |
# | E     | 6  | 9  |
# 
# #### Tasks:
# - Initialize centroids at A and D.
# - Assign each point to the nearest centroid.
# - Compute new centroids.
# - Repeat assignment and update steps for 2 iterations.
# - Plot the final clusters manually.
# 

# ### **Step 2: Silhouette Score (Manual)**  
# Using the final clusters from Step 1:
# - Compute intra-cluster distance `a(i)` for point C.
# - Compute nearest-cluster distance `b(i)` for point C.
# - Calculate silhouette score `s(i) = (b(i) - a(i)) / max(a(i), b(i))`.
# - Interpret the score.
# 

# ---
# ## **Part 2: Python Implementation of Clustering Algorithms**  
# ### **Step 1: Load Dataset and Visualize**  
# - Use synthetic data from `make_blobs`.
# - Plot the raw data.
# 

# In[ ]:


from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

X, _ = make_blobs(n_samples=100, centers=3, cluster_std=1.0, random_state=42)
plt.scatter(X[:, 0], X[:, 1])
plt.title("Raw Data")
plt.show()


# ### **Step 2: K-Means Clustering**  
# - Apply `KMeans(n_clusters=3)`.
# - Plot clustered data.
# - Compute silhouette score.
# 

# In[ ]:


from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(X)
plt.scatter(X[:, 0], X[:, 1], c=labels)
plt.title("K-Means Clustering")
plt.show()

print("Silhouette Score:", silhouette_score(X, labels))


# ### **Step 3: Hierarchical Clustering**  
# - Use `AgglomerativeClustering(n_clusters=3)`.
# - Plot results.
# 

# In[ ]:


from sklearn.cluster import AgglomerativeClustering

agg = AgglomerativeClustering(n_clusters=3)
labels_agg = agg.fit_predict(X)
plt.scatter(X[:, 0], X[:, 1], c=labels_agg)
plt.title("Agglomerative Clustering")
plt.show()


# ### **Step 4: DBSCAN Clustering**  
# - Apply `DBSCAN(eps=0.8, min_samples=5)`.
# - Plot clusters and noise.
# 

# In[ ]:


from sklearn.cluster import DBSCAN

dbscan = DBSCAN(eps=0.8, min_samples=5)
labels_db = dbscan.fit_predict(X)
plt.scatter(X[:, 0], X[:, 1], c=labels_db)
plt.title("DBSCAN Clustering")
plt.show()


# ---
# ## **Submission Requirements**  
# 📌 Submit a `.ipynb` notebook with:
# - Manual clustering steps.
# - Python code and plots.
# - A short reflection comparing clustering methods and their interpretability.
# 
# ---
# *Ali Fahim*  
# *University of Tehran*  
# 
