#!/usr/bin/env python
# coding: utf-8

# # **Machine Learning Homework: Boosted Trees for Feature Generation in SVM/LDA**
# 
# ---
# 
# ## **Objective**  
# Students should implement a **classification pipeline** using the **Breast Cancer Wisconsin (Diagnostic) dataset**. The task involves **data preprocessing, feature engineering using Gradient Boosting, and classification using SVM and LDA**.  
# 

# ---
# 
# ## **Part 1: Understanding the Dataset**
# ### **Step 1: Download the Dataset**  
# Students should download the dataset from the **UCI Machine Learning Repository**:  
# 🔗 [Breast Cancer Wisconsin (Diagnostic) Dataset](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)  
# 
# ### **Step 2: Dataset Overview**  
# - **569 instances** with **30 numerical features**.
# - Features extracted from **digitized images of fine needle aspirates (FNA) of breast masses**.
# 
# ### **Step 3: Target Variable**  
# The dataset’s target variable is **Diagnosis**, which has two possible classes:  
# - **M (Malignant)** → Cancerous tumors.  
# - **B (Benign)** → Non-cancerous tumors.  
# 
# ### **Step 4: Feature Analysis**  
# Each sample has **30 numerical features**, grouped as follows:  
# - **First 10 features:** Mean values of cell properties.  
# - **Second 10 features:** Standard deviation (std) values of the same properties.  
# - **Final 10 features:** Worst (largest) values recorded.  
# 
# These feature groups help assess **average properties (mean), variability within samples (std), and extreme cases (worst values)**, which can be crucial for distinguishing between benign and malignant tumors.  
# 
# ### **Step 5: Exploratory Data Analysis**  
# Students must:  
# ✅ **Visualize feature distributions** (histograms, box plots).  
# ✅ **Check correlations** between features.  
# ✅ **Identify potential outliers**.  
# 

# In[ ]:





# ---
# ## **Part 2: Data Preprocessing**  
# Students will:  
# ✅ **Handle missing values** (if any).  
# ✅ **Standardize numerical features** using normalization.  
# ✅ **Split the dataset** into **training, validation, and testing sets** (e.g., 70-15-15 split).  
# 

# In[ ]:





# ---
# 
# ## **Part 3: Generate Features Using Boosted Trees**  
# Students will:  
# ✅ **Train a Gradient Boosting classifier** using Scikit-Learn (`GradientBoostingClassifier`).  
# ✅ **Extract leaf indices** from the trained trees.  
# ✅ **Convert leaf indices into numerical features**, creating a new feature set.  
# 
# Hints:  
# ```python
# from sklearn.ensemble import GradientBoostingClassifier
# 
# gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1)
# gb_model.fit(X_train, y_train)
# new_features = gb_model.apply(X_train)  # Extract leaf indices
# ```

# In[ ]:





# ---
# 
# ## **Part 4: Train SVM and LDA Using the Boosted Features**  
# Students will:  
# ✅ **Train a Support Vector Machine (SVM)** model with different kernels (`linear`, `polynomial`, `RBF`).  
# ✅ **Train a Linear Discriminant Analysis (LDA) classifier**.  
# ✅ **Compare classification performance** using evaluation metrics.  
# 
# Hints:  
# ```python
# from sklearn.svm import SVC
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# 
# svm_model = SVC(kernel='rbf', C=1.0)
# svm_model.fit(new_features, y_train)
# 
# lda_model = LinearDiscriminantAnalysis()
# lda_model.fit(new_features, y_train)
# ```

# In[ ]:





# ---
# 
# ## **Part 5: Optimize the Learning Process**  
# Students will:  
# ✅ **Perform Feature Selection and Extraction** (e.g., PCA, Recursive Feature Elimination).  
# ✅ **Implement Regularization techniques (L1 or L2)** to improve generalization.  
# ✅ **Apply Hyperparameter tuning** using **Grid Search or Randomized Search**.  
# ✅ **Use Cross-Validation and Learning Curves** to debug models.  
# 
# ### **Applying Regularization in LDA and SVM using Scikit-Learn**  
# Regularization helps control model complexity and prevent overfitting. In **LDA**, apply **L2 regularization** by using the `shrinkage` parameter in `LinearDiscriminantAnalysis(shrinkage='auto')`, which stabilizes covariance estimates when sample sizes are small. For **SVM**, control regularization with the `C` parameter in `SVC(C=1.0, kernel='rbf')`. **Lower `C` increases regularization**, encouraging a simpler decision boundary, while **higher `C` reduces regularization**, fitting data more tightly. In **linear SVM**, explicitly set `penalty='l1'` or `penalty='l2'` in `LinearSVC()`.  
# 
# Hints:
# ```python
# from sklearn.model_selection import GridSearchCV
# 
# param_grid = {'C': [0.1, 1, 10]}
# grid_search = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=5)
# grid_search.fit(new_features, y_train)
# ```

# In[ ]:





# ---
# 
# ## **Submission Requirements**  
# Students must submit:  
# 📌 **A Python notebook (.ipynb)** containing:  
#    - All preprocessing steps.  
#    - Model training and evaluation.  
#    - Optimized models with explanations.  
# 
# 📌 **A short report summarizing**:  
#    - Key challenges and solutions.  
#    - Performance comparison between SVM and LDA.  
# 
# 
# ---
# 
# ## **Important Note on AI Assistance**
# 
# You are allowed to use AI tools (e.g., chatbots, code assistants) as a part of your investigation to help with coding, debugging, or brainstorming ideas. **However, your final analysis and the accompanying written report must be entirely your own work.** The report should clearly reflect your personal insights and understanding of the classifier behaviors, not just the output produced by AI.
# 
# ---
# 
# Best wishes to all students in completing this assignment! I hope this structured exercise **enhances your understanding of ensemble learning, classification models and feature engineering** while improving your practical programming skills.  
# 
# *Ali Fahim*  
# *University of Tehran*  
# 
