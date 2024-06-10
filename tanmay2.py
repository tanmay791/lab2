#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
columns = ['ID', 'Diagnosis'] + [f'feature_{i}' for i in range(1, 31)]
data = pd.read_csv(url, header=None, names=columns)

# Display the first few rows of the dataset
print("First few rows of the dataset:")
print(data.head())

# Prepare the data
X = data.drop(['ID', 'Diagnosis'], axis=1)
y = data['Diagnosis'].map({'M': 1, 'B': 0})  # Convert diagnosis to binary

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Decision Tree Classifier model
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)

# Evaluate the Decision Tree Classifier model
dt_y_pred = dt_model.predict(X_test)
dt_accuracy = accuracy_score(y_test, dt_y_pred)
dt_cm = confusion_matrix(y_test, dt_y_pred)

print("\nDecision Tree Classifier Model")
print(f'Decision Tree Accuracy: {dt_accuracy}')
print('Confusion Matrix:')
print(dt_cm)

# Visualize the confusion matrix for Decision Tree Classifier
plt.figure(figsize=(6,4))
sns.heatmap(dt_cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - Decision Tree Classifier')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


# In[ ]:




