#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv(r"C:\Users\Dell\Desktop\Agile lab\heart.csv")

# Split features and target
x = df.drop("target", axis=1)
y = df["target"]

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Train Decision Tree model
model = DecisionTreeClassifier(random_state=42)
model.fit(x_train, y_train)

# Make predictions
y_pred = model.predict(x_test)
print(y_pred)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Accuracy comparison plot
models = ["Decision Tree"]
accuracies = [accuracy]
colors = ['orange']

plt.figure(figsize=(8, 5))
bars = plt.bar(models, accuracies, color=colors)

for bar, acc in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 0.02, f"{acc:.4f}", 
             ha='center', color='white', fontweight='bold')

plt.xlabel("Machine Learning Models")
plt.ylabel("Accuracy Score")
plt.title("Accuracy of Decision Tree Model (Test Size = 0.2)")
plt.ylim(0.50, 1.0) 
plt.grid(axis='y', linestyle="--", alpha=0.7)

plt.show()


# In[3]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv(r"C:\Users\Dell\Desktop\Agile lab\heart.csv")

# Split features and target
x = df.drop("target", axis=1)
y = df["target"]

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Train Gaussian Naïve Bayes model
model = GaussianNB()
model.fit(x_train, y_train)

# Make predictions
y_pred = model.predict(x_test)
print(y_pred)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Accuracy comparison plot
models = ["Gaussian Naïve Bayes"]
accuracies = [accuracy]
colors = ['blue']

plt.figure(figsize=(8, 5))
bars = plt.bar(models, accuracies, color=colors)

for bar, acc in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 0.02, f"{acc:.4f}", 
             ha='center', color='white', fontweight='bold')

plt.xlabel("Machine Learning Models")
plt.ylabel("Accuracy Score")
plt.title("Accuracy of Gaussian Naïve Bayes Model (Test Size = 0.2)")
plt.ylim(0.50, 1.0) 
plt.grid(axis='y', linestyle="--", alpha=0.7)

plt.show()


# In[4]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv(r"C:\Users\Dell\Desktop\Agile lab\heart.csv")

# Split features and target
x = df.drop("target", axis=1)
y = df["target"]

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Train Random Forest model
model = RandomForestClassifier(random_state=42)
model.fit(x_train, y_train)

# Make predictions
y_pred = model.predict(x_test)
print(y_pred)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Accuracy comparison plot
models = ["Random Forest"]
accuracies = [accuracy]
colors = ['green']

plt.figure(figsize=(8, 5))
bars = plt.bar(models, accuracies, color=colors)

for bar, acc in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 0.02, f"{acc:.4f}", 
             ha='center', color='white', fontweight='bold')

plt.xlabel("Machine Learning Models")
plt.ylabel("Accuracy Score")
plt.title("Accuracy of Random Forest Model (Test Size = 0.2)")
plt.ylim(0.50, 1.0) 
plt.grid(axis='y', linestyle="--", alpha=0.7)

plt.show()


# In[5]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv(r"C:\Users\Dell\Desktop\Agile lab\heart.csv")

# Split features and target
x = df.drop("target", axis=1)
y = df["target"]

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Train K-Nearest Neighbors model
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(x_train, y_train)
knn_pred = knn_model.predict(x_test)
knn_accuracy = accuracy_score(y_test, knn_pred)

# Train Support Vector Machine model
svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(x_train, y_train)
svm_pred = svm_model.predict(x_test)
svm_accuracy = accuracy_score(y_test, svm_pred)

# Accuracy comparison plot
models = ["KNN", "SVM"]
accuracies = [knn_accuracy, svm_accuracy]
colors = ['blue', 'red']

plt.figure(figsize=(8, 5))
bars = plt.bar(models, accuracies, color=colors)

for bar, acc in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 0.02, f"{acc:.4f}", 
             ha='center', color='white', fontweight='bold')

plt.xlabel("Machine Learning Models")
plt.ylabel("Accuracy Score")
plt.title("Accuracy of KNN and SVM Models (Test Size = 0.2)")
plt.ylim(0.50, 1.0) 
plt.grid(axis='y', linestyle="--", alpha=0.7)

plt.show()


# In[6]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv(r"C:\Users\Dell\Desktop\Agile lab\heart.csv")

# Split features and target
x = df.drop("target", axis=1)
y = df["target"]

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Train K-Nearest Neighbors model
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(x_train, y_train)
knn_pred = knn_model.predict(x_test)
knn_accuracy = accuracy_score(y_test, knn_pred)

# Train Support Vector Machine model
svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(x_train, y_train)
svm_pred = svm_model.predict(x_test)
svm_accuracy = accuracy_score(y_test, svm_pred)

# Train Logistic Regression model
log_model = LogisticRegression(random_state=42)
log_model.fit(x_train, y_train)
log_pred = log_model.predict(x_test)
log_accuracy = accuracy_score(y_test, log_pred)

# Accuracy comparison plot
models = ["KNN", "SVM", "Logistic Regression"]
accuracies = [knn_accuracy, svm_accuracy, log_accuracy]
colors = ['blue', 'red', 'green']

plt.figure(figsize=(8, 5))
bars = plt.bar(models, accuracies, color=colors)

for bar, acc in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 0.02, f"{acc:.4f}", 
             ha='center', color='white', fontweight='bold')

plt.xlabel("Machine Learning Models")
plt.ylabel("Accuracy Score")
plt.title("Accuracy of KNN, SVM, and Logistic Regression Models (Test Size = 0.2)")
plt.ylim(0.50, 1.0) 
plt.grid(axis='y', linestyle="--", alpha=0.7)

plt.show()


# In[ ]:




