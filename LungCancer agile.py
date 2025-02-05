#!/usr/bin/env python
# coding: utf-8

# In[2]:


# decision tree
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv(r"C:\Users\Dell\Desktop\Agile lab\lungcancer.csv")


df["GENDER"] = df["GENDER"].map({"M": 0, "F": 1})
df["LUNG_CANCER"] = df["LUNG_CANCER"].map({"NO": 0, "YES": 1})


x = df.drop("LUNG_CANCER", axis=1)
y = df["LUNG_CANCER"]


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


model = DecisionTreeClassifier(random_state=42)
model.fit(x_train, y_train)


y_pred = model.predict(x_test)
print(y_pred)


accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# In[4]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv(r"C:\Users\Dell\Desktop\Agile lab\lungcancer.csv")

# Encode categorical variables
df["GENDER"] = df["GENDER"].map({"M": 0, "F": 1})
df["LUNG_CANCER"] = df["LUNG_CANCER"].map({"NO": 0, "YES": 1})


x = df.drop("LUNG_CANCER", axis=1)
y = df["LUNG_CANCER"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


model = GaussianNB()
model.fit(x_train, y_train)


y_pred = model.predict(x_test)
print(y_pred)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# In[6]:


# random forest

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv(r"C:\Users\Dell\Desktop\Agile lab\lungcancer.csv")

# Encode categorical variables
df["GENDER"] = df["GENDER"].map({"M": 0, "F": 1})
df["LUNG_CANCER"] = df["LUNG_CANCER"].map({"NO": 0, "YES": 1})

# Split features and target
x = df.drop("LUNG_CANCER", axis=1)
y = df["LUNG_CANCER"]

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


# In[7]:


# svm
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns



df = pd.read_csv(r"C:\Users\Dell\Desktop\Agile lab\lungcancer.csv")

# Encode categorical variables
df["GENDER"] = df["GENDER"].map({"M": 0, "F": 1})
df["LUNG_CANCER"] = df["LUNG_CANCER"].map({"NO": 0, "YES": 1})

# Split features and target
x = df.drop("LUNG_CANCER", axis=1)
y = df["LUNG_CANCER"]

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Train SVM model
model = SVC(random_state=42)
model.fit(x_train, y_train)

# Make predictions
y_pred = model.predict(x_test)
print(y_pred)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# In[8]:


# KNN

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv(r"C:\Users\Dell\Desktop\Agile lab\lungcancer.csv")


df["GENDER"] = df["GENDER"].map({"M": 0, "F": 1})
df["LUNG_CANCER"] = df["LUNG_CANCER"].map({"NO": 0, "YES": 1})


x = df.drop("LUNG_CANCER", axis=1)
y = df["LUNG_CANCER"]


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


model = KNeighborsClassifier()
model.fit(x_train, y_train)


y_pred = model.predict(x_test)
print(y_pred)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# In[10]:


# logistics
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv(r"C:\Users\Dell\Desktop\Agile lab\lungcancer.csv")

# Encode categorical variables
df["GENDER"] = df["GENDER"].map({"M": 0, "F": 1})
df["LUNG_CANCER"] = df["LUNG_CANCER"].map({"NO": 0, "YES": 1})

# Split features and target
x = df.drop("LUNG_CANCER", axis=1)
y = df["LUNG_CANCER"]

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Train Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(x_train, y_train)

# Make predictions
y_pred = model.predict(x_test)
print(y_pred)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# In[13]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv(r"C:\Users\Dell\Desktop\Agile lab\lungcancer.csv")

# Encode categorical variables
df["GENDER"] = df["GENDER"].map({"M": 0, "F": 1})
df["LUNG_CANCER"] = df["LUNG_CANCER"].map({"NO": 0, "YES": 1})

# Split features and target
x = df.drop("LUNG_CANCER", axis=1)
y = df["LUNG_CANCER"]

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Train Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(x_train, y_train)

# Make predictions
y_pred = model.predict(x_test)
print(y_pred)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Accuracy comparison plot
models = ["Random Forest", "KNN", "Decision Tree", "Naïve Bayes", "Logistic Regression", "SVM"]
accuracies = [0.9677, 0.951, 0.9677, 0.951, accuracy, 0.9677]
colors = ['green', 'blue', 'orange', 'red', 'pink', 'yellow']

plt.figure(figsize=(8, 5))
bars = plt.bar(models, accuracies, color=colors)

for bar, acc in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 0.02, f"{acc:.4f}", 
             ha='center', color='white', fontweight='bold')

plt.xlabel("Machine Learning Models")
plt.ylabel("Accuracy Score")
plt.title("Accuracy Comparison of Different ML Models (Test Size = 0.201)")
plt.ylim(0.90, 1.0) 
plt.grid(axis='y', linestyle="--", alpha=0.7)

plt.show()


# In[ ]:




