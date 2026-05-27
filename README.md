# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Start

2.Import required libraries: pandas, numpy, sklearn (for preprocessing, splitting, model, and evaluation).

3.Load dataset: Read Placement_Data.csv using pandas.read_csv().

4.Data preprocessing: a.Drop irrelevant columns: sl_no (serial number), salary (since placement is predicted before salary). b.Encode categorical variables (gender, ssc_b, hsc_b, hsc_s, degree_t, workex, specialisation, status) into numerical values using LabelEncoder.

5.Define features and target: a.Features X = all columns except status. b.Target y = status column (0 = Not Placed, 1 = Placed).

6.Split dataset: Divide into training and testing sets using train_test_split with 80% training and 20% testing.

7.Standardize features: Apply StandardScaler to scale the numeric values for better model convergence.

8.Build Logistic Regression model: a.Initialize Logistic Regression with max_iter=200. b.Train (fit) the model using training data (X_train, y_train).

9.Predict placement status: Use the trained model to predict on X_test.

10.Evaluate performance:
## Program:

Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

Developed by: Ajay Karthick M

RegisterNumber:  212225040014
```
import pandas as pd
import numpy as np
df = pd.read_csv('Placement_Data.csv')
df
df1 = df.copy()
df1
df1 = df1.drop(['sl_no', 'salary'], axis=1)
df1.isnull().sum()
df1.duplicated().sum()
df1
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df1['gender'] = le.fit_transform(df1['gender'])
df1['ssc_b'] = le.fit_transform(df1['ssc_b'])
df1['hsc_b'] = le.fit_transform(df1['hsc_b'])
df1['hsc_s'] = le.fit_transform(df1['hsc_s'])
df1['degree_t'] = le.fit_transform(df1['degree_t'])
df1['workex'] = le.fit_transform(df1['workex'])
df1['specialisation'] = le.fit_transform(df1['specialisation'])
df1['status'] = le.fit_transform(df1['status'])
df1
x = df1.iloc[:, :-1]
y = df1['status']
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(solver="liblinear")
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
accuracy = accuracy_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)
cr = classification_report(y_test, y_pred)
print("Accuracy Score:", accuracy)
print("\nConfusion Matrix:\n", confusion)
print("\nClassification Report:\n", cr)
from sklearn import metrics
cn_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion, display_labels=['true', 'false'])
cn_display.plot()
```

## Output:

<img width="734" height="449" alt="592213857-1958d288-2088-47e0-94b3-c82e39986167" src="https://github.com/user-attachments/assets/ca894043-63c2-4700-9cfa-6fc82b27bcdf" />

<img width="790" height="595" alt="592213934-7368452c-6ed8-4bc2-8efd-0f5aba11072f" src="https://github.com/user-attachments/assets/462face7-3df3-46fd-ad22-0ee8effe5f43" />


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
