# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## AlgorithmImport the required libraries.

1.Upload and read the dataset.
2.Check for any null values using the isnull() function.
3.From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.
4.Find the accuracy of the model and predict the required values by importing the required module from sklearn.

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Rishi chandran R
RegisterNumber: 212223043005
*/
```
```
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
data = pd.read_csv("Employee.csv")
print(data.head())
print(data.info())
print(data.isnull().sum())
print(data["left"].value_counts())
le = LabelEncoder()
data["salary"] = le.fit_transform(data["salary"])
x = data[["satisfaction_level", "last_evaluation", "number_project", 
          "average_montly_hours", "time_spend_company", "Work_accident", 
          "promotion_last_5years", "salary"]]
y = data["left"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=100)
dt = DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train, y_train)
y_pred = dt.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy of the Decision Tree Model:", accuracy)
input_data = pd.DataFrame([[0.5, 0.8, 9, 260, 6, 0, 1, 2]], columns=x.columns)
predicted_class = dt.predict(input_data)
print("Prediction for input data:", predicted_class)
plt.figure(figsize=(12, 8))
plot_tree(dt, feature_names=x.columns, class_names=["Stayed", "Left"], filled=True)
plt.title("Decision Tree - Employee Churn Prediction")
```

## Output:
![Screenshot 2025-04-24 142943](https://github.com/user-attachments/assets/9dbf69e5-7e9e-4388-9d2c-d8a41867ef29)

![Screenshot 2025-04-24 143002](https://github.com/user-attachments/assets/ba64792a-444c-4f78-ae32-6f87431a4586)

![Screenshot 2025-04-24 143002](https://github.com/user-attachments/assets/c5bf714f-1370-43dc-af66-8e4128f85d82)


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
