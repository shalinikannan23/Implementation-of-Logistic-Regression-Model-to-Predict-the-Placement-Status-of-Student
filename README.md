# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.  Import the standard libraries.

2. Upload the dataset and check for any null or duplicated values using .isnull() and .duplicated() function respectively.
3. Import LabelEncoder and encode the dataset.

4. Import LogisticRegression from sklearn and apply the model on the dataset.

5. Predict the values of array.

6. Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.
7. Apply new unknown values

## Program:
```
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
# Developed by: SHALINI.K
# RegisterNumber:  212222240095
# Import Library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#Read The File
dataset=pd.read_csv('Placement_Data_Full_Class.csv')
dataset
dataset.head(10)
dataset.tail(10)
# Dropping the serial number and salary column
dataset=dataset.drop(['sl_no','ssc_p','workex','ssc_b'],axis=1)
dataset
dataset.shape
dataset.info()
dataset["gender"]=dataset["gender"].astype('category')
dataset["hsc_b"]=dataset["hsc_b"].astype('category')
dataset["hsc_s"]=dataset["hsc_s"].astype('category')
dataset["degree_t"]=dataset["degree_t"].astype('category')
dataset["specialisation"]=dataset["specialisation"].astype('category')
dataset["status"]=dataset["status"].astype('category')
dataset.info()
dataset["gender"]=dataset["gender"].cat.codes
dataset["hsc_b"]=dataset["hsc_b"].cat.codes
dataset["hsc_s"]=dataset["hsc_s"].cat.codes
dataset["degree_t"]=dataset["degree_t"].cat.codes
dataset["specialisation"]=dataset["specialisation"].cat.codes
dataset["status"]=dataset["status"].cat.codes
dataset.info()
dataset
# selecting the features and labels
x=dataset.iloc[:, :-1].values
y=dataset.iloc[: ,-1].values
y
# dividing the data into train and test
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
dataset.head()
y_train.shape
x_train.shape
# Creating a Classifier using Sklearn
from sklearn.linear_model import LogisticRegression
clf=LogisticRegression(random_state=0,solver='lbfgs',max_iter=1000).fit(x_train,y_train)
# Printing the acc
clf=LogisticRegression()
clf.fit(x_train,y_train)
clf.score(x_test,y_test)
# Predicting for random value
clf.predict([[1	,78.33,	1,	2,	77.48,	2,	86.5,	0,	66.28]])
```

## Output:
### Read CSV File:
![1](https://github.com/shalinikannan23/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118656529/06f03811-8b27-46a9-9ee1-e92a82526e4e)
### To read 1st ten Data(Head):
![2](https://github.com/shalinikannan23/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118656529/15a2d9d0-9f53-4136-9160-c97ce7d8698d)
### To read last ten Data(Tail):
![3](https://github.com/shalinikannan23/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118656529/085ec7ef-63fd-40f1-8113-d945026096e5)
### Dropping the serial number and salary column:
![4](https://github.com/shalinikannan23/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118656529/209fe561-064f-48bd-90e4-d0cfdfb29aef)
### Dataset Information:
![6](https://github.com/shalinikannan23/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118656529/580bcfbd-afc0-4bb1-a44d-d4623687e515)
### Dataset after changing object into category:
![7](https://github.com/shalinikannan23/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118656529/4185c6fe-016b-4630-a896-561257b97234)
### Dataset after changing category into integer:
![8](https://github.com/shalinikannan23/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118656529/eef76c9d-70a2-4dad-838f-8cfd5c3e6b8d)
### Displaying the Dataset:
![9](https://github.com/shalinikannan23/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118656529/52d99f9d-85c0-4cce-91b2-3ac0f7dc8e81)
### Selecting the features and labels:
![10](https://github.com/shalinikannan23/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118656529/5961a94b-f854-4486-9649-5f7a46d93529)
### Dividing the data into train and test:
![11](https://github.com/shalinikannan23/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118656529/b2c41d22-80eb-4bea-9f01-f35bcd05aad1)
### Creating a Classifier using Sklearn:
![13](https://github.com/shalinikannan23/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118656529/59ef639a-219d-45d4-8c97-dae07b9c5b80)
### Predicting for random value:
![14](https://github.com/shalinikannan23/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118656529/8411a5e1-3606-48c8-bc3c-2bbbc08f6cc6)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
