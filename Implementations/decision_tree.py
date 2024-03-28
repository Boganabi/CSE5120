
# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

# load dataset
dataset = pd.read_csv("Social_Network_Ads.csv") 
# professor has a different dataset that does not have user_id and gender but the below if statement will make them the same
X = dataset.iloc[:, :-1] # get everything except for last column
y = dataset.iloc[:, -1] # get everything in the last column

useSameData = True
if useSameData:
    X = dataset.iloc[:, 2:4] # gets columns 3 and 4

# train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# standardization
sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

# model development
classifier = DecisionTreeClassifier(criterion='entropy', random_state=0) # assume dataset is linearly separable
classifier.fit(X_train, y_train)

# test
print(classifier.predict(sc.transform([[30,3700]])))

# accuracy
y_pred = classifier.predict(X_test)

# confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

print(accuracy_score(y_test, y_pred))