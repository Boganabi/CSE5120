#Step 1:
# Import libraries
# In this section, you can use a search engine to look for the functions that will help you implement the following steps
import pandas as pd

from category_encoders import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from joblib import dump

#Step 2:
# Load dataset and show basic statistics
# 1. Show dataset size (dimensions)
# 2. Show what column names exist for the 49 attributes in the dataset
# 3. Show the distribution of the target class CES 4.0 Percentile Range column
# 4. Show the percentage distribution of the target class CES 4.0 Percentile Range column
df = pd.read_csv("disadvantaged_communities.csv")

print("Dataset dimensions:", df.shape)
print("Column names:", df.columns)
print("Distribution of CES 4.0 Percentile Range column:\n", df.groupby("CES 4.0 Percentile Range").size())
print("Distribution of CES 4.0 Percentile Range column as a percentage:\n", df["CES 4.0 Percentile Range"].value_counts(normalize=True) * 100)

# Step 3:
#Clean the dataset - you can eitherhandle the missing values in the dataset
# with the mean of the columns attributes or remove rows the have missing values.

# null_count = df.isnull().sum().sum()
# print('Number of null values:', null_count)
# since the above commented code shows that only ~200 values are missing, i will opt to drop null values
df.dropna(inplace=True)

# Step 4: 
#Encode the Categorical Variables - Using OrdinalEncoder from the category_encoders library to encode categorical variables as ordinal integers
enc = OrdinalEncoder() # the cols parameter, if left out, will just encode the string or categorical features
df = enc.fit_transform(df)

# Step 5: 
# Separate predictor variables from the target variable (attributes (X) and target variable (y) as we did in the class)
# Create train and test splits for model development. Use the 90% and 20% split ratio
# Use stratifying (stratify=y) to ensure class balance in train/test splits
# Name them as X_train, X_test, y_train, and y_test
# Name them as X_train, X_test, y_train, and y_test
# X_train = [] # Remove this line after implementing train test split
# X_test = [] # Remove this line after implementing train test split
y = df["CES 4.0 Percentile Range"]
X = df.loc[:, df.columns != "CES 4.0 Percentile Range"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y) # assuming 90/10 split and not 90/20 lol

# Do not do steps 6 - 8 for the Ramdom Forest Model
# Step 6:
# Standardize the features (Import StandardScaler here)
sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

# Step 7:
# Below is the code to convert X_train and X_test into data frames for the next steps
# cols = X_train.columns 
# i think the above code is meant to be this:
cols = X.columns
X_train = pd.DataFrame(X_train, columns=[cols]) # pd is the imported pandas lirary - Import pandas as pd
X_test = pd.DataFrame(X_test, columns=[cols]) # pd is the imported pandas lirary - Import pandas as pd

# Step 8 - Build and train the SVM classifier
# Train SVM with the following parameters. (use the parameters with the highest accuracy for the model)
   #    1. RBF kernel
   #    2. C=10.0 (Higher value of C means fewer outliers)
   #    3. gamma = 0.3 (Linear)
classifier = SVC(kernel='rbf', C=10.0, gamma=0.3) # assume dataset is linearly separable, gamma of 0.3 indicates linear
classifier.fit(X_train, y_train)

# Test the above developed SVC on unseen pulsar dataset samples
y_pred = classifier.predict(X_test)

# compute and print accuracy score
acc = accuracy_score(y_test, y_pred)
print("Accuracy score for SVM:", acc)

# Save your SVC model (whatever name you have given your model) as .sav to upload with your submission
# You can use the library pickle to save and load your model for this assignment
# i prefer to use the joblib library since its better built for large numpy arrays, like in scikit
dump(classifier, 'SVC.sav') 

# Optional: You can print test results of your model here if you want. Otherwise implement them in evaluation.py file
print(y_pred)

# Get and print confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
# cm = [[]]
# Below are the metrics for computing classification accuracy, precision, recall and specificity
TP = cm[0,0]
TN = cm[1,1]
FP = cm[0,1]
FN = cm[1,0]

# Compute Precision and use the following line to print it
precision = TP / (TP + FP) # 0 # Change this line to implement Precision formula
print('Precision : {0:0.3f}'.format(precision))

# Compute Recall and use the following line to print it
recall = TP / (TP + FN) # 0 # Change this line to implement Recall formula
print('Recall or Sensitivity : {0:0.3f}'.format(recall))

# Compute Specificity and use the following line to print it
specificity = TN / (TN + FP) # 0 # Change this line to implement Specificity formula
print('Specificity : {0:0.3f}'.format(specificity))


# Step 9: Build and train the Random Forest classifier
# Train Random Forest  with the following parameters.
# (n_estimators=10, random_state=0)
rfc = RandomForestClassifier(n_estimators=10, random_state=0)
rfc.fit(X_train, y_train)

# Test the above developed Random Forest model on unseen DACs dataset samples
y_pred = rfc.predict(X_test)

# compute and print accuracy score
acc = accuracy_score(y_test, y_pred)
print("Accuracy score for Random Forest:", acc)

# Save your Random Forest model (whatever name you have given your model) as .sav to upload with your submission
# You can use the library pickle to save and load your model for this assignment
dump(rfc, "RandomForest.sav")

# Optional: You can print test results of your model here if you want. Otherwise implement them in evaluation.py file
print(y_pred)

# Get and print confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Below are the metrics for computing classification accuracy, precision, recall and specificity
TP = cm[0,0]
TN = cm[1,1]
FP = cm[0,1]
FN = cm[1,0]

# Compute Classification Accuracy and use the following line to print it
classification_accuracy = (TP + TN) / (TP + TN + FP + FN)
print('Classification accuracy : {0:0.4f}'.format(classification_accuracy))

# Compute Precision and use the following line to print it
precision = TP / (TP + FP) # Change this line to implement Precision formula
print('Precision : {0:0.3f}'.format(precision))


# Compute Recall and use the following line to print it
recall = TP / (TP + FN) # Change this line to implement Recall formula
print('Recall or Sensitivity : {0:0.3f}'.format(recall))

# Compute Specificity and use the following line to print it
specificity = TN / (TN + FP) # Change this line to implement Specificity formula
print('Specificity : {0:0.3f}'.format(specificity))