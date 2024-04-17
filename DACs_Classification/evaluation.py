
# 1. import any required library to load dataset, open files (os), print confusion matrix and accuracy score
import pandas as pd

from category_encoders import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from joblib import load
from sklearn.metrics import confusion_matrix

# 2. Create test set if you like to do the split programmatically or if you have not already split the data at this point
df = pd.read_csv("disadvantaged_communities.csv")
df.dropna(inplace=True)
enc = OrdinalEncoder() # the cols parameter, if left out, will just encode the string or categorical features
df = enc.fit_transform(df)
y = df["CES 4.0 Percentile Range"]
X = df.loc[:, df.columns != "CES 4.0 Percentile Range"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y) # assuming 80/20 split and not 90/20 lol

sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

# 3. Load your saved model for dissadvantaged communities classification 
#that you saved in dissadvantaged_communities_classification.py via Pikcle
svc = load("SVC.pkl")
rf = load("RandomForest.pkl")

# 4. Make predictions on test_set created from step 2
svc_pred = svc.predict(X_test)
rf_pred = rf.predict(X_test)

# 5. use predictions and test_set (X_test) classifications to print the following:
#    1. confution matrix, 2. accuracy score, 3. precision, 4. recall, 5. specificity
#    You can easily find the formulae for Precision, Recall, and Specificity online.

# Get and print confusion matrix
cm = confusion_matrix(y_test, svc_pred)
# Below are the metrics for computing classification accuracy, precision, recall and specificity
TP = cm[0,0]
TN = cm[1,1]
FP = cm[0,1]
FN = cm[1,0]

print("Statistics for SVC:")
print("Confusion matrix:\n", cm)

# Compute Precision and use the following line to print it
precision = TP / (TP + FP) # 0 # Change this line to implement Precision formula
print('Precision : {0:0.3f}'.format(precision))

# Compute Recall and use the following line to print it
recall = TP / (TP + FN) # 0 # Change this line to implement Recall formula
print('Recall or Sensitivity : {0:0.3f}'.format(recall))

# Compute Specificity and use the following line to print it
specificity = TN / (TN + FP) # 0 # Change this line to implement Specificity formula
print('Specificity : {0:0.3f}'.format(specificity))


cm = confusion_matrix(y_test, rf_pred)
# Below are the metrics for computing classification accuracy, precision, recall and specificity
TP = cm[0,0]
TN = cm[1,1]
FP = cm[0,1]
FN = cm[1,0]

print("Statistics for RandomForst:")
print("Confusion matrix:\n", cm)

# Compute Precision and use the following line to print it
precision = TP / (TP + FP) # 0 # Change this line to implement Precision formula
print('Precision : {0:0.3f}'.format(precision))

# Compute Recall and use the following line to print it
recall = TP / (TP + FN) # 0 # Change this line to implement Recall formula
print('Recall or Sensitivity : {0:0.3f}'.format(recall))

# Compute Specificity and use the following line to print it
specificity = TN / (TN + FP) # 0 # Change this line to implement Specificity formula
print('Specificity : {0:0.3f}'.format(specificity))