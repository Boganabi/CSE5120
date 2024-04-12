# 1. import any required library to load dataset, open files (os), print confusion matrix and accuracy score


# 2. Create test set if you like to do the split programmatically or if you have not already split the data at this point


# 3. Load your saved model for dissadvantaged communities classification 
#that you saved in dissadvantaged_communities_classification.py via Pikcle


# 4. Make predictions on test_set created from step 2


# 5. use predictions and test_set (X_test) classifications to print the following:
#    1. confution matrix, 2. accuracy score, 3. precision, 4. recall, 5. specificity
#    You can easily find the formulae for Precision, Recall, and Specificity online.

# Get and print confusion matrix
cm = [[]]
# Below are the metrics for computing classification accuracy, precision, recall and specificity
TP = cm[0,0]
TN = cm[1,1]
FP = cm[0,1]
FN = cm[1,0]