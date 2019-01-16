from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.model_selection import train_test_split 
from sklearn import linear_model 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, r2_score, cohen_kappa_score
from sklearn.naive_bayes import GaussianNB
import pandas as pd
import matplotlib.pyplot as plt
import graphviz
import csv

#Import the dataset 
dataset = pd.read_csv("CE802_Ass_2018_Data.csv")

#TEST SET CSV FILE
testset = pd.read_csv("CE802_Ass_2018_Test.csv")
newfeatures = testset.drop('Class', axis=1)

#Split the data into features (F1-F14) and labels (TRUE or FALSE)
features = dataset.drop('Class', axis=1) 
labels = dataset['Class'] 

#Divide the data into training sets and test sets
#Assign 20% of the data for test and 80% for training
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.20)  

#Train the Decision Tree model and set max_leaf_nodes to 10 as a form of pre-pruning
dtclassifier = DecisionTreeClassifier(max_leaf_nodes = 10)  
dtclassifier.fit(train_features, train_labels)  

#Make predictions on the new test dataset -> ex.3
newlabels = dtclassifier.predict(newfeatures)

#Write the new predicted labels in a CSV file -> ex.3
with open('New_Labels.csv', mode='w') as csv_file:
    fieldnames = ['Class']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames, lineterminator = '\n')

    for i in range(len(newlabels)):
        writer.writerow({'Class': newlabels[i]})

#Make predictions on the test data
label_predct_1 = dtclassifier.predict(test_features)  

print("-----------------------DECISION TREE CLASSIFIER-------------------------------")
print("####  CONFUSION MATRIX  ####")
print(confusion_matrix(test_labels, label_predct_1), "\n")  

print("######################################################")
print(classification_report(test_labels, label_predct_1))  
print("######################################################")

dtaccuracy =  dtclassifier.score(test_features,test_labels)*100
print("The prediction accuracy of the Decision Tree Classifier is: %.2f" %dtaccuracy,"%")

kappa = cohen_kappa_score(test_labels, label_predct_1) * 100
print("Kappa statistics is %.2f" %kappa, "%")

print("-----------------------END OF DECISION TREE CLASSIFIER-------------------------------", "\n")

#####################################################################################################

print("-----------------------KNN CLASSIFIER-------------------------------")
# Create k-NN  object
knn = KNeighborsClassifier()

# Train the k-NN model using the training sets
knn.fit(train_features, train_labels)

#Make predictions on the test data
label_predct_2 = knn.predict(test_features)

print("####  CONFUSION MATRIX  ####")
print(confusion_matrix(test_labels, label_predct_2), "\n")  

print("######################################################")
print(classification_report(test_labels, label_predct_2))  
print("######################################################")

knnaccuracy =  knn.score(test_features,test_labels)*100
print("The prediction accuracy of the k-NN Classifier is: %.2f" %knnaccuracy,"%")

kappa2 = cohen_kappa_score(test_labels, label_predct_2) * 100
print("Kappa statistics is %.2f" %kappa2, "%")

print("-----------------------END OF KNN CLASSIFIER-------------------------------", "\n")

print("-----------------------NAIVE BAYES CLASSIFIER-------------------------------")
# Create Create a Gaussian Classifier object
gnb = GaussianNB()

# Train the k-NN model using the training sets
gnb.fit(train_features, train_labels)

#Make predictions on the test data
label_predct_3 = gnb.predict(test_features)

print("####  CONFUSION MATRIX  ####")
print(confusion_matrix(test_labels, label_predct_3), "\n")  

print("######################################################")
print(classification_report(test_labels, label_predct_3))  
print("######################################################")

gnbaccuracy =  gnb.score(test_features,test_labels)*100
print("The prediction accuracy of the Gaussian Naive Bayes Classifier is: %.2f" %gnbaccuracy,"%")

kappa3 = cohen_kappa_score(test_labels, label_predct_3) * 100
print("Kappa statistics is %.2f" %kappa3, "%")

print("-----------------------END OF NAIVE BAYES CLASSIFIER-------------------------------", "\n")