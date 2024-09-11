#!/usr/bin/env python
# coding: utf-8

# # Credit Card Fraud Detection Project

# Credit card fraud has become a significant concern in today's digital era. With the increasing reliance on online transactions, fraudsters have become more sophisticated, making it crucial to develop robust systems for detecting and preventing fraudulent activities. Credit card fraud can lead to financial losses for both individuals and financial institutions, erode customer trust, and impact overall economic stability.
# 
# A credit card fraud detection project aims to identify fraudulent transactions by leveraging machine learning and data analysis techniques. By analyzing historical transaction data, these projects aim to build models that can accurately distinguish between legitimate and fraudulent transactions, allowing for timely intervention and prevention of fraud.
# 
# Project Objective: The primary objective of a credit card fraud detection project is to develop an effective system that can identify fraudulent transactions with high accuracy and minimize false positives (legitimate transactions wrongly flagged as fraud). This involves building machine learning models that can learn from historical data and generalize well to detect new and emerging fraud patterns.

# ### Importing libraries required

# In[112]:


# Import the necessary modules

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns
import itertools
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix, recall_score, f1_score


# ### Load the dataset

# In[174]:


# Load the csv file

dataframe = pd.read_csv("creditcard.csv")
dataframe.head()


# In[171]:


print(dataframe.columns)


# In[172]:


dataframe.info()


# In[173]:


dataframe.describe()


# ### DATA QUALITY CHECK
# 
# 
# #### Check for NULL/MISSING values

# In[117]:


round(100 * (dataframe.isnull().sum()/len(dataframe)),2).sort_values(ascending=False)


# In[118]:


# percentage of missing values in each row
round(100 * (dataframe.isnull().sum(axis=1)/len(dataframe)),2).sort_values(ascending=False)


# Note:
# There are no missing / Null values either in columns or rows

# #### Duplicate check

# In[183]:


dataframe_d=dataframe.copy()
dataframe_d.drop_duplicates(subset=None, inplace=True)


# In[120]:


dataframe.shape


# In[121]:


dataframe_d.shape


# Note:
# Duplicate are found in the records

# In[122]:


#assigning duplicates removed dataset to original dataset
dataframe=dataframe_d
dataframe.shape


# ###  Exploratory Data Analysis

# In[123]:


non_fraud = len(dataframe[dataframe.Class == 0])
fraud = len(dataframe[dataframe.Class == 1])
fraud_percent = (fraud / (fraud + non_fraud)) * 100
l=[non_fraud,fraud]
labels = ["Genuine", "Fraud"]
my_color=['lightblue','red']

print("Number of Genuine transactions: ", non_fraud)
print("Number of Fraud transactions: ", fraud)
print("Percentage of Genuine transactions: {:.4f}".format(100-fraud_percent))
print("Percentage of Fraud transactions: {:.4f}".format(fraud_percent))
plt.pie(l,labels=labels,shadow=True,colors=my_color)
plt.legend()
plt.show()


# #### Here a pie is plotted to represent the non fraud and genuine transactions.Red ones are fraud blue color represent fraud tranctions

# In[124]:


# Visualize the "Labels" column in our dataset

labels = ["Genuine", "Fraud"]
count_classes = dataframe.value_counts(dataframe['Class'], sort= True)
count_classes.plot(kind = "bar", rot = 0)
plt.title("Class Count")
plt.ylabel("Count")
plt.xticks(range(2), labels)
plt.show()


# #### The above bar graph describes the count of genuine and fraud transactions

# ### Transactions amount

# In[176]:


fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,6))
s = sns.boxplot(ax = ax1, x="Class", y="Amount", hue="Class",data=dataframe, palette="PRGn",showfliers=True)
s = sns.boxplot(ax = ax2, x="Class", y="Amount", hue="Class",data=dataframe, palette="PRGn",showfliers=False)
plt.show();


# In[178]:


tmp = dataframe[['Amount','Class']].copy()
class_0 = tmp.loc[tmp['Class'] == 0]['Amount']
class_1 = tmp.loc[tmp['Class'] == 1]['Amount']
class_0.describe()


# In[179]:


class_1.describe()


# The real transaction have a larger mean value, larger Q1, smaller Q3 and Q4 and larger outliers; fraudulent transactions have a smaller Q1 and mean, larger Q4 and smaller outliers.
# 
# 

# ## Features density plot

# In[168]:


var = dataframe.columns.values

i = 0
t0 = dataframe.loc[dataframe['Class'] == 0]
t1 = dataframe.loc[dataframe['Class'] == 1]

sns.set_style('whitegrid')
plt.figure()
fig, ax = plt.subplots(8,4,figsize=(16,28))

for feature in var:
    i += 1
    plt.subplot(8,4,i)
    sns.kdeplot(t0[feature], bw=0.5,label="Class = 0")
    sns.kdeplot(t1[feature], bw=0.5,label="Class = 1")
    plt.xlabel(feature, fontsize=12)
    locs, labels = plt.xticks()
    plt.tick_params(axis='both', which='major', labelsize=12)
plt.show();


# For some of the features we can observe a good selectivity in terms of distribution for the two values of Class: V4, V11 have clearly separated distributions for Class values 0 and 1, V12, V14, V18 are partially separated, V1, V2, V3, V10 have a quite distinct profile, whilst V25, V26, V28 have similar profiles for the two values of Class.
# 
# In general, with just few exceptions (Time and Amount), the features distribution for legitimate transactions (values of Class = 0) is centered around 0, sometime with a long queue at one of the extremities. In the same time, the fraudulent transactions (values of Class = 1) have a skewed (asymmetric) distribution.

# #### There is clearly a lot more variability in the transaction values for non-fraudulent transactions.

# In[128]:


#Checking correlation in heatmap
plt.figure(figsize=(30,20))
corr=dataframe.corr()
sns.heatmap(corr,cmap="coolwarm",annot=True)


# #### We observe that most of the data features are not correlated. This is because before publishing, most of the features were presented to a Principal Component Analysis (PCA) algorithm. The features V1 to V28 are most probably the Principal Components resulted after propagating the real features through PCA. We do not know if the numbering of the features reflects the importance of the Principal Components.

# In[129]:


# Perform Scaling
scaler = StandardScaler()
dataframe["NormalizedAmount"] = scaler.fit_transform(dataframe["Amount"].values.reshape(-1, 1))
dataframe.drop(["Amount", "Time"], inplace= True, axis= 1)

Y = dataframe["Class"]
X = dataframe.drop(["Class"], axis= 1)


# In[130]:


X.head()


# In[131]:


Y.head()


# ## Splitting the data 

# In[132]:


# Split the data
(train_X, test_X, train_Y, test_Y) = train_test_split(X, Y, test_size= 0.3, random_state= 42)

print("Shape of train_X: ", train_X.shape)
print("Shape of test_X: ", test_X.shape)


# #### Let's train different models on our dataset and observe which algorithm works better for our problem.
# 
# Let's apply Random Forests , Decision Trees,Logistic Regression algorithms to our dataset.

# ##  Decision Tree Classifier
# 
# ### Model Building

# In[159]:


# Decision Tree Classifier
decision_tree = DecisionTreeClassifier()
decision_tree.fit(train_X, train_Y)

predictions_dt = decision_tree.predict(test_X)
decision_tree_score = decision_tree.score(test_X, test_Y) * 100
print("Decision Tree Score: ", decision_tree_score)


# ### Confusion Matrix 

# In[161]:


confusion_matrix_dt = confusion_matrix(test_Y, predictions_dt.round())
print("Confusion Matrix - Decision Tree")
print(confusion_matrix_dt)


# In[160]:


plot_confusion_matrix(confusion_matrix_dt, classes=[0, 1], title= "Confusion Matrix - Decision Tree")


# ### Scores

# In[166]:


print("Evaluation of Descision Tree Model")
print()
metrics(test_Y, predictions_dt.round())


# ## Random Forest
# 
# ### Model Building

# In[158]:


# Random Forest
random_forest = RandomForestClassifier(n_estimators= 100)
random_forest.fit(train_X, train_Y)

predictions_rf = random_forest.predict(test_X)
random_forest_score = random_forest.score(test_X, test_Y) * 100
print("Random Forest Score: ", random_forest_score)


# ###  Confusion Matrix

# In[162]:


# Plot confusion matrix for Random Forests

confusion_matrix_rf = confusion_matrix(test_Y, predictions_rf.round())
print("Confusion Matrix - Random Forest")
print(confusion_matrix_rf)


# In[163]:


plot_confusion_matrix(confusion_matrix_rf, classes=[0, 1], title= "Confusion Matrix - Random Forest")


# ### Scores

# In[165]:


print("Evaluation of Random Forest Model")
print()
metrics(test_Y, predictions_rf.round())


# ## Logistic Regression
# 
# ### Model Building

# In[157]:


from sklearn.linear_model import LogisticRegression
log = LogisticRegression()
log.fit(train_X,train_Y)
predictions_lg=log.predict(test_X)
log_score = log.score(test_X, test_Y) * 100

print("Logistic Regression Score:",log_score)


# ### Confusion matrix

# In[137]:


# Plot confusion matrix for Logistic Regression

confusion_matrix_lg = confusion_matrix(test_Y, predictions_lg.round())
print("Confusion Matrix -Logistic regression ")
print(confusion_matrix_lg)
plot_confusion_matrix(confusion_matrix_lg, classes=[0, 1], title= "Confusion Matrix -Logistic regression ")


# ###  Scores 

# In[167]:


print("Evaluation of Logistic Regression")
print()
metrics(test_Y, predictions_lg.round())


# In[138]:


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()



# In[145]:


# The below function prints the following necesary metrics

def metrics(actuals, predictions):
    print("Accuracy: {:.5f}".format(accuracy_score(actuals, predictions)))
    print("Precision: {:.5f}".format(precision_score(actuals, predictions)))
    print("Recall: {:.5f}".format(recall_score(actuals, predictions)))
    print("F1-score: {:.5f}".format(f1_score(actuals, predictions)))
    


# ### Clearly, Random Forest model works better than Decision Trees and Logistic regression

# However, if we look closely, we can see that our dataset has a major **class imbalance** issue. 
# More than 99% of transactions are real (avoid fraud), while 0.17% of transactions are fraudulent.
# 
# 
# If we train our model with such a distribution without considering the imbalance problems, it predicts the label with a higher value given to actual transactions (since there is more evidence about them) and so achieves more accuracy.

# There are several methods that can be used to address the class imbalance issue. One of these is oversampling.
#  
# Oversampling the minority class is one way to deal with unbalanced datasets. Duplicating examples from the minority class is the simplest method, but these examples don't provide any new insight into the model. 
# 
# Instead, fresh examples can be created by synthesising the current ones. The **Synthetic Minority Oversampling Technique**, or **SMOTE** for short, is a technique for data augmentation for the minority class.

# ### Oversampling on RandomForest
# 

# In[148]:


# Performing oversampling on RF

from imblearn.over_sampling import SMOTE

X_resampled, Y_resampled = SMOTE().fit_resample(X, Y)
print("Resampled shape of X: ", X_resampled.shape)
print("Resampled shape of Y: ", Y_resampled.shape)

value_counts = Counter(Y_resampled)
print(value_counts)

(train_X, test_X, train_Y, test_Y) = train_test_split(X_resampled, Y_resampled, test_size= 0.3, random_state= 42)


# In[149]:


# Build the Random Forest classifier on the new dataset

rf_resampled = RandomForestClassifier(n_estimators = 100)
rf_resampled.fit(train_X, train_Y)
predictions_resampled = rf_resampled.predict(test_X)
random_forest_score_resampled = rf_resampled.score(test_X, test_Y) * 100


# In[150]:


predictions_resampled


# In[151]:


# Visualize the confusion matrix

cm_resampled = confusion_matrix(test_Y,predictions_resampled.round())
print("Confusion Matrix - Random Forest")
print(cm_resampled)


# In[152]:


plot_confusion_matrix(cm_resampled, classes=[0, 1], title= "Confusion Matrix - Random Forest After Oversampling")


# In[153]:


print("Evaluation of Random Forest Model")
print()
metrics(test_Y, predictions_resampled.round())


# Now it is evident that after addressing the class imbalance problem, our Random forest classifier with SMOTE performs far better than the Random forest classifier withour SMOTE

# In[154]:


a=[[1.10321543,-0.040296215,1.267332089,1.28909147,-0.735997164,0.288069163,-0.586056786,0.18937971,0.782332892,-0.267975067,-0.45031128,0.936707715,0.708380406,-0.468647288,0.354574063,-0.246634656,-0.009212378,-0.595912406,-0.575681622,-0.113910177,-0.024612006,0.196001953,0.013801654,0.103758331,0.364297541,-0.382260574,0.092809187,0.037050517,9.99
]]


# In[155]:


print(rf_resampled.predict(a))

Since output is 0 thre is no Fraud
# In[156]:


import pickle
pickle.dump(rf_resampled,open("model.pkl","wb"))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




