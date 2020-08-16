# ML_hello-world
Implementing tutorial located: https://machinelearningmastery.com/machine-learning-in-python-step-by-step/


Starting with setting up the environment. 
1. downloaded anaconda 
2. started a new conda environment "hello-world"
3. created testing script for environment 
![Image of script](https://github.com/hkhailee/ML_hello-world/blob/master/test-script.png)
4. ran script 
![Image of scriptOutput](https://github.com/hkhailee/ML_hello-world/blob/master/test-script-output.png)
5. created a load libraries script 
```rom pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
```
6. ran script without errors
7. created a data reading script (data-script) to get the iris dataset:
```#import pandas
import pandas as pd
# Load dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pd.read_csv(url, names=names)
```
8. added ``` print(dataset.shape) ``` to data-script, output ```(150, 5)``` for 150 rows 5 columns
9. added ```print(dataset.head(20))``` to data-script, output first 20 rows of iris data
10. added ```print(dataset.describe())``` to data-script outputed count, mean, the min and max values as well as some percentiles for the dataset 
11. added ```print(dataset.groupby('class').size())``` to data-script output : 
```class
Iris-setosa        50
Iris-versicolor    50
Iris-virginica     50
dtype: int64
```
function was able to organize data based on a column value.<br /> 
<strong>----end expirmentation---- </strong><br /> 
<strong>----begin data visualization----</strong>
  
  using different pyplot functions to view attributes from the data set 
  Gaussian distribution or regular distrubution:
  https://en.wikipedia.org/wiki/Normal_distribution
  
  evaluation of data to beigin suggestions of a high correlation and a predictable relationship.
  that can then be manipulated with algorithms. 
<strong>--- model creation and algorithms ----</strong><br />
  
  
  From our colllected data set we will use 80% to train our models 
  then 20% to check which model is the best for this set of data
  
  https://machinelearningmastery.com/k-fold-cross-validation/
- That k-fold cross validation is a procedure used to estimate the skill of the model on new data.
  Shuffle the dataset randomly.<br /> 
    Split the dataset into k groups<br /> 
    For each unique group:<br /> 
      Take the group as a hold out or test data set<br /> 
      Take the remaining groups as a training data set<br /> 
      Fit a model on the training set and evaluate it on the test set<br /> 
      Retain the evaluation score and discard the model<br /> 
  Summarize the skill of the model using the sample of model evaluation scores<br /> 
  
  Bias -> limited flecibility to learn the true signal from the the data set <br />
  Variance -> errror from sensiticity to small fluctations in the training set <br />
  
  For example: if you have a data set of 6 [0.1, 0.2, 0.3, 0.4, 0.5, 0.6] <br />
  we can split and shuffle the data using kfold = KFold(3, True, 1) <br />
  you can then use Kfold.split(data) from scikit-learn (which we have imported) <br />
  to have three sets of data that will be trained on 2 and tested on one to see how well a certain model preforms <br />
  ```train: [0.1 0.4 0.5 0.6], test: [0.2 0.3]
  train: [0.2 0.3 0.4 0.6], test: [0.1 0.5]
  train: [0.1 0.2 0.3 0.5], test: [0.4 0.6]
  ```
  <strong>Models</strong><br />
  1. Logistic Regression (LR)
  2. Linear Discriminant Analysis (LDA)
  3. K-Nearest Neighbors (KNN).
  4. Classification and Regression Trees (CART).
  5. Gaussian Naive Bayes (NB).
  6. Support Vector Machines (SVM).
  
  https://machinelearningmastery.com/randomness-in-machine-learning/
  
  running the following code in our script:
```
#spot check algorithms 
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))

#evaluate each model in turn 
results = []
names = []
for name, model in models:
	kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
	cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
	results.append(cv_results)
	names.append(name)
	print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
```

output:
```
LR: 0.941667 (0.065085)
LDA: 0.975000 (0.038188)
KNN: 0.958333 (0.041667)
CART: 0.950000 (0.040825)
NB: 0.950000 (0.055277)
SVM: 0.983333 (0.033333)
```

running with 35 k splits output:
```
LR: 0.952381 (0.116642)
LDA: 0.973810 (0.086307)
KNN: 0.959524 (0.100340)
CART: 0.950000 (0.111270)
NB: 0.950000 (0.111270)
SVM: 0.980952 (0.077372)
```

we can see that support vector machine have the least amount of error for this data set, with Linear Discriminant Analysis being a close second <br />
more on SVM: https://en.wikipedia.org/wiki/Support_vector_machine<br /> 
An SVM model is a representation of the examples as points in space, mapped so that the examples of the separate categories are divided by a clear gap that is as wide as possible.

To see which set was actually better in the long run, Changing k back to the original 10. Adding this to our data-script
```
# Make predictions on validation dataset
model = SVC(gamma='auto')
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)

# Evaluate predictions for SVC
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

#make predictions
model = LinearDiscriminantAnalysis()
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)

# Evaluate predictions for LDA
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))
```

output:

```
0.9666666666666667
[[11  0  0]
 [ 0 12  1]
 [ 0  0  6]]
                 precision    recall  f1-score   support

    Iris-setosa       1.00      1.00      1.00        11
Iris-versicolor       1.00      0.92      0.96        13
 Iris-virginica       0.86      1.00      0.92         6

       accuracy                           0.97        30
      macro avg       0.95      0.97      0.96        30
   weighted avg       0.97      0.97      0.97        30

1.0
[[11  0  0]
 [ 0 13  0]
 [ 0  0  6]]
                 precision    recall  f1-score   support

    Iris-setosa       1.00      1.00      1.00        11
Iris-versicolor       1.00      1.00      1.00        13
 Iris-virginica       1.00      1.00      1.00         6

       accuracy                           1.00        30
      macro avg       1.00      1.00      1.00        30
   weighted avg       1.00      1.00      1.00        30
 ```
 
 the first percentage and confusion matrix coming from SVC the second from LDA we can see that LDA was actually the better model choice. <br />
 
<strong>---end model creation and algorithms ----</strong><br />
