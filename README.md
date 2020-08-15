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
<b>----end expirmentation---- <\b><br /> 
<b>----begin data visualization----<\b>
  
  using different pyplot functions to view attributes from the data set 
  Gaussian distribution or regular distrubution:
  https://en.wikipedia.org/wiki/Normal_distribution
  
  evaluation of data to beigin suggestions of a high correlation and a predictable relationship.
  that can then be manipulated with algorithms. 
<b>--- model creation and algorithms ----<\b>
  
  
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
  
