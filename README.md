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
<b>----end expirmentation---- <b><br /> 
<b>----begin data visualization----<b>
  
  using different pyplot functions to view attributes from the data set 
  Gaussian distribution or regular distrubution:
  https://en.wikipedia.org/wiki/Normal_distribution
