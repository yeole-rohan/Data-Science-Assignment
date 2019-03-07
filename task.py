#import the needed libraries
import matplotlib.pyplot as plt
#setting dimmension of a graph
plt.rcParams["figure.figsize"] = (12, 10)
import pandas as pd
import numpy as np
import seaborn as sns
from pandas.plotting import scatter_matrix
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.preprocessing import Imputer
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

#load the dataset
dataset = pd.read_csv("bank-additional-full.csv")

#data Exploration
dataset.info()

#checking for missing values
dataset.apply(lambda x: sum(x.isnull()),axis=0)

#head of a data with default 5 row and total 21 columns
print(dataset.head())

#Target Varibale distribution
count = dataset.groupby("y").size()
percent = count/len(dataset)*100
print(percent)

#Impute Outliner Function
def impute_outliners(df,column,minimum,maximum):
    col_values = df[column].values
    df[column] = np.where(np.logical_or(col_values < minimum,col_values > maximum),              col_values.mean(),col_values)
    return df

#lets see statistic of Numerical variables before Outlier treatment
dataset.describe()

# scatter plot matrix
scatter_matrix(dataset)
plt.show()
