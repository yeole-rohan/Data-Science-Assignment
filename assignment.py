# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 19:20:06 2019

@author: lenovo
"""


import numpy as np
import pandas as pd
import random
import warnings
warnings.filterwarnings("ignore")
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV

dataset=pd.read_csv("bank-additional-full.csv")
dataset=dataset.drop(['duration'],axis=1)
print(dataset.shape)
dataset.head()
dataset.info()


#Exploratory Analysis
#Categorical Variables
categorical_variables = ['job', 'marital', 'education', 'default','housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome','y']
for col in categorical_variables:
    plt.figure(figsize=(10,4))
    sns.barplot(dataset[col].value_counts().values, dataset[col].value_counts().index)
    plt.title(col)
    plt.tight_layout()



for col in categorical_variables:
    plt.figure(figsize=(10,4))
    #Returns counts of unique values for each outcome for each feature.
    pos_counts = dataset.loc[dataset.y.values == 'yes', col].value_counts() 
    neg_counts = dataset.loc[dataset.y.values == 'no', col].value_counts()
    
    all_counts = list(set(list(pos_counts.index) + list(neg_counts.index)))
    
    #Counts of how often each outcome was recorded.
    freq_pos = (dataset.y.values == 'yes').sum()
    freq_neg = (dataset.y.values == 'no').sum()
    
    pos_counts = pos_counts.to_dict()
    neg_counts = neg_counts.to_dict()
    
    all_index = list(all_counts)
    all_counts = [pos_counts.get(k, 0) / freq_pos - neg_counts.get(k, 0) / freq_neg for k in all_counts]

    sns.barplot(all_counts, all_index)
    plt.title(col)
    plt.tight_layout()
    

## Creating new variables (variable name + '_un') to capture the information if the missing values are at random or is there
## a pattern in the missing values.
significant_cat_variables = ['education','job','housing','loan']
for var in significant_cat_variables:
    dataset[var + '_un'] = (dataset[var] == 'unknown').astype(int)

def cross_tab_imputation(dataset,f1,f2):
    jobs=list(dataset[f1].unique())
    edu=list(dataset[f2].unique())
    dataframes=[]
    for e in edu:
        datasete=dataset[dataset[f2]==e]
        datasetjob=datasete.groupby(f1).count()[f2]
        dataframes.append(datasetjob)
    xx=pd.concat(dataframes,axis=1)
    xx.columns=edu
    xx=xx.fillna(0)
    return xx

cross_tab_imputation(dataset,'job','education')

dataset['job'][dataset['age']>60].value_counts()

dataset.loc[(dataset['age']>60) & (dataset['job']=='unknown'), 'job'] = 'retired'
dataset.loc[(dataset['education']=='unknown') & (dataset['job']=='management'), 'education'] = 'university.degree'
dataset.loc[(dataset['education']=='unknown') & (dataset['job']=='services'), 'education'] = 'high.school'
dataset.loc[(dataset['education']=='unknown') & (dataset['job']=='housemaid'), 'education'] = 'basic.4y'
dataset.loc[(dataset['job'] == 'unknown') & (dataset['education']=='basic.4y'), 'job'] = 'blue-collar'
dataset.loc[(dataset['job'] == 'unknown') & (dataset['education']=='basic.6y'), 'job'] = 'blue-collar'
dataset.loc[(dataset['job'] == 'unknown') & (dataset['education']=='basic.9y'), 'job'] = 'blue-collar'
dataset.loc[(dataset['job']=='unknown') & (dataset['education']=='professional.course'), 'job'] = 'technician'

cross_tab_imputation(dataset,'job','education')


jobhousing=cross_tab_imputation(dataset,'job','housing')
jobloan=cross_tab_imputation(dataset,'job','loan')



def fillhous(dataset,jobhousing):
    """Function for imputation via cross-tabulation to fill missing values for the 'housing' categorical feature"""
    jobs=['housemaid','services','admin.','blue-collar','technician','retired','management','unemployed','self-employed','entrepreneur','student']
    house=["no","yes"]
    for j in jobs:
        ind=dataset[np.logical_and(np.array(dataset['housing']=='unknown'),np.array(dataset['job']==j))].index
        mask=np.random.rand(len(ind))<((jobhousing.loc[j]['no'])/(jobhousing.loc[j]['no']+jobhousing.loc[j]['yes']))
        ind1=ind[mask]
        ind2=ind[~mask]
        dataset.loc[ind1,"housing"]='no'
        dataset.loc[ind2,"housing"]='yes'
    return dataset

def fillloans(dataset,jobloan):
    """Function for imputation via cross-tabulation to fill missing values for the 'loan' categorical feature"""
    jobs=['housemaid','services','admin.','blue-collar','technician','retired','management','unemployed','self-employed','entrepreneur','student']
    loan=["no","yes"]
    for j in jobs:
        ind=dataset[np.logical_and(np.array(dataset['loan']=='unknown'),np.array(dataset['job']==j))].index
        mask=np.random.rand(len(ind))<((jobloan.loc[j]['no'])/(jobloan.loc[j]['no']+jobloan.loc[j]['yes']))
        ind1=ind[mask]
        ind2=ind[~mask]
        dataset.loc[ind1,"loan"]='no'
        dataset.loc[ind2,"loan"]='yes'
    return dataset

dataset=fillhous(dataset,jobhousing)

dataset=fillloans(dataset,jobloan)

numerical_variables = ['age','campaign', 'pdays', 'previous', 'emp.var.rate', 'cons.price.idx','cons.conf.idx','euribor3m',
                      'nr.employed']
dataset[numerical_variables].describe()

def drawhist(dataset,feature):
    plt.hist(dataset[feature])
    
drawhist(dataset,'pdays')
plt.show()

plt.hist(dataset.loc[dataset.pdays != 999, 'pdays'])
plt.show()

pd.crosstab(dataset['pdays'],dataset['poutcome'], values=dataset['age'], aggfunc='count', normalize=True)

#Add new categorical variables to our dataframe.
dataset['pdays_missing'] = 0
dataset['pdays_less_5'] = 0
dataset['pdays_greater_15'] = 0
dataset['pdays_bet_5_15'] = 0
dataset['pdays_missing'][dataset['pdays']==999] = 1
dataset['pdays_less_5'][dataset['pdays']<5] = 1
dataset['pdays_greater_15'][(dataset['pdays']>15) & (dataset['pdays']<999)] = 1
dataset['pdays_bet_5_15'][(dataset['pdays']>=5)&(dataset['pdays']<=15)]= 1
dataset_dropped_pdays = dataset.drop('pdays', axis=1);


dataset_with_dummies=pd.get_dummies(dataset_dropped_pdays)

def dropfeature(dataset,f):
    """Drops one of the dummy variables."""
    dataset=dataset.drop(f,axis=1)
    return dataset

features_dropped = ['default_no','housing_no','loan_no','y_no','marital_single','contact_cellular',
                    'education_unknown','job_unknown','housing_unknown','loan_unknown', 'pdays_less_5']
dataset_clean = dropfeature(dataset_with_dummies, features_dropped)


def drawheatmap(dataset):
    '''Builds the heat map for the given data'''
    f, ax = plt.subplots(figsize=(15, 15))
    sns.heatmap(dataset.corr(method='spearman'), annot=False, cmap='coolwarm')
    
def drawhist(dataset,feature):
    '''Draws an histogram for a feature in a data frame (dataset)'''
    plt.hist(dataset[feature])

def functionreplace(dataset,fea,val1,val2):
    '''Replaces value (val1) with value (val2) in the data frame (dataset) for a feature (fea)'''
    dataset[fea].replace(val1,val2)
    return dataset

def drawbarplot(dataset,x,y):
    '''Draws a bar plot for a given feature x and y in a data frame'''
    sns.barplot(x=x, y=y, data=dataset)


drawheatmap(dataset_clean)



def getmeanauc(aucs,model):
    ''' Calculates the mean AUC for all the cross-validated samples and computes the value of C (Regularization Parameter) 
    for which max mean auc is obtained'''
    #Initialize empty array to hold mean AUC values.
    meanauc = []
    maxmean = 0 #Initial value for maximum mean AUC
    models_with_1_param = ['Logistic_Regression' , 'Ada_Boost']
    models_with_2_params = ['Decision_Tree' , 'Random_Forest', 'Grad_Boost']
    for c in aucs: #For loop to append AUC value to meanauc array.
        meanauc.append(np.mean(aucs[c]))
        if maxmean < np.mean(aucs[c]):
            maxmean = np.mean(aucs[c]) #Adjust value of maxmean
            cval = c
    if model in models_with_1_param:
        print("C value for max auc is: ",cval)
        print("Max Mean Auc corresponding to the optimal value of C = ", maxmean)
        return meanauc,cval
    if model in models_with_2_params:
        listSL=cval.split('L')
        splitval=int(listSL[0]) #Stores minimum split value for max AUC
        leafval=int(listSL[1]) #Stores minimum leaf value for max AUC
        print("min_sample_split value for max auc is:",splitval)
        print("min_sample_leaf value for max auc is:",leafval)
        print("Max mean AUC corresponding to optimal leaf and split value = ",maxmean)
        return meanauc,splitval,leafval

def plot_mean_auc_LR(aucs,cs, label):
    '''Plots different values of mean auc versus the hyperparameter C'''
    plt.plot(np.log10(cs),aucs, label = label )
    plt.xlabel("C (Regularization Parameter)")
    plt.ylabel("Mean AUC")
    plt.legend()
        
def plotfeatureimportances(train, importance):
    '''Plots feature importance in a sorted order and shows the most significant variables at the top'''
    X = list(train.columns)
    X.remove('y_yes')
    feature_importance_dataset = pd.DataFrame(data = importance, index = X, columns=['coefficient_values'])
    feature_importance_dataset['sort'] = feature_importance_dataset.coefficient_values.abs()
    sorted_feature_imp_dataset = feature_importance_dataset.sort_values(by='sort', ascending=False).drop('sort', axis=1)
    fig, ax = plt.subplots()
    fig.set_size_inches(10, 15)
    sns.barplot(np.array(sorted_feature_imp_dataset.coefficient_values), np.array(sorted_feature_imp_dataset.index.values))
    plt.title('Feature Importances')
    plt.xlabel('Coefficients')
    plt.ylabel('Feature Names')
    
def plotfeatureimp(fl,col):
    '''Plots the feature importance of all the independent variables in the model'''
    f=plt.figure(figsize=(10,15))
    plt.barh(range(len(fl)),fl)
    plt.yticks(range(len(col[:-1])),col[:-1])
    
def plotAUCdatasetRF(aucs,leafs,splits):
    '''Plots AUC for each value of Leaf and Split combination'''
    for i in range(len(splits)):
        plt.plot(leafs,aucs[len(leafs)*i:len(leafs)*i+len(leafs)], label = 'Split value= ' + str(splits[i]))
    plt.legend()
    plt.xlabel('Leaf Values')
    plt.ylabel('Mean AUC')
    
def plot_mean_auc_Ada_Boost(aucs, estimators, label):
    '''Plots different values of mean auc versus the Estimators for AdaBoosting'''
    plt.plot(estimators,aucs, label = label )
    plt.xlabel("Estimators")
    plt.ylabel("Mean AUC")
    plt.legend()
    
    
    
#Model Building and Evaluation
from sklearn.model_selection import train_test_split
train, test = train_test_split(dataset_clean, train_size=0.8, random_state=5)
print('Original:', (dataset_clean.y_yes).mean(), 'Train:', (train.y_yes).mean(), 'Test:', (test.y_yes).mean())

train, test = train_test_split(dataset_clean, train_size=0.8, stratify=dataset_clean.y_yes.values, random_state=5)
print('Original:', (dataset_clean.y_yes).mean(), 'Train:', (train.y_yes).mean(), 'Test:', (test.y_yes).mean())


#Logistic Regression Model
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
def LRmodel(train , validate , c, l_1 = False):
    '''Fits the Logistic Regression Model on the training data set and calculates evaluation metrics on the validation set
    with the regularization constant C'''
    X=list(train.columns) #Create list of column labels from training data
    Y='y_yes'
    X.remove('y_yes')
    scaler = StandardScaler().fit(train[X]) #Standardize features by removing the mean and scaling to unit variance
    train_std = scaler.transform(train[X]) #Compute the mean and standard deviation on training set
    validate_std = scaler.transform(validate[X])
    xtrain = train_std
    ytrain = train[Y]
    xval = validate_std
    yval = validate[Y]
    if l_1:
        logreg = LogisticRegression(C = c, penalty= 'l1') #Apply logistic regression on L1 penalty
    else:
        logreg = LogisticRegression(C=c)
    logreg.fit(xtrain,ytrain)
    pred_proba_val = logreg.predict_proba(xval)[:,1]
    auc = metrics.roc_auc_score(yval, pred_proba_val)
    fpr, tpr, threshold = metrics.roc_curve(yval, pred_proba_val)
    accuracy = metrics.accuracy_score(yval, logreg.predict(xval))
    return auc,logreg.coef_, tpr, fpr, threshold, accuracy

#AdaBoostClassifier Model
from sklearn.ensemble import AdaBoostClassifier
def adaboost(settrain,settest, nestimator = 100):
    X=list(settrain.columns)
    Y='y_yes'
    X.remove('y_yes')
    xtrain=settrain[X]
    ytrain=settrain[Y]
    xtest=settest[X]
    ytest=settest[Y]
    #Instantiate a Decision Stump
    dt_stump = DecisionTreeClassifier(max_depth=1, min_samples_leaf=1)
    
    #Instantitate an AdaBoostClassifier using the decision stump defined above
    ad = AdaBoostClassifier(base_estimator=dt_stump, n_estimators=nestimator)
    
    #fit the AdaBoostClassifier on the training data
    ad.fit(xtrain,ytrain)
    
    #Predict the Y values for the test/validation data
    Y_pred = ad.predict(xtest)
    
    #Predict class probabilities of input validation data
    adplot=ad.predict_proba(xtest)
    
    
    adpre=adplot[:,1]
    
    #Computation to compute AUC score
    adfpr, adtpr, adthresholds=metrics.roc_curve(ytest,adpre)
    adscore=metrics.roc_auc_score(ytest,adpre)
    
    #Feature importances. The higher the score, the more important the feature.
    ii=ad.feature_importances_
    return adscore,ii


from sklearn.cross_validation import KFold
from sklearn import metrics
from sklearn import preprocessing
def kfoldadaboost(dataset, k, estimators):
    aucs ={}
    kf=KFold(len(dataset),k) #Provides indices to split data in train/test sets
    for e in estimators:
        for train_idx, vali_idx in kf:
            cv_train,cv_validate=dataset.iloc[train_idx,:], dataset.iloc[vali_idx,:]
            
            #Run AdaBoostClassifier function defined above based on user input
            core,f= adaboost(cv_train,cv_validate, nestimator = e) 
            
            #storing the auc Scores in the aucs dictionary for all the estimator values.
            aucs[e] = []
            aucs[e].append(core)
    return aucs
adaauc_test,adafea=adaboost(train,test,adac)
plotfeatureimportances(train,adafea)

