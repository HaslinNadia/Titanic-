import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

st.header("Welcome to my Streamlit App")
st.title("Titanic Dataset Analysis and Prediction")

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

train.isnull().sum().sort_values(ascending = False)
test.isnull().sum().sort_values(ascending = False)
train = train.drop(['Ticket', 'Cabin'], axis = 1)
test = test.drop(['Ticket', 'Cabin'], axis = 1)

mode = train['Embarked'].dropna().mode()[0]
mode
train['Embarked'].fillna(mode, inplace = True)
median = test['Fare'].dropna().median()
median
test['Fare'].fillna(median, inplace = True)

combine = pd.concat([train, test], axis = 0).reset_index(drop = True)
combine['Sex'] = combine['Sex'].map({'male': 0, 'female': 1})
age_nan_indices = list(combine[combine['Age'].isnull()].index)
len(age_nan_indices)

for index in age_nan_indices:
    median_age = combine['Age'].median()
    predict_age = combine['Age'][(combine['SibSp'] == combine.iloc[index]['SibSp']) 
                                 & (combine['Parch'] == combine.iloc[index]['Parch'])
                                 & (combine['Pclass'] == combine.iloc[index]["Pclass"])].median()
    if np.isnan(predict_age):
        combine['Age'].iloc[index] = median_age
    else:
        combine['Age'].iloc[index] = predict_age

combine['Age'].isnull().sum()
combine['Fare'] = combine['Fare'].map(lambda x: np.log(x) if x > 0 else 0)
combine = combine.drop(['Name','Embarked','Parch','SibSp','Pclass'], axis = 1)

train = combine[:len(train)]
test = combine[len(train):]
train['Survived'] = train['Survived'].astype('int')
test = test.drop('Survived', axis = 1)

X_train = train.drop('Survived', axis = 1)
Y_train = train['Survived']
X_test = test.copy()

option = st.sidebar.selectbox(
    'Choose One',
     ['Logistic Regression','Support Vector Machines','k-NN','Gausssian Naive Bayes','Decision Tree','Random Forest'])

if option=='Logistic Regression':
    logreg = LogisticRegression()
    logreg.fit(X_train, Y_train)
    Y_pred = logreg.predict(X_test)
    acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
    
    st.write(acc_log)

elif option=='Support Vector Machines':
    svc = SVC()
    svc.fit(X_train, Y_train)
    Y_pred = svc.predict(X_test)
    acc_svc = round(svc.score(X_train, Y_train) * 100, 2)
    
    st.write(acc_svc)

elif option=='k-NN':
    knn = KNeighborsClassifier(n_neighbors = 5)
    knn.fit(X_train, Y_train)
    Y_pred = knn.predict(X_test)
    acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
      
    st.write(acc_knn)

elif option=='Gausssian Naive Bayes':
    gaussian = GaussianNB()
    gaussian.fit(X_train, Y_train)
    Y_pred = gaussian.predict(X_test)
    acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)
      
    st.write(acc_gaussian)

elif option=='Decision Tree':
    decision_tree = DecisionTreeClassifier()
    decision_tree.fit(X_train, Y_train)
    Y_pred = decision_tree.predict(X_test)
    acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
          
    st.write(acc_decision_tree)

else:
    random_forest = RandomForestClassifier(n_estimators = 100)
    random_forest.fit(X_train, Y_train)
    Y_pred = random_forest.predict(X_test)
    acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
             
    st.write( acc_random_forest)