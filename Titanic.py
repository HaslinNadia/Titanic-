import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import codecs
import seaborn as sns
import sweetviz as sv
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import pickle

def st_display_sweetviz(report_html,width=1000,height=500):
	report_file = codecs.open(report_html,'r')
	page = report_file.read()
	components.html(page,width=width,height=height,scrolling=True)

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

train.isnull().sum().sort_values(ascending = False)
test.isnull().sum().sort_values(ascending = False)
train = train.drop(['Ticket', 'Cabin'], axis = 1)
test = test.drop(['Ticket', 'Cabin'], axis = 1)

mode = train['Embarked'].dropna().mode()[0]
train['Embarked'].fillna(mode, inplace = True)
median = test['Fare'].dropna().median()
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
combine = combine.drop(['Name','Embarked','Parch','SibSp','Pclass','PassengerId'], axis = 1)

train = combine[:len(train)]
test = combine[len(train):]
train['Survived'] = train['Survived'].astype('int')
test = test.drop('Survived', axis = 1)
X_train = train.drop('Survived', axis = 1)
y_train = train['Survived']
X_test = test.copy()

menu = ["Home","Sweetviz"]
choice = st.sidebar.selectbox("Menu",menu)

if choice == "Home":
  
  html_temp = """
    <div style="background-color:red;padding:10px;border-radius:10px">
    <h1 style="color:white;text-align:center;"> Titanic Dataset Analysis and Prediction</h1>
    </div>
    """
  st.write("""Haslin Nadia Roslan,
  Participant MOHE Data Track, Batch 3,
  2022""")
  components.html(html_temp)
  from PIL import Image
  image = Image.open('titanic.jpg')
  st.image(image, caption='Titanic Cruise', width=690)
  st.title("Titanic Dataset")
  st.dataframe(train)
  st.write("Dataset Description:")
  st.write("Survived = 0 (Passenger DID NOT SURVIVED) | Survived = 1 (Passenger SURVIVED)")
  st.write("Sex = 0 (MALE) | Sex = 1 (FEMALE) ")
  st.title("Modelling")
  option = st.selectbox(
      'Choose one',
      ['Logistic Regression','Support Vector Machines','k-NN','Gausssian Naive Bayes','Decision Tree','Random Forest'])

  if option=='Logistic Regression':
      logreg = LogisticRegression()
      logreg.fit(X_train, y_train)
      Y_pred = logreg.predict(X_test)
      acc_log = round(logreg.score(X_train, y_train) * 100, 2)
      
      st.write("Logistic Regression value = ", acc_log)

  elif option=='Support Vector Machines':
      svc = SVC()
      svc.fit(X_train, y_train)
      Y_pred = svc.predict(X_test)
      acc_svc = round(svc.score(X_train, y_train) * 100, 2)
      
      st.write("SVM value = ",acc_svc)

  elif option=='k-NN':
      knn = KNeighborsClassifier(n_neighbors = 5)
      knn.fit(X_train, y_train)
      Y_pred = knn.predict(X_test)
      acc_knn = round(knn.score(X_train, y_train) * 100, 2)
        
      st.write("k-NN value = ",acc_knn)

  elif option=='Gausssian Naive Bayes':
      gaussian = GaussianNB()
      gaussian.fit(X_train, y_train)
      Y_pred = gaussian.predict(X_test)
      acc_gaussian = round(gaussian.score(X_train, y_train) * 100, 2)
        
      st.write("Gaussian Naive Bayes value = ", acc_gaussian)

  elif option=='Decision Tree':
      decision_tree = DecisionTreeClassifier()
      decision_tree.fit(X_train, y_train)
      Y_pred = decision_tree.predict(X_test)
      acc_decision_tree = round(decision_tree.score(X_train, y_train) * 100, 2)
      
      st.write("Decision tree value = ", acc_decision_tree)

  else:
      random_forest = RandomForestClassifier(n_estimators = 100)
      random_forest.fit(X_train, y_train)
      Y_pred = random_forest.predict(X_test)
      acc_random_forest = round(random_forest.score(X_train, y_train) * 100, 2)
              
      st.write("Random Forest value = ",acc_random_forest)

  random_forest = RandomForestClassifier(n_estimators = 100)
  random_forest.fit(X_train, y_train)
  Y_pred = random_forest.predict(X_test)
  pickle_out = open("randomforest.pkl", "wb") 
  pickle.dump(random_forest, pickle_out) 
  pickle_out.close()

  pickle_in = open('randomforest.pkl', 'rb')
  classifier = pickle.load(pickle_in)

  st.title('Survive or Not Prediction')
  st.write('Please refer dataset above for more details')
  Age = st.number_input("Age (1-80):")
  Sex = st.number_input("Gender male(0) female(1):")
  Fare =  st.number_input("Fare (1-6):")
    
  submit = st.button('Predict')
  if submit:
    prediction = classifier.predict([[Age,Sex,Fare]])
    if prediction == 1:
        st.write('THIS TYPE OF PASSENGER MOST LIKELY WILL SURVIVE')
    else:
        st.write("THIS KIND OF PASSENGER PROBABLY WILL NOT SURVIVE")

else:
  st.title("EDA Overview")
  if st.button("Generate Sweetviz Report"):
      st_display_sweetviz("report1.html")
