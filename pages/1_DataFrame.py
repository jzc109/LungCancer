import streamlit as st
import pandas as pd
import joblib
import pickle
from sklearn.model_selection import train_test_split, learning_curve,GridSearchCV, StratifiedKFold,RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from scipy.stats import randint as sp_randint
import shap
import numpy as np
from sklearn.ensemble import BaggingClassifier

import os
os.chdir("D:/pyweb/ML_WEB")

# 导入数据

@st.cache
def load_data(path):
    df = pd.read_csv(path)
    return df
df = load_data("lung.csv")


# 数据分割
X = df.drop('mal', axis=1)
y = df['mal']

subset =df[['age','sex', 'upleft', 'maxlen', 'F1','F2','F3','F4','F5','F7', 'F8','F9', 'F10','F11','F15']]
X = subset

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,random_state=42)
st.write(X_train.head(10))

# 模型参数设置，已训练
rf = RandomForestClassifier(max_depth=5,min_samples_leaf=4,min_samples_split=2,n_estimators=100, random_state=2023)
# lr = LogisticRegression(random_state=2023,max_iter=10000,solver='liblinear',C=10, penalty='l1')
# svm = SVC(probability=True, random_state=2023,gamma=0.1, kernel='linear')
# ada = AdaBoostClassifier(learning_rate=0.01, n_estimators=200,random_state=2023)
# mlp = MLPClassifier(random_state=2023,max_iter=2000,alpha= 0.001, solver='adam')
# gbm = GradientBoostingClassifier(learning_rate=0.01, max_depth=2, min_samples_leaf=1, min_samples_split=2,
# n_estimators=164, subsample=0.9)
bagging = BaggingClassifier(base_estimator=rf, max_features=0.9, max_samples=0.5, n_estimators=10,random_state=2023)
# 建立stacking模型
# estimators = [('rf', rf), ('svm', svm), ('lr', lr),('ada',ada),('mlpnn',mlp),('gbm',gbm),('bagging',bagging)]
# stack = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(C=2), stack_method='auto')

# 训练各个模型

rf.fit(X_train, y_train)
joblib.dump(rf, "rf.pkl")
# lr.fit(X_train, y_train)
# svm.fit(X_train, y_train)
# ada.fit(X_train, y_train)
# mlp.fit(X_train, y_train)
# gbm.fit(X_train, y_train)
bagging.fit(X_train, y_train)
joblib.dump(bagging, "bagging.pkl")
# stack.fit(X_train, y_train)
st.write('training data  is loaded!')
# pickle.dump(rf_fit, open("rf.pkl",'wb'))
# joblib.dump(lr, "lr.pkl")
# joblib.dump(svm, "svm.pkl")
# joblib.dump(ada, "ada.pkl")
# joblib.dump(mlp, "mlp.pkl")
# joblib.dump(gbm, "gbm.pkl")
# pickle.dump(bagging_fit, open("bagging.pkl",'wb'))

