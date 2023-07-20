import streamlit as st
import pickle
import pandas as pd
import joblib

rf = joblib.load("rf.pkl")
bagging = joblib.load("bagging.pkl")


activities=['Random Forest','Bagging']
option=st.sidebar.selectbox('Which model would you like to use?',activities)
st.subheader(option)

age = st.number_input(" Age(Years)", value=0, min_value=0, max_value=120, step=1)
maxlen = st.number_input("Tumor Length", value=0.00, min_value=0.00, max_value=50.00, step=0.01)
upleft = st.radio('upleft', ['Up Left', 'Not Up Left'])
sex = st.selectbox("Select gender:", ["Male", "Female"])

F1 = st.radio('F1', ['YES', 'NO'])
F2 = st.radio('F2', ['YES', 'NO'])
F3 = st.radio('F3', ['YES', 'NO'])
F4 = st.radio('F4', ['YES', 'NO'])
F5 = st.radio('F5', ['YES', 'NO'])
F7 = st.radio('F7', ['YES', 'NO'])
F8 = st.radio('F8', ['YES', 'NO'])
F9 = st.radio('F9', ['YES', 'NO'])
F10 = st.radio('F10', ['YES', 'NO'])
F11 = st.radio('F11', ['YES', 'NO'])
F15 = st.radio('F15', ['YES', 'NO'])
X = pd.DataFrame([[age, sex, upleft, maxlen, F1, F2, F3, F4, F5, F7, F8, F9, F10, F11, F15]])
X = X.replace(["YES", "NO"], [1, 0])
X = X.replace(['Male', 'Female'], [1, 0])
X = X.replace(['Up Left', 'Not Up Left'], [1, 0])

if st.button('Submit'):
        if option=='Random Forest':
            prediction_rf = rf.predict(X)[0]
            st.text(f"This patient is {prediction_rf}")
        elif option=='Bagging':
            prediction_bagging = rf.predict(X)[0]
            st.text(f"This patient is {prediction_bagging}")

