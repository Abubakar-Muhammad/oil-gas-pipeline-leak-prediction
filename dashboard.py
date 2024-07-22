import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
import pandas as pd
import pickle

#LSTM_model = load_model('LSTM_model.keras')
SVM_model = pickle.load(open('svm_model.pkl','rb'))
RF_model = pickle.load(open('rf_model.pkl','rb'))
scaler = pickle.load(open('scaler.pkl','rb'))

def predict_LSTM(X):
    # X = X.reshape((1,1,X.shape[0]))
    pred = LSTM_model.predict(X)
    # return '%.2f'%(pred[0][0])
    return pred[0][0]
    # return pred


def predict_RF(X):
    data = scaler.transform(X)
    pred = RF_model.predict(data)
    # return '%.2f'%(pred
    return pred
    

def predict_SVM(X):
    data = scaler.transform(X)
    pred = SVM_model.predict(data)
    # return '%.2f'%(pred)
    return pred


st.write("Oil Spillage Detection")
st.write('Data Values')

# Wellhead Temp. (C)	Wellhead Press (psi)	MMCFD- gas	BOPD (barrel of oil produced per day)	BWPD (barrel of water produced per day)	BSW - basic solid and water (%)	CO2 mol. (%) @ 25 C & 1 Atm.	Gas Grav.	CR-corrosion defect	Leak
left_column,middle_column,right_column = st.columns(3)
# You can use a column just like st.sidebar:

# 
left_column.text_input('Wellhead Temp. (C)',key='temp')
left_column.text_input('Wellhead Press (psi)',key='pressure')
left_column.text_input('MMCFD- gas',key='mmcfd')
left_column.text_input('BOPD (barrel of oil produced per day)',key='bopd')
left_column.text_input('BWPD (barrel of water produced per day)',key='bwpd')
left_column.text_input('BSW - basic solid and water (%)',key='bsw')
left_column.text_input('CO2 mol. (%) @ 25 C & 1 Atm.',key='co2')
left_column.text_input('Gas Grav.',key='gas')
# left_column.text_input('CR-corrosion defect',key='corr')

def lstmOnclick():
    X = [val for key,val in st.session_state.items() if not key is 'output']
    X = np.array(X,dtype=np.float32)
    X = X.reshape(1,1,-1)
    pred = predict_LSTM(X)
    print(pred)
    right_column.text(f'Output: {"No Leak" if pred<=0 else "Leak" }')
    
def svmOnclick():
    X = [val for key,val in st.session_state.items() if not key is 'output']
    X = np.array(X)
    X = X.reshape(1,-1)
    pred = predict_SVM(X)
    print(pred)
    right_column.text(f'Output: {"No Leak" if pred<=0 else "Leak"}')
    
def rfOnclick():
    X = [val for key,val in st.session_state.items() if not key is 'output']
    X = np.array(X)
    X = X.reshape(1,-1)
    pred = predict_RF(X)
    print(pred)
    right_column.text(f'Output: {"No Leak" if pred<=0 else "Leak"}')
middle_column.button('LSTM Model Predict', on_click=lstmOnclick)
middle_column.button('SVM Model Predict', on_click=svmOnclick)
middle_column.button('RF Model Predict', on_click=rfOnclick)

right_column.text('Output Prediction')
# right_column.text('Output:')
# right_column.text('RF Model Prediction')




