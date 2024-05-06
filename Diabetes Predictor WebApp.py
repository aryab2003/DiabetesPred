# -*- coding: utf-8 -*-
"""
Created on Mon May  6 20:22:49 2024

@author: HP
"""

import numpy as np
import pickle
import streamlit as st

loaded_model=pickle.load(open('C:/Users/HP/OneDrive/Desktop/DiabetesPred/trained_model.sav','rb'))


def diabetes_prediction(input_data):
    
    input_data=(4,110,92,0,0,37.6,0.191,30)
    input_data_as_numpy_array=np.asarray(input_data)
    input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)

    prediction=loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0]==0):
      return 'The person is not diabetic'
    else:
      return 'The person is diabetic'
  
    
def main():
    
    st.title('Diabetes Predictor')

    
    Pregnancies=st.text_input('Number of Pregnancies')
    Glucose=st.text_input('Glucose level')
    BloodPressure=st.text_input('Blood Pressure Level')
    SkinThickness=st.text_input('Skin Thickness Level')
    Insulin=st.text_input('Insulin Level')
    BMI=st.text_input('BMI value')
    DiabetesPedigreeFunction=st.text_input('Diabetes Pedigree Function value')
    Age=st.text_input('Age of Person')
    
    diagnosis = ''
    
    
    if st.button('Diabetes Test Result'):
        diagnosis = diabetes_prediction([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])
        
        
    st.success(diagnosis)
    
    
if __name__ == '__main__':
    main()