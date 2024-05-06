# -*- coding: utf-8 -*-
"""
Created on Mon May  6 20:14:25 2024

@author: HP
"""

import numpy as np
import pickle

loaded_model=pickle.load(open('C:/Users/HP/OneDrive/Desktop/DiabetesPred/trained_model.sav','rb'))


input_data=(10,168,74,0,0,38,0.537,34)

input_data_as_numpy_array=np.asarray(input_data)
input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)

std_data=scaler.transform(input_data_reshaped)
print(std_data)
prediction=loaded_model.predict(std_data)
print(prediction)

if (prediction[0]==0):
  print('The person is not diabetic')
else:
  print('The person is diabetic')