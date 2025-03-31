# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 17:45:40 2017

@author: NishitP
"""

import pickle
import os

def detecting_fake_news(var):
    """Function to predict whether news is fake or real"""
    model_path = 'final_model.sav'  # Ensure the correct path
    
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found!")
        return
    
    # Load the model
    with open(model_path, 'rb') as file:
        load_model = pickle.load(file)
    
    # Make predictions
    prediction = load_model.predict([var])[0]
    prob = load_model.predict_proba([var])[0][1]
    
    print("The given statement is:", prediction)
    print("The truth probability score is:", prob)

if __name__ == '__main__':
    var = input("Please enter the news text you want to verify: ")
    print("You entered:", var)
    detecting_fake_news(var)

