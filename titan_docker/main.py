#!flask/bin/python

from math import floor
import os
from flask import Flask, request, jsonify
import json
import pandas as pd
from sklearn import linear_model
from sklearn import datasets
import pickle
import numpy as np
import sklearn
import sklearn.datasets
import sklearn.ensemble
import sklearn.model_selection
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model

gen_data = pd.read_csv('gen_data.csv')

gen_data = pd.get_dummies(gen_data, columns=['fuel'], drop_first=True)
X = gen_data.drop(['co2'], axis=1)
y = gen_data[['co2']]

y2 = np.ravel(y)
X_train, X_test, y_train, y_test = train_test_split(X, y2, test_size=0.30, random_state=1)

# creating and saving the model
rf = sklearn.ensemble.RandomForestRegressor(n_estimators=500)
rf.fit(X_train, y_train)
pickle.dump(rf, open('gen_model_rf', 'wb'))

# creating the Linear Regression Model
#regression_model = LinearRegression()
#regression_model.fit(X_train, y_train)

poly = PolynomialFeatures(degree=3, interaction_only=True)




X_train2 = poly.fit_transform(X_train)
X_test2 = poly.fit_transform(X_test)

poly_lr = linear_model.LinearRegression()

poly_lr = poly_lr.fit(X_train2, y_train)
pickle.dump(poly_lr, open('gen_model_poly_lr', 'wb'))




app = Flask(__name__)

@app.route('/isAlive')
def index():
    return "Our Server is LIVE! "


@app.route('/predict', methods=['POST'])

def get_prediction():
    # Works only for a single sample
    data_in_py_dictionary_format = request.get_json()  # Get data posted as a json    
    
    # to return a group of the key-value
    # pairs in the dictionary
    dict_items = data_in_py_dictionary_format.items()
    
    # Convert object to a list
    dict_as_list = list(dict_items)

    # Convert list to an array
    dict_as_array = np.array(dict_as_list)
    
    # Convert the array to a tuple that we can access the individual indexes easily
    dict_as_tuple = dict_as_array[0][1], dict_as_array[1][1], dict_as_array[2][1], dict_as_array[3][1]
    
    data = np.array(dict_as_tuple)

    model_rf = pickle.load(open('gen_model_rf', 'rb'))

    prediction_rf = model_rf.predict([data])  # runs globally loaded model on the data


    # -- Poly Linear Regression -- #
    
    poly = PolynomialFeatures(degree=3, interaction_only=True)
    X_test3 = poly.fit_transform([data])
    model_lr = pickle.load(open('gen_model_poly_lr', 'rb'))
    prediction_lr = model_lr.predict([X_test3[0]])  # runs globally loaded model on the data
    output = floor((prediction_rf[0] + prediction_lr[0]) / 2)
    
    return str(output)

if __name__ == 'main':
    if os.environ['ENVIRONMENT'] == 'production':
        app.run(port=10, host='0.0.0.0')