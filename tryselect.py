import json

from flask import Flask,request,render_template,url_for,jsonify,session
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib as joblib
import os

model= joblib.load("model_pickle")


kkk = Flask(__name__)

@kkk.route('/')
def index():
    return render_template('jhj.html')

@kkk.route('/')
@kkk.route("/api",methods=['POST'])
def api():
    output = request.get_json()
 # This is the output that was stored in the JSON within the browser

    result = json.loads(output) #this converts the json output to a python dictionary


    a = []
    i = 0
    while i < 134:
        a.append(0)
        i += 1

        if i == 134:
            break
    f= result[0]
    b = result[1]
    c= result[2]
    d = result[3]
    e = result[4]
    a[f] = 1
    a[b] = 1
    a[c] = 1
    a[d] = 1
    a[e] = 1
    input_data_as_numpy_array = np.asarray(a)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    prediction = model.predict(input_data_reshaped)

    d = dict(enumerate(prediction.flatten(), 1))
    print(d)

    #this shows the json converted as a python dictionary
    return jsonify({"result": d})

@kkk.route("/test", methods=['POST','GET'])
def test():

    output = request.get_json()
 # This is the output that was stored in the JSON within the browser

    result = json.loads(output) #this converts the json output to a python dictionary


    a = []
    i = 0
    while i < 134:
        a.append(0)
        i += 1

        if i == 134:
            break
    f= result[0]
    b = result[1]
    c= result[2]
    d = result[3]
    e = result[4]
    a[f] = 1
    a[b] = 1
    a[c] = 1
    a[d] = 1
    a[e] = 1
    input_data_as_numpy_array = np.asarray(a)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    prediction = model.predict(input_data_reshaped)

    d = dict(enumerate(prediction.flatten(), 1))
    print(d)




    #this shows the json converted as a python dictionary
    return (d)

if __name__ == "__main__":
    kkk.run(debug=True)