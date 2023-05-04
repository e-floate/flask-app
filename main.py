import numpy as np
import flask
import pickle
import os
from sklearn.svm import *

app = flask.Flask(__name__)
port = int(os.getenv("PORT", 9099))
model = pickle.load(open("url_model.pkl","rb")) #load AI model

@app.route('/predict', methods=['POST']) #receives data as a POST request
def predict():

    features = np.array(flask.request.get_json(force=True)['features']) #gets the JSON request
    #if URL host is on whitelist csv, return false
    
    prediction = model.predict([features]) #feeds JSON into the AI model
    
    #if prediction is false, add url to whitelist csv
    return str(prediction) #returns prediction as a Boolean
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=port)