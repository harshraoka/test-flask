from flask import Flask,request
import pandas as pd
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route("/",methods=["POST"])
def result():
    output = request.get_json()
    rand = output["symptoms"]
    
    
    def prediction(rand):
    # loading dataset         
        url = 'https://github.com/harshraoka/dataset/blob/main/Training.csv?raw=true'
        training = pd.read_csv(url, index_col=0) 
    

    # removing unnesecarry data points
        training.drop('Unnamed: 133', axis=1, inplace=True)

    # test train split 
        X_train = training.drop('prognosis', axis=1)
        y_train = training['prognosis']
        y_train = np.array(y_train).reshape(y_train.shape[0], 1)

    # decision tree
        from sklearn.tree import DecisionTreeClassifier
        tree = DecisionTreeClassifier(random_state=42)
        tree.fit(X_train.values, y_train) #training
        pred1 = tree.predict(rand) # predicting
        return pred1
    
    randlist=[]
    randlist.append(rand)
    str = prediction(randlist)
    s=str[0]
    cal = {}

    cal['disease'] = s
    

    return (cal)

