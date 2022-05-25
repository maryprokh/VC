#FLASK APP FOR RAISIN TYPE PREDICTION MODEL

#import necessary modules
from flask import Flask, request,render_template
from sklearn.preprocessing import StandardScaler
import numpy as np
import pickle
import os 

#create a list of raisin types
CATEGORIES = ['Kecimen', 'Besni']

#create app object
app = Flask(__name__)
#load the model and the standart scaler
model = pickle.load(open('model_raisin.pkl', 'rb'))
sc= pickle.load(open('standart_scale.pkl','rb'))

#create home page - return a created template
@app.route('/')
def home():
    return render_template('index.html')

#create predict page
@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
#create list of features entered in the fields on the app page
    int_features = [float(x) for x in request.form.values()]
#normalize the vector of features
    final_features = sc.transform([int_features])
#predict the result and return it on the app page
    prediction = model.predict(final_features)
    output = CATEGORIES[int(prediction)]
    return render_template('index.html', prediction_text='Raisin type should be {}'.format(output))

#launch the app 
if __name__ == "__main__":
    app.run(host=os.getenv('IP', '0.0.0.0'), 
            port=int(os.getenv('PORT', 4444)))
