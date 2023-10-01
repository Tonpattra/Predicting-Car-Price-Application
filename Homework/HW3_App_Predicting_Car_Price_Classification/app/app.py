from flask import Flask, request, render_template
import pickle
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
# import cloudpickle
# feature scaling helps improve reach convergence faster
scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)

with open ("standard.pkl", 'rb') as file:
    standard = pickle.load(file)
app = Flask(__name__)
filename = 'price_predict.model'
loaded_model = pickle.load(open(filename, 'rb'))

# print(os.listdir())

file_path = "bestmodel.pkl"
with open(file_path, "rb") as file_1:
    new_model = pickle.load(file_1)

file_path_classification = "../best_classification.pkl"
with open(file_path_classification, "rb") as file_2:
    classification_model = pickle.load(file_2)

# import cloudpickle

# Attempting to load a pickled object from a file
# try:
#     with open("bestmodel.pkl", 'rb') as file_1:
#         new_model = pickle.load(file_1)
# except Exception as e:
#     print(f"Error loading pickled object: {e}")

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')
#textbox1 = power
#textbox2 = engine
#textbox3 = years

def conv(inp, model, typr) :
    # variable = [ i for i, j in locals().items() if j == model][0]
    # print(variable)
    # if variable == "new_model" :
    if typr == "regression" :
        try : 

            ans = standard.transform([inp])
            intercept = np.ones((ans.shape[0], 1))
            inter   = np.concatenate((intercept, ans), axis=1)
            print(inter)
            pric = np.exp(model.predict(inter))
            return pric.item()
    # else :
        except :

            ans = standard.transform([inp])
            pric = np.exp(model.predict(ans))
            return pric.item()
    else :
        ans = standard.transform([inp])
        intercept = np.ones((ans.shape[0], 1)) 
        inter   = np.concatenate((intercept, ans), axis=1)
        print(inter.shape)
        a = model.predict(inter)
        return np.argmax(a, axis=1).item()
        
@app.route('/receive_data', methods=['POST'])
def receive_data():
    textbox1_data = request.form.get('textbox1')
    if not(textbox1_data) :
        textbox1_data = 82
    textbox2_data = request.form.get('textbox2')
    if not(textbox2_data) :
        textbox2_data = 1248
    textbox3_data = request.form.get('textbox3')
    if not(textbox3_data) :
        textbox3_data = 2017

    # print(type(new_model))
    result = [float(textbox1_data), float(textbox2_data), float(textbox3_data)]
    # X_train = scaler.fit_transform(result)
    load_data = conv(result, loaded_model, 'regression')
    new_pred = conv(result, new_model, 'regression')
    classification_pred = conv(result, classification_model, 'classification')

    # print(load_data)
    # print(new_pred)

    # Render the response.html template and pass the data to it
    return render_template('/response.html', 
                           textbox1_data= load_data, textbox2_data= new_pred, textbox3_data= classification_pred
                           )
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)

