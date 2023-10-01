from flask import Flask, request, render_template
import pickle
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
import cloudpickle
import mlflow
app = Flask(__name__)

# Load the fitted StandardScaler
with open("Homework/HW3_App_Predicting_Car_Price_Classification/app/standard.pkl", 'rb') as file:
    standard = pickle.load(file)

# Load the regression model
filename = 'Homework/HW3_App_Predicting_Car_Price_Classification/app/price_predict.model'
loaded_model = pickle.load(open(filename, 'rb'))

# Load the new_model
file_path = "Homework/HW3_App_Predicting_Car_Price_Classification/app/bestmodel.pkl"
with open(file_path, "rb") as file_1:
    new_model = pickle.load(file_1)

# Load the classification model
file_path_classification = "Homework/HW3_App_Predicting_Car_Price_Classification/best_classification.pkl"
with open(file_path_classification, "rb") as file_2:
    classification_model = pickle.load(file_2)

@app.route('/')
def index():
    return render_template('index.html')

def conv(inp, model, typr):
    try:
        ans = standard.transform([inp])
        intercept = np.ones((ans.shape[0], 1))
        inter = np.concatenate((intercept, ans), axis=1)
        if typr == 'regression':
            pric = np.exp(model.predict(inter))
            return pric.item()
        else:
            a = model.predict(inter)
            return np.argmax(a, axis=1).item()
    except Exception as e:
        # Handle any errors here, such as input data formatting or model prediction issues
        print(f"Error in conv function: {e}")
        return None

@app.route('/receive_data', methods=['POST'])
def receive_data():
    textbox1_data = request.form.get('textbox1')
    textbox2_data = request.form.get('textbox2')
    textbox3_data = request.form.get('textbox3')

    # Convert input data to float, handle missing values if needed
    try:
        textbox1_data = float(textbox1_data) if textbox1_data else 82.0
        textbox2_data = float(textbox2_data) if textbox2_data else 1248.0
        textbox3_data = float(textbox3_data) if textbox3_data else 2017.0
    except ValueError as e:
        print(f"Error converting input data: {e}")
        # Handle the error as needed

    # Perform predictions
    load_data = conv([textbox1_data, textbox2_data, textbox3_data], loaded_model, 'regression')
    new_pred = conv([textbox1_data, textbox2_data, textbox3_data], new_model, 'regression')
    classification_pred = conv([textbox1_data, textbox2_data, textbox3_data], classification_model, 'classification')

    # Render the response.html template and pass the data to it
    return render_template('/response.html', 
                           textbox1_data=load_data, textbox2_data=new_pred, textbox3_data=classification_pred)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)