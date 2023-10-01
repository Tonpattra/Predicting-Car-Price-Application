from flask import Flask, request, render_template
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

# feature scaling helps improve reach convergence faster
scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)

with open ("./standard.pkl", 'rb') as file:
    standard = pickle.load(file)
app = Flask(__name__)
filename = './price_predict.model'
loaded_model = pickle.load(open(filename, 'rb'))

@app.route('/')
def index():
    return render_template('index.html')
#textbox1 = power
#textbox2 = engine
#textbox3 = years

def conv(inp) :
    ans = standard.transform([inp])
    pric = np.exp(loaded_model.predict(ans))
    return pric.item()

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

    result = [float(textbox1_data), float(textbox2_data), float(textbox3_data)]
    # X_train = scaler.fit_transform(result)
    load_data = conv(result)


    # Render the response.html template and pass the data to it
    return render_template('response.html', 
                           textbox1_data=load_data)
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)

