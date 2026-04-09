import joblib
from flask import Flask, render_template, request
import numpy as np

app = Flask(__name__)

model = joblib.load('random_forest_model.pkl')



@app.route('/predict', methods=['POST'])
def predict():
    try:
        N = request.form.get('N')
        P = request.form.get('P')
        K = request.form.get('K')
        temperature = request.form.get('temperature')
        humidity = request.form.get('humidity')
        ph = request.form.get('ph')
        rainfall = request.form.get('rainfall')

        features = [float(N), float(P), float(K), float(temperature), 
                    float(humidity), float(ph), float(rainfall)]
        input_data = np.array([features])
        prediction = model.predict(input_data)

        return render_template('index.html', 
            prediction=str(prediction[0]),
            N=N, P=P, K=K, temperature=temperature,
            humidity=humidity, ph=ph, rainfall=rainfall)
    except Exception as e:
        return render_template('index.html', prediction="Error: " + str(e))

@app.route('/')
def welcome():
    return render_template('welcome.html')

@app.route('/predict-page')
def predict_page():
    return render_template('index.html')
if __name__ == '__main__':
    app.run(debug=True)