from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load('golf_model.pkl')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = [request.form['Outlook'], request.form['Temperature'], request.form['Humidity'], request.form['Windy']]
    data = np.array(data).reshape(1, -1)
    prediction = model.predict(data)[0]
    result = "Play" if prediction == 1 else "Don't Play"
    return render_template('result.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
