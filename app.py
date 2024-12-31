from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load('golf_model.pkl')

@app.route('/')
def home():
    return render_template('home.html', result=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect form data
        outlook = int(request.form['Outlook'])
        temperature = int(request.form['Temperature'])
        humidity = int(request.form['Humidity'])
        windy = int(request.form['Windy'])

        # Prepare input for the model
        features = [[outlook, temperature, humidity, windy]]

        # Predict using the model
        prediction = model.predict(features)

        # Convert prediction to a user-friendly message
        result = "Yes, you can play golf!" if prediction[0] == 1 else "No, you should not play golf."
    except Exception as e:
        result = f"Error: {str(e)}"

    # Render the same page with the result
    return render_template('home.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
