from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)
model = joblib.load('xgb_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = [float(request.form[f'V{i}']) for i in range(1, 29)]
        amount = float(request.form['Amount'])
        input_data = np.array([data + [amount]])
        prediction = model.predict(input_data)

        result = "⚠️ Fraudulent Transaction Detected!" if prediction[0] == 1 else "✅ Legitimate Transaction."
        return render_template('index.html', prediction=result)

    except Exception as e:
        return render_template('index.html', prediction=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
