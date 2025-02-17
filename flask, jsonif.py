from flask import Flask, request, jsonify
import pandas as pd
import joblib


app = Flask(__name__)


model = joblib.load("my_model.pkl")
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Receive data from the request
        data = request.get_json()
        if not data or 'features' not in data:
            return jsonify({"error": "Missing 'features' key in request"}), 400
        
        # Convert the data into a DataFrame
        feature_names = ['Age', 'Season', 'Category', 'Preferred Payment Method', 'Previous Purchases', 'Review Rating']
        features = pd.DataFrame([data['features']], columns=feature_names)

        # Predict using the model
        prediction = model.predict(features)[0]

        # Convert the value to a regular float
        prediction = float(prediction)

        return jsonify({"predicted_price": prediction})

    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500
if __name__ == '__main__':
    app.run(debug=True)