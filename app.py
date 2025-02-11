from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import pickle

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
with open('titanic_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # Get input features from form
    pclass = int(request.form['pclass'])
    sex = 1 if request.form['sex'] == 'male' else 0
    age = float(request.form['age'])
    sibsp = int(request.form['sibsp'])
    parch = int(request.form['parch'])
    fare = float(request.form['fare'])
    embarked = 0 if request.form['embarked'] == 'C' else (1 if request.form['embarked'] == 'Q' else 2)

    # Create a DataFrame from the input values
    input_data = pd.DataFrame([[pclass, sex, age, sibsp, parch, fare, embarked]],
                              columns=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'])

    # Make prediction
    prediction = model.predict(input_data)
    result = "Survived" if prediction[0] == 1 else "Did not survive"

    return render_template('index.html', prediction_text=f'The person is predicted to: {result}')

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
