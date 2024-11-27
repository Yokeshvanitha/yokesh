from flask import Flask, render_template, request
import numpy as np
import pickle

# Initialize the Flask application
app = Flask(__name__)

# Load your pre-trained model
model = pickle.load(open('trained_model.pkl', 'rb'))

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        try:
            # Collect form data
            age = int(request.form['age'])
            sex = int(request.form['sex'])
            cp = int(request.form['cp'])
            trestbps = int(request.form['trestbps'])
            chol = int(request.form['chol'])
            fbs = int(request.form['fbs'])
            restecg = int(request.form['restecg'])
            thalach = int(request.form['thalach'])
            exang = int(request.form['exang'])
            oldpeak = float(request.form['oldpeak'])
            slope = int(request.form['slope'])

            # Prepare input data for prediction
            input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope]])
            
            # Make prediction
            prediction = model.predict(input_data)

            # Interpret the result
            result = 'Patient Have Heart Disease, Please Consult The Doctor' if prediction[0] == 1 else 'The Patient Normal'
        except Exception as e:
            result = f"Error: {str(e)}"

    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(host="0.0.0.0",port=5000)
