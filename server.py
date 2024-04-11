from flask import Flask, render_template, request, redirect, url_for
from sklearn.preprocessing import LabelEncoder
import joblib
import numpy as np

app = Flask(__name__)
loaded_model = joblib.load('lmod.pk1')
le = LabelEncoder()

@app.route('/')
def home():
    return render_template('interface.html')

@app.route('/predict', methods=['POST'])
def predict():
    areaincome = float(request.form['areaincome'])
    houseage = float(request.form['houseage'])
    noofrooms = float(request.form['noofrooms'])
    noofbedrooms = float(request.form['noofbedrooms'])
    areapopulation = float(request.form['areapopulation'])
   

    input_data = np.array([areaincome, houseage,noofrooms, noofbedrooms,areapopulation]).reshape(1, -1)
    predicted_revenue = loaded_model.predict(input_data)
    print(f"input_data: {input_data}")
    
    return redirect(url_for('result', prediction=predicted_revenue[0]))

@app.route('/result')
def result():
    prediction = request.args.get('prediction')
    return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(port=8059)