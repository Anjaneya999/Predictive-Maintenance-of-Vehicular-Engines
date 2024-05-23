from flask import Flask, render_template, request
import pickle
import numpy as np
from quality import predict_quality

cat_model = pickle.load(open('cat_model.pkl', 'rb'))
app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict_emissions():
    manufacturer = request.form.get('Make')
    model = request.form.get('Model')
    v_class = request.form.get('Vehicle Class')
    engine = np.float64(request.form.get('Engine Size (L)'))
    cylinders = int(request.form.get('Cylinders'))
    transmission = request.form.get('Transmission')
    fuel_type = request.form.get('Fuel Type')
    fuel_consumption1 = float(request.form.get('Fuel Consumption City (L/100 km)'))
    fuel_consumption2 = float(request.form.get('Fuel Consumption Hwy (L/100 km)'))
    fuel_consumption3 = float(request.form.get('Fuel Consumption Comb (L/100 km)'))
    fuel_consumption4 = float(request.form.get('Fuel Consumption Comb (mpg)'))

    result = cat_model.predict(np.array([manufacturer, model, v_class, engine, cylinders, transmission,
                                         fuel_type, fuel_consumption1, fuel_consumption2, fuel_consumption3,
                                         fuel_consumption4]).reshape(1, -1))
    classified = predict_quality(v_class, result)

    return render_template('index.html', result=result, classified=classified)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
