from flask import Flask, request, jsonify, render_template
from arima_model import prepare_data, arima_forecast
from lstm_model import lstm_forecast

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    year = int(request.form['year'])
    model_type = request.form['model']
    data = prepare_data('crime.csv', year)
    categories = data['offense_category_id'].unique()
    result = []
    steps = 30  # Definisikan nilai steps di sini

    for category in categories:
        try:
            if model_type == 'ARIMA':
                category_result = arima_forecast(data, category)
            elif model_type == 'LSTM':
                category_result = lstm_forecast(data, category, steps=steps)  # Pastikan parameter steps dimasukkan

            result.append({
                "category": category,
                "predictions": [{"day": i+1, "prediction": pred} for i, pred in enumerate(category_result)]
            })
        except ValueError as e:
            print(f"Error processing category {category}: {e}")
            result.append({
                "category": category,
                "predictions": [{"day": i+1, "prediction": 0} for i in range(steps)]
            })
    
    return jsonify(result=result)

if __name__ == "__main__":
    app.run(debug=True)
