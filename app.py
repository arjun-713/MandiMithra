from flask import Flask, request, jsonify
from utils import get_predicted_price

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    crop = data.get('crop')
    region = data.get('region')
    date = data.get('date')

    prediction = get_predicted_price(crop, region, date)
    
    return jsonify({
        'crop': crop,
        'region': region,
        'predicted_price': prediction,
        'status': 'success'
    })

if __name__ == '__main__':
    app.run(debug=True)
