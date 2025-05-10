from flask import Flask, request, jsonify
import datetime
import random

app = Flask(__name__)

@app.route('/mandi-data', methods=['GET'])
def mandi_data():
    api_key = request.args.get("api-key")
    format_type = request.args.get("format", "json")
    state = request.args.get("filters[state.keyword]")
    district = request.args.get("filters[district]")
    market = request.args.get("filters[market]")
    commodity = request.args.get("filters[commodity]")
    variety = request.args.get("filters[variety]", "Standard")
    grade = request.args.get("filters[grade]", "A")

    # Simulated current date
    date = datetime.datetime.now().strftime("%Y-%m-%d")

    # Sample dynamic values to simulate API behavior
    min_price = random.randint(1100, 1400)
    max_price = random.randint(1500, 1800)
    modal_price = random.randint(min_price + 50, max_price - 50)

    response = {
        "state": state,
        "district": district,
        "market": market,
        "commodity": commodity,
        "variety": variety,
        "grade": grade,
        "date": date,
        "unit": "Rs/quintal",
        "min_price": min_price,
        "max_price": max_price,
        "modal_price": modal_price
    }

    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True)
