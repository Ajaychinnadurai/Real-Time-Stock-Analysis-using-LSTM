from flask import Flask, render_template, jsonify
import pymongo
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow.keras as tf
app = Flask(__name__)

# MongoDB
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["stock_market"]

@app.route("/")
def home():
    return render_template("index.html")

# 🔍 Company suggestions
@app.route("/companies")
def companies():
    names = db.list_collection_names()
    return jsonify(names)

# Stock analysis
@app.route("/stock/<company>")
def stock_data(company):
    collection = db[company]
    data = list(collection.find({}, {"_id": 0}).sort("date", 1))

    if not data:
        return jsonify({"error": "No data found"})

    df = pd.DataFrame(data)

    analysis = {
        "dates": df["date"].astype(str).tolist(),
        "close": df["close"].tolist(),
        "high": df["high"].tolist(),
        "low": df["low"].tolist(),
        "volume": df["volume"].tolist(),
        "max_price": float(df["high"].max()),
        "min_price": float(df["low"].min()),
        "avg_close": float(df["close"].mean())
    }

    return jsonify(analysis)

def model_predict(company):
    model = tf.models.load_model(r"D:\Real Time Stock Analysis using LSTM\lstm_stock_model.keras", custom_objects=None, compile=True)
    scaler = MinMaxScaler(feature_range=(0, 1))
    LOOK_BACK = 60
    features = ["open", "high", "low", "close", "volume"]
    collection = db[company]
    data = list(collection.find().sort("date", 1))

    if len(data) < LOOK_BACK:
        return None

    df = pd.DataFrame(data)
    df = df[features]
    scaled_data = scaler.fit_transform(df)

    last_60_days = scaled_data[-LOOK_BACK:]
    last_60_days = last_60_days.reshape(1, LOOK_BACK, 5)

    predicted_scaled = model.predict(last_60_days)

    dummy = np.zeros((1, 5))
    dummy[0, 3] = predicted_scaled

    predicted_price = scaler.inverse_transform(dummy)[0, 3]
    return round(predicted_price, 2)

@app.route("/predict/<company>")
def predict(company):
    price = model_predict(company)
    if price is None:
        return jsonify({"error": "Not enough data"})
    return jsonify({"tomorrow_price": price})

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)

    from flask import Flask, render_template, jsonify
