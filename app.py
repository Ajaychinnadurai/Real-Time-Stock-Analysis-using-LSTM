from flask import Flask, render_template, jsonify
import pymongo
import pandas as pd

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

# 📊 Stock analysis
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

if __name__ == "__main__":
    app.run(debug=True)
