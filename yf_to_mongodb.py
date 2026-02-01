import yfinance as yf
from pymongo import MongoClient, errors

# MongoDB connection
client = MongoClient("mongodb://localhost:27017/")
db = client["stock_market"]

def store_stock(symbol, exchange):
    yf_symbol = f"{symbol}.NS" if exchange == "NSE" else f"{symbol}.BO"
    print(f"Fetching {yf_symbol}")

    stock = yf.download(yf_symbol, period="max", progress=False)

    if stock.empty:
        print(f"No data for {yf_symbol}")
        return

    stock.columns = stock.columns.get_level_values(0)
    collection = db[f"{symbol}_{exchange}"]

    collection.create_index([("date", 1)], unique=True)

    data = []
    for row in stock.itertuples():
        data.append({
            "symbol": yf_symbol,
            "exchange": exchange,
            "date": row[0].to_pydatetime(),
            "open": float(row[1]),
            "high": float(row[2]),
            "low": float(row[3]),
            "close": float(row[4]),
            "volume": int(row[5])
        })

    try:
        collection.insert_many(data, ordered=False)
        print(f"Inserted {symbol} ({exchange})")
    except errors.BulkWriteError:
        print(f"Duplicates skipped for {symbol} ({exchange})")

# Example usage
nse_list = ["RELIANCE", "TCS", "INFY"]
bse_list = ["RELIANCE", "TCS"]

for s in nse_list:
    store_stock(s, "NSE")

for s in bse_list:
    store_stock(s, "BSE")
