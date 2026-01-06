import yfinance as yf
from pymongo import MongoClient, errors
import pandas as pd

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

    collection.create_index(
        [("symbol", 1), ("date", 1)],
        unique=True
    )

    data = []

    for row in stock.itertuples():
        date = row[0]
        data.append({
            "symbol": yf_symbol,
            "exchange": exchange,
            "date": date.to_pydatetime(),
            "open": float(row[1]),
            "high": float(row[2]),
            "low": float(row[3]),
            "close": float(row[4]),
            "volume": int(row[5])
        })

    try:
        collection.insert_many(data, ordered=False)
        print(f"Inserted data for {symbol} ({exchange})")
    except errors.BulkWriteError as e:
        print(f"Duplicates skipped for {symbol} ({exchange})")

# Test
nse_list = ["RELIANCE", "TCS", "INFY"]
bse_list = ["RELIANCE", "TCS"]

for sym in nse_list:
    store_stock(sym, "NSE")

for sym in bse_list:
    store_stock(sym, "BSE")
