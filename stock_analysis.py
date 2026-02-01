import pymongo
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# -------------------------
# 1. MongoDB Connection
# -------------------------
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["stock_market"]
collection = db["RELIANCE_BSE", "RELIANCE_NSE", "TCS_BSE", "TCS_NSE", "INFy_NSE"][0] 
data = list(collection.find())

if len(data) == 0:
    raise Exception("No data found in MongoDB collection")

# -------------------------
# 2. Convert to DataFrame
# -------------------------
df = pd.DataFrame(data)
df.drop(columns=["_id"], inplace=True)

df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date")

features = ["open", "high", "low", "close", "volume"]
df = df[features]

print("Data Loaded:", df.shape)

# -------------------------
# 3. Scaling
# -------------------------
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df)

# -------------------------
# 4. Create LSTM Dataset
# -------------------------
LOOK_BACK = 60

X, y = [], []
for i in range(LOOK_BACK, len(scaled_data)):
    X.append(scaled_data[i-LOOK_BACK:i])
    y.append(scaled_data[i, 3])  # close price

X = np.array(X)
y = np.array(y)

# -------------------------
# 5. Train-Test Split
# -------------------------
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print("Training Samples:", X_train.shape)
print("Testing Samples:", X_test.shape)

# -------------------------
# 6. Build LSTM Model
# -------------------------
model = Sequential()
model.add(LSTM(64, return_sequences=True, input_shape=(LOOK_BACK, 5)))
model.add(Dropout(0.2))
model.add(LSTM(64))
model.add(Dropout(0.2))
model.add(Dense(1))

model.compile(optimizer="adam", loss="mse")

# -------------------------
# 7. Train Model
# -------------------------
model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=32,
    validation_data=(X_test, y_test)
)
# -------------------------
# 8. Save Model
# -------------------------
model.save("lstm_stock_model.keras")
print("Model saved as lstm_stock_model.keras")
