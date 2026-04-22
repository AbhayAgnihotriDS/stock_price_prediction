from flask import Flask, jsonify, request, render_template
import yfinance as yf
import numpy as np
import torch
import pickle
import os
from datetime import timedelta

# Custom modules
from model import load_model
from indicators import add_all_indicators, prepare_frontend_data, get_model_features

app = Flask(__name__)

# =========================================
# 📁 BASE PATH
# =========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# =========================================
# 🔥 LOAD MODEL + SCALERS
# =========================================
model = load_model(os.path.join(BASE_DIR, "lstm_model.pth"))

scaler_x = pickle.load(open(os.path.join(BASE_DIR, "scaler_x.pkl"), "rb"))
scaler_y = pickle.load(open(os.path.join(BASE_DIR, "saler_y.pkl"), "rb"))


# =========================================
# 🏠 HOME ROUTE
# =========================================
@app.route("/")
def home():
    return render_template("index.html")


# =========================================
# 📊 INDICATORS API (FOR CHART)
# =========================================
@app.route("/indicators")
def indicators():
    try:
        ticker = request.args.get("ticker", "AAPL").upper()

        df = yf.download(ticker, period="6mo", interval="1d")

        if df.empty:
            return jsonify({"error": "No data found"})

        df = add_all_indicators(df)
        df = df.dropna()

        if df.empty:
            return jsonify({"error": "Not enough data after indicators"})

        data = prepare_frontend_data(df)

        return jsonify(data)

    except Exception as e:
        return jsonify({"error": str(e)})


# =========================================
# 🔮 PREDICTION API (FOR OVERLAY)
# =========================================
@app.route("/predict")
def predict():
    try:
        ticker = request.args.get("ticker", "AAPL").upper()

        df = yf.download(ticker, period="6mo", interval="1d")

        if df.empty:
            return jsonify({"error": "No data found"})

        df = add_all_indicators(df)
        df = df.dropna()

        if len(df) < 50:
            return jsonify({"error": "Not enough data for prediction"})

        # Extract features (must match training)
        features = get_model_features(df)

        # Scale
        scaled = scaler_x.transform(features)

        # Sequence
        sequence_length = 50
        seq = scaled[-sequence_length:]
        seq = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)

        # Predict
        with torch.no_grad():
            pred_scaled = model(seq).numpy()

        prediction = scaler_y.inverse_transform(pred_scaled)

        # 🔥 Next date for overlay
        last_date = df.index[-1]
        next_date = (last_date + timedelta(days=1)).strftime('%Y-%m-%d')

        return jsonify({
            "prediction": float(prediction[0][0]),
            "next_date": next_date
        })

    except Exception as e:
        return jsonify({"error": str(e)})


# =========================================
# 📊 MULTI STOCK COMPARISON
# =========================================
@app.route("/multi-stock")
def multi_stock():
    try:
        tickers = request.args.get("tickers", "AAPL,TSLA").upper().split(",")

        result = {}

        for t in tickers:
            df = yf.download(t.strip(), period="1mo", interval="1d")

            if not df.empty:
                result[t] = df["Close"].dropna().tolist()
            else:
                result[t] = []

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)})


# =========================================
# 🔍 SEARCH API
# =========================================
@app.route("/search")
def search():
    query = request.args.get("q", "").upper()

    stocks = [
        "AAPL", "TSLA", "MSFT", "GOOGL", "AMZN",
        "META", "NVDA", "NFLX", "BABA", "INTC"
    ]

    results = [s for s in stocks if query in s]

    return jsonify(results)


# =========================================
# 🚀 RUN SERVER
# =========================================
if __name__ == "__main__":
    app.run(debug=True)