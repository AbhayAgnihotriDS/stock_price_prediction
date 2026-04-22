# 📈 AI Stock Prediction Dashboard

An end-to-end intelligent stock market analysis system that combines **Deep Learning (LSTM)** with **technical indicators** to forecast stock prices and generate actionable **BUY / SELL / HOLD signals**.
This project provides an interactive trading-style dashboard for visualizing market trends, model predictions, and strategy signals in real time.

---

# 🚀 Project Overview

The goal of this project is to build a **smart stock analysis platform** that:

* Learns from historical market data
* Predicts future stock prices using LSTM
* Uses technical indicators to validate trends
* Displays insights visually through an interactive chart

It bridges **Machine Learning + Financial Analysis + Web Development** into one unified system.

---

# 🧠 Key Features

## 🔮 1. LSTM-Based Prediction

* Uses Long Short-Term Memory (LSTM) neural network
* Learns sequential dependencies in stock price data
* Predicts the **next-day closing price**

---

## 📊 2. Interactive Chart (TradingView Style)

* Candlestick chart visualization
* Smooth zooming and panning
* Multi-layer data display

---

## 📉 3. Technical Indicators

The system calculates and visualizes:

* **EMA (20 & 50)** → Trend direction
* **RSI (Relative Strength Index)** → Overbought/Oversold levels
* **MACD (Moving Average Convergence Divergence)** → Momentum

---

## 💡 4. AI-Based Trading Signals

Signals are generated using a hybrid approach:

* LSTM prediction (future price direction)
* Indicator confirmation (trend + strength)

### Example Logic:

* ✅ BUY → Prediction ↑ + Uptrend + Strong momentum
* 🔴 SELL → Prediction ↓ + Downtrend + Weak momentum
* ⚪ HOLD → No clear signal

---

## 🔄 5. Dynamic Stock Selection

* Supports multiple stocks (AAPL, TSLA, MSFT, etc.)
* Easily extendable to other markets (NSE/BSE)

---

## ⚡ 6. Real-Time Data Processing (Extendable)

* Designed to support live data APIs
* Can be upgraded to streaming/WebSocket

---

# 🏗️ System Architecture

```text
Frontend (Chart UI)
        ↓
Flask Backend (API Layer)
        ↓
Data Processing + Indicators
        ↓
LSTM Model Prediction
        ↓
Signal Generation Logic
        ↓
Response → Frontend Visualization
```

---

# 📊 Model Details

## 🔹 Model Type

* Long Short-Term Memory (LSTM)

## 🔹 Input Features

* Close Price
* EMA20
* EMA50
* RSI
* MACD
* Volume

## 🔹 Input Shape

```text
(samples, 60 timesteps, multiple features)
```

## 🔹 Output

* Predicted next-day closing price

---

# ⚙️ Installation & Setup

## 1️⃣ Clone Repository

```bash
git clone https://github.com/your-username/repo-name.git
cd repo-name
```

---

## 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 3️⃣ Run Application

```bash
python app.py
```

---

## 4️⃣ Open in Browser

```text
http://127.0.0.1:5000
https://stock-price-prediction-ehlk.onrender.com/
```

---

# 📂 Project Structure

```text
project/
│
├── app.py                 # Flask backend
├── model/
│   ├── lstm_model.h5     # Trained model (ignored in Git)
│   └── scaler.pkl        # Scaler (ignored)
│
├── static/
│   ├── css/
│   ├── js/
│   └── assets/
│
├── templates/
│   └── index.html
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

# 🔬 How the Model Works

1. Collect historical stock data
2. Apply preprocessing & scaling
3. Create sequences of last 60 timesteps
4. Train LSTM to learn patterns
5. Predict next price
6. Combine prediction with indicators
7. Generate trading signals

---

# 📈 Example Workflow

```text
Past Data → LSTM → Prediction
          + Indicators → Signal
          → Chart Visualization
```

---

# ⚠️ Limitations

* Predictions are probabilistic, not guaranteed
* Performance depends on data quality
* Market is affected by external factors (news, events)
* Not suitable for direct financial decision-making

---

# 🚀 Future Improvements

* 📡 Real-time streaming data
* 📅 Multi-day forecasting (next 5–10 days)
* 📊 Backtesting engine
* 🤖 Auto trading bot integration
* 🌐 Cloud deployment (Render / AWS)
* 📱 Mobile responsive UI

---

# 🧪 Possible Enhancements

* Add sentiment analysis (news, Twitter)
* Use Transformer models (advanced DL)
* Add portfolio tracking system
* Risk management module

---

# 📸 Screenshots

> Add screenshots here (VERY IMPORTANT for portfolio)

---

# ⚠️ Disclaimer

This project is built for **educational and research purposes only**.
It should not be used as financial advice or for real trading decisions.

---

# 👨‍💻 Author

**Abhay Agnihotri**

---

# ⭐ Support

If you like this project:

* ⭐ Star the repository
* 🍴 Fork it
* 📢 Share with others

---
