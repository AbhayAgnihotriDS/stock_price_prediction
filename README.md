# 📈 AI Stock Prediction Dashboard

An advanced stock market analysis and prediction system that combines deep learning (LSTM) with technical indicators to generate intelligent buy/sell signals and visualize market trends.

---

## 🚀 Features

- 🔮 LSTM-based stock price prediction
- 📊 Interactive candlestick charts
- 📉 Technical indicators:
  - EMA (20 & 50)
  - RSI
  - MACD
- 💡 AI-driven Buy/Sell signals
- 🔄 Dynamic stock selection (AAPL, TSLA, etc.)
- ⚡ Real-time data visualization
- 🎯 Clean and modern UI (inspired by trading platforms)

---

## 🧠 How It Works

1. Historical stock data is collected using APIs
2. Data is preprocessed and scaled
3. LSTM model analyzes past 60 timesteps
4. Model predicts future stock price
5. Indicators confirm market trend
6. System generates BUY / SELL / HOLD signals
7. Results are displayed on an interactive chart

---

## 🏗️ Tech Stack

- **Frontend:** HTML, CSS, JavaScript
- **Charting:** Lightweight Charts
- **Backend:** Flask (Python)
- **Machine Learning:** TensorFlow / Keras (LSTM)
- **Data Processing:** Pandas, NumPy, Scikit-learn

---

## 📊 Model Details

- Model: Long Short-Term Memory (LSTM)
- Input: Multi-feature time series (Close, EMA, RSI, MACD, Volume)
- Output: Next-day stock price prediction
- Sequence Length: 60 timesteps

---

## ⚙️ Installation

```bash
git clone https://github.com/your-username/repo-name.git
cd repo-name
pip install -r requirements.txt
python app.py
