import pandas as pd
import numpy as np


# =========================================
# 🔥 ADD ALL INDICATORS
# =========================================
def add_all_indicators(df):
    df = df.copy()

    # ✅ FIX: Ensure Close is always a Series
    close = df["Close"]
    if hasattr(close, "columns"):  # if DataFrame
        close = close.iloc[:, 0]

    volume = df["Volume"]
    if hasattr(volume, "columns"):
        volume = volume.iloc[:, 0]

    # -----------------------------
    # 📊 EMA (Trend)
    # -----------------------------
    df["EMA_20"] = close.ewm(span=20, adjust=False).mean()
    df["EMA_50"] = close.ewm(span=50, adjust=False).mean()

    # -----------------------------
    # 📉 RSI
    # -----------------------------
    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()

    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))

    # -----------------------------
    # 📈 MACD
    # -----------------------------
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()

    df["MACD"] = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]

    # -----------------------------
    # 📊 Bollinger Bands (FIXED)
    # -----------------------------
    df["BB_Middle"] = close.rolling(window=20).mean()
    std = close.rolling(window=20).std()   # ✅ FIX HERE

    df["BB_Upper"] = df["BB_Middle"] + (2 * std)
    df["BB_Lower"] = df["BB_Middle"] - (2 * std)

    # -----------------------------
    # 📦 Volume
    # -----------------------------
    df["Volume_MA"] = volume.rolling(window=20).mean()

    # -----------------------------
    # 📊 VWAP
    # -----------------------------
    df["VWAP"] = (close * volume).cumsum() / volume.cumsum()

    # -----------------------------
    # 🔥 BUY / SELL SIGNAL
    # -----------------------------
    df["Signal"] = 0

    df.loc[
        (df["MACD"] > df["MACD_Signal"]) &
        (df["RSI"] < 70) &
        (df["EMA_20"] > df["EMA_50"]),
        "Signal"
    ] = 1

    df.loc[
        (df["MACD"] < df["MACD_Signal"]) &
        (df["RSI"] > 30) &
        (df["EMA_20"] < df["EMA_50"]),
        "Signal"
    ] = -1

    df["Trade"] = df["Signal"].map({
        1: "BUY",
        -1: "SELL",
        0: "HOLD"
    })

    return df


# =========================================
# 🔥 LSTM FEATURE SELECTION (IMPORTANT)
# =========================================
def get_model_features(df):
    """
    MUST match training features (input_size = 5)
    """
    return df[[
        "Close",
        "RSI",
        "MACD",
        "Volume",
        "EMA_20"
    ]].values


# =========================================
# 📊 CANDLESTICK DATA (FOR FRONTEND)
# =========================================
def get_candlestick_data(df):
    candles = []

    open_ = df["Open"].squeeze()
    high_ = df["High"].squeeze()
    low_ = df["Low"].squeeze()
    close_ = df["Close"].squeeze()

    for i in range(len(df)):
        candles.append({
            "time": df.index[i].strftime('%Y-%m-%d'),
            "open": float(open_.iloc[i]),
            "high": float(high_.iloc[i]),
            "low": float(low_.iloc[i]),
            "close": float(close_.iloc[i])
        })

    return candles


# =========================================
# 🚀 FINAL FRONTEND DATA PREPARATION
# =========================================
def prepare_frontend_data(df):
    df = df.copy()

    # ✅ Fix multi-column issue
    def safe_series(col):
        data = df[col]
        if hasattr(data, "columns"):
            data = data.iloc[:, 0]
        return data.fillna(0)   # ✅ remove NaN

    return {
        # Candlestick data
        "candles": get_candlestick_data(df),

        # EMA
        "ema20": safe_series("EMA_20").tolist(),
        "ema50": safe_series("EMA_50").tolist(),

        # MACD
        "macd": safe_series("MACD").tolist(),
        "macd_signal": safe_series("MACD_Signal").tolist(),

        # RSI
        "rsi": safe_series("RSI").tolist(),

        # Volume
        "volume": safe_series("Volume").tolist(),

        # Bollinger Bands
        "bb_upper": safe_series("BB_Upper").tolist(),
        "bb_lower": safe_series("BB_Lower").tolist(),

        # Signals
        "trade": df["Trade"].fillna("HOLD").tolist()
    }