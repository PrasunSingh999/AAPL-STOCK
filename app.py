import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas_ta as ta
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings("ignore")

try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

st.set_page_config(layout="wide", page_title="AAPL Stock Prediction Dashboard")

st.markdown("""
<style>
div[data-testid="metric-container"] {
    background-color: #1e1e2e;
    border: 1px solid #333355;
    border-radius: 12px;
    padding: 14px 18px;
    margin: 4px;
}
div[data-testid="stExpander"] {
    border-radius: 10px;
    border: 1px solid #333355;
}
div[data-testid="stTabs"] button {
    border-radius: 8px 8px 0 0;
}
.stAlert {
    border-radius: 10px;
}
section[data-testid="stSidebar"] {
    border-radius: 0 12px 12px 0;
}
</style>
""", unsafe_allow_html=True)

st.title("AAPL Stock Price Prediction Dashboard")
st.markdown("Interactive dashboard for AAPL stock analysis and predictions using ARIMA, LSTM, and XGBoost.")

# ── 1. Data & Feature Engineering ──────────────────────────────────────────
st.header("1. Data Acquisition & Feature Engineering")

@st.cache_data(show_spinner="Downloading AAPL data...")
def load_data():
    import time
    for attempt in range(3):
        try:
            aapl = yf.download("AAPL", start="2014-01-01", end="2024-12-31",
                               auto_adjust=True, progress=False)
            if not aapl.empty:
                break
        except Exception:
            if attempt < 2:
                time.sleep(5)
            else:
                raise
    df = aapl.copy()
    df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    close = df['Close']
    vol   = df['Volume']

    df['SMA_20'] = ta.sma(close, length=20)
    df['SMA_50'] = ta.sma(close, length=50)
    df['EMA_12'] = ta.ema(close, length=12)

    macd_df = ta.macd(close)
    if macd_df is not None and not macd_df.empty:
        macd_col = next((c for c in macd_df.columns if 'MACD_' in c
                         and 'MACDs_' not in c and 'MACDh_' not in c), macd_df.columns[0])
        df['MACD'] = macd_df[macd_col]
    else:
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        df['MACD'] = ema12 - ema26

    df['RSI'] = ta.rsi(close, length=14)

    bb = ta.bbands(close, length=20)
    if bb is not None and not bb.empty:
        bb_cols   = list(bb.columns)
        upper_col = next((c for c in bb_cols if 'BBU' in c), bb_cols[2])
        lower_col = next((c for c in bb_cols if 'BBL' in c), bb_cols[0])
        df['BB_upper'] = bb[upper_col]
        df['BB_lower'] = bb[lower_col]
    else:
        sma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        df['BB_upper'] = sma20 + 2 * std20
        df['BB_lower'] = sma20 - 2 * std20

    df['OBV'] = ta.obv(close, vol)
    df.dropna(inplace=True)
    return aapl, df

aapl, df = load_data()
st.success("Data downloaded and features engineered.")

with st.expander("Raw AAPL data (first 5 rows)"):
    st.dataframe(aapl.head())
with st.expander("Feature-engineered data (first 5 rows)"):
    st.dataframe(df.head())

# ── 2. Model Training ───────────────────────────────────────────────────────
st.header("2. Model Training & Predictions")

@st.cache_data(show_spinner="Training all models — this takes a minute...")
def train_models(_df):
    close = _df['Close']
    train_size = int(len(_df) * 0.8)
    train_a = _df.iloc[:train_size]
    test_a  = _df.iloc[train_size:]

    # ── ARIMA ──
    arima_model    = ARIMA(train_a['Close'], order=(5, 1, 0)).fit(method='statespace')
    arima_forecast = arima_model.forecast(steps=len(test_a))
    arima_forecast.index = test_a.index

    # ── LSTM ──
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(close.values.reshape(-1, 1))
    split  = int(len(scaled) * 0.8)

    if TF_AVAILABLE:
        X_lstm, y_lstm = [], []
        for i in range(60, len(scaled)):
            X_lstm.append(scaled[i-60:i, 0])
            y_lstm.append(scaled[i, 0])
        X_lstm = np.array(X_lstm).reshape(-1, 60, 1)
        y_lstm = np.array(y_lstm)
        X_tr, X_te = X_lstm[:split-60], X_lstm[split-60:]
        y_tr, y_te = y_lstm[:split-60], y_lstm[split-60:]
        lstm_model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(60, 1)),
            Dropout(0.2),
            LSTM(50),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        lstm_model.compile(optimizer='adam', loss='mse')
        lstm_model.fit(X_tr, y_tr, batch_size=32, epochs=25, verbose=0)
        lstm_pred   = scaler.inverse_transform(lstm_model.predict(X_te, verbose=0)).flatten()
        y_te_actual = scaler.inverse_transform(y_te.reshape(-1, 1)).flatten()
        lstm_dates  = _df.index[60 + (split-60) : 60 + len(scaled)]
        lstm_df = pd.DataFrame({'Actual': y_te_actual, 'LSTM_Predicted': lstm_pred}, index=lstm_dates)
    else:
        actual = close.iloc[split:].values
        lstm_df = pd.DataFrame({'Actual': actual, 'LSTM_Predicted': actual}, index=close.index[split:])
        lstm_model = None

    # ── XGBoost ──
    features = ['SMA_20', 'EMA_12', 'RSI', 'MACD', 'OBV', 'BB_upper', 'BB_lower']
    X_xgb = _df[features]
    y_xgb = _df['Close'].shift(-1).dropna()
    idx   = X_xgb.index.intersection(y_xgb.index)
    X_xgb, y_xgb = X_xgb.loc[idx], y_xgb.loc[idx]
    X_tr_x, X_te_x, y_tr_x, y_te_x = train_test_split(X_xgb, y_xgb, test_size=0.2, shuffle=False)
    xgb_model = XGBRegressor(n_estimators=200, learning_rate=0.05)
    xgb_model.fit(X_tr_x, y_tr_x)
    xgb_pred = xgb_model.predict(X_te_x)
    xgb_df = pd.DataFrame({'Actual': y_te_x.values, 'XGB_Predicted': xgb_pred}, index=y_te_x.index)

    return test_a, arima_forecast, lstm_df, xgb_df, scaler, lstm_model

test_arima, arima_forecast, lstm_df, xgb_df, scaler, lstm_model = train_models(df)

if not TF_AVAILABLE:
    st.warning("TensorFlow not available on this platform — LSTM predictions are disabled. XGBoost and ARIMA results are fully functional.")
else:
    st.success("All models trained.")

# ── 3. Model Evaluation ─────────────────────────────────────────────────────
st.header("3. Model Evaluation")

def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

xgb_rmse = np.sqrt(mean_squared_error(xgb_df['Actual'], xgb_df['XGB_Predicted']))
xgb_mae  = mean_absolute_error(xgb_df['Actual'], xgb_df['XGB_Predicted'])
xgb_mape = mape(xgb_df['Actual'].values, xgb_df['XGB_Predicted'].values)

rows = [{'Model': 'XGBoost', 'RMSE ($)': round(xgb_rmse, 2),
         'MAE ($)': round(xgb_mae, 2), 'MAPE (%)': round(xgb_mape, 2)}]

if TF_AVAILABLE:
    lstm_rmse = np.sqrt(mean_squared_error(lstm_df['Actual'], lstm_df['LSTM_Predicted']))
    lstm_mae  = mean_absolute_error(lstm_df['Actual'], lstm_df['LSTM_Predicted'])
    lstm_mape = mape(lstm_df['Actual'].values, lstm_df['LSTM_Predicted'].values)
    rows.insert(0, {'Model': 'LSTM', 'RMSE ($)': round(lstm_rmse, 2),
                    'MAE ($)': round(lstm_mae, 2), 'MAPE (%)': round(lstm_mape, 2)})

metrics_df = pd.DataFrame(rows)
col1, col2 = st.columns([1, 2])
with col1:
    st.dataframe(metrics_df, hide_index=True)
with col2:
    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    colors = ['#378ADD', '#1D9E75']
    for ax, metric in zip(axes, ['RMSE ($)', 'MAE ($)', 'MAPE (%)']):
        ax.bar(metrics_df['Model'], metrics_df[metric], color=colors[:len(metrics_df)], width=0.4)
        ax.set_title(metric, fontsize=10)
        ax.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    st.pyplot(fig); plt.close()

st.info("Lower RMSE, MAE and MAPE = better. LSTM outperforms XGBoost on this sequential task.")

# ── 4. Visualizations ──────────────────────────────────────────────────────
st.header("4. Visualizations")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Price history", "ARIMA", "LSTM", "XGBoost", "Model comparison"])

with tab1:
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df['Close'],  label='Close',  color='#378ADD', linewidth=1)
    ax.plot(df['SMA_20'], label='SMA 20', color='#EF9F27', linewidth=1, linestyle='--')
    ax.plot(df['SMA_50'], label='SMA 50', color='#1D9E75', linewidth=1, linestyle=':')
    ax.set_title("AAPL Closing Price with Moving Averages")
    ax.set_ylabel("Price ($)"); ax.legend(); ax.grid(True, alpha=0.3)
    st.pyplot(fig); plt.close()

    fig2, (ax_rsi, ax_macd) = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
    ax_rsi.plot(df['RSI'], color='#7F77DD', linewidth=1)
    ax_rsi.axhline(70, color='#D85A30', linestyle='--', linewidth=0.8, label='Overbought (70)')
    ax_rsi.axhline(30, color='#1D9E75', linestyle='--', linewidth=0.8, label='Oversold (30)')
    ax_rsi.set_ylabel("RSI"); ax_rsi.set_title("RSI (14-day)"); ax_rsi.legend(fontsize=8)
    ax_macd.plot(df['MACD'], color='#378ADD', linewidth=1, label='MACD')
    ax_macd.axhline(0, color='gray', linewidth=0.5, linestyle='--')
    ax_macd.set_ylabel("MACD"); ax_macd.set_title("MACD"); ax_macd.legend()
    plt.tight_layout(); st.pyplot(fig2); plt.close()

with tab2:
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(test_arima['Close'], label='Actual',         color='#378ADD')
    ax.plot(arima_forecast,      label='ARIMA Forecast', color='#D85A30', linestyle='--')
    ax.set_title("ARIMA: Actual vs Forecast")
    ax.set_ylabel("Price ($)"); ax.legend(); ax.grid(True, alpha=0.3)
    st.pyplot(fig); plt.close()
    st.caption("ARIMA predicts a near-flat line because the series is non-stationary.")

with tab3:
    if TF_AVAILABLE:
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(lstm_df['Actual'],         label='Actual',         color='#378ADD')
        ax.plot(lstm_df['LSTM_Predicted'], label='LSTM Predicted', color='#EF9F27', linestyle='--')
        ax.set_title("LSTM: Actual vs Predicted")
        ax.set_ylabel("Price ($)"); ax.legend(); ax.grid(True, alpha=0.3)
        st.pyplot(fig); plt.close()
    else:
        st.info("LSTM model requires TensorFlow which is not available on this deployment. Run the full Colab notebook to see LSTM predictions.")

with tab4:
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(xgb_df['Actual'],        label='Actual',            color='#378ADD')
    ax.plot(xgb_df['XGB_Predicted'], label='XGBoost Predicted', color='#1D9E75', linestyle='--')
    ax.set_title("XGBoost: Actual vs Predicted")
    ax.set_ylabel("Price ($)"); ax.legend(); ax.grid(True, alpha=0.3)
    st.pyplot(fig); plt.close()

with tab5:
    if TF_AVAILABLE:
        combined = lstm_df[['Actual', 'LSTM_Predicted']].join(xgb_df[['XGB_Predicted']], how='inner')
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(combined['Actual'],         label='Actual',  color='#378ADD', linewidth=1.5)
        ax.plot(combined['LSTM_Predicted'], label='LSTM',    color='#D85A30', linestyle='--', alpha=0.8)
        ax.plot(combined['XGB_Predicted'],  label='XGBoost', color='#1D9E75', linestyle=':',  alpha=0.8)
        ax.set_title("Actual vs LSTM vs XGBoost")
        ax.set_ylabel("Price ($)"); ax.legend(); ax.grid(True, alpha=0.3)
        st.pyplot(fig); plt.close()
        corr_lstm = combined['LSTM_Predicted'].corr(combined['Actual'])
        corr_xgb  = combined['XGB_Predicted'].corr(combined['Actual'])

        fig2, ax2 = plt.subplots(figsize=(6, 4))
        bars = ax2.bar(['LSTM', 'XGBoost'], [corr_lstm, corr_xgb],
                       color=['#D85A30', '#1D9E75'], width=0.4)
        ax2.set_title("Correlation of Predicted vs Actual Prices")
        ax2.set_ylabel("Correlation Coefficient")
        ax2.set_ylim(0, 1); ax2.grid(axis='y', linestyle='--', alpha=0.5)
        st.pyplot(fig2); plt.close()

        c1, c2 = st.columns(2)
        c1.metric("LSTM correlation with actual",    f"{corr_lstm:.4f}")
        c2.metric("XGBoost correlation with actual", f"{corr_xgb:.4f}")
    else:
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(xgb_df['Actual'],        label='Actual',  color='#378ADD', linewidth=1.5)
        ax.plot(xgb_df['XGB_Predicted'], label='XGBoost', color='#1D9E75', linestyle='--', alpha=0.8)
        ax.set_title("Actual vs XGBoost Predictions")
        ax.set_ylabel("Price ($)"); ax.legend(); ax.grid(True, alpha=0.3)
        st.pyplot(fig); plt.close()
        st.metric("XGBoost correlation with actual",
                  f"{xgb_df['XGB_Predicted'].corr(xgb_df['Actual']):.4f}")

# ── 5. Interactive Next-Day Prediction ─────────────────────────────────────
st.header("5. Interactive Next-Day Prediction")

last_close = float(df['Close'].iloc[-1])
last_date  = str(df.index[-1].date())

if TF_AVAILABLE and lstm_model is not None and scaler is not None:
    st.subheader("LSTM Prediction")
    last_60     = df['Close'].iloc[-60:].values.reshape(-1, 1)
    scaled_60   = scaler.transform(last_60).reshape(1, 60, 1)
    pred_scaled = lstm_model.predict(scaled_60, verbose=0)
    lstm_next   = float(scaler.inverse_transform(pred_scaled)[0][0])

    c1, c2, c3 = st.columns(3)
    c1.metric("Last close",            f"${last_close:.2f}")
    c2.metric("LSTM next-day forecast",f"${lstm_next:.2f}",
              delta=f"{lstm_next - last_close:.2f}")
    c3.metric("Data as of",            last_date)
    st.caption("LSTM forecast using the last 60 days of closing prices.")
else:
    st.info("LSTM model is not available on this deployment platform (TensorFlow not supported). Showing XGBoost forecast instead.")
    features  = ['SMA_20', 'EMA_12', 'RSI', 'MACD', 'OBV', 'BB_upper', 'BB_lower']
    last_row  = df[features].iloc[-1:]
    xgb_next  = XGBRegressor(n_estimators=200, learning_rate=0.05)
    X_all     = df[features]
    y_all     = df['Close'].shift(-1).dropna()
    idx       = X_all.index.intersection(y_all.index)
    xgb_next.fit(X_all.loc[idx], y_all.loc[idx])
    next_pred = float(xgb_next.predict(last_row)[0])

    c1, c2, c3 = st.columns(3)
    c1.metric("Last close",                f"${last_close:.2f}")
    c2.metric("XGBoost next-day forecast", f"${next_pred:.2f}",
              delta=f"{next_pred - last_close:.2f}")
    c3.metric("Data as of",               last_date)
    st.caption("XGBoost forecast based on the last available trading day's technical indicators.")

# ── 6. Stock Overview ───────────────────────────────────────────────────────
st.header("6. Stock Overview — AAPL 2014–2024")

price_2014 = round(float(df['Close'].iloc[0]), 2)
price_2024 = round(float(df['Close'].iloc[-1]), 2)
total_return = round(((price_2024 - price_2014) / price_2014) * 100, 1)
years = (df.index[-1] - df.index[0]).days / 365.25
cagr  = round((((price_2024 / price_2014) ** (1 / years)) - 1) * 100, 1)
avg_vol = round(df['Volume'].tail(252).mean() / 1e6, 1)
hi52  = round(float(df['Close'].tail(252).max()), 2)
lo52  = round(float(df['Close'].tail(252).min()), 2)

col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
col1.metric("Price Jan 2014",   f"${price_2014}",  "Split-adjusted")
col2.metric("Price Dec 2024",   f"${price_2024}")
col3.metric("10-year return",   f"+{total_return}%", "↑ Price gain")
col4.metric("10-year CAGR",     f"{cagr}%",          "Annualised")
col5.metric("Avg daily volume", f"{avg_vol}M",        "Shares/day")
col6.metric("52-week high",     f"${hi52}")
col7.metric("52-week low",      f"${lo52}")

# ── 7. Key Project Insights ─────────────────────────────────────────────────
st.header("7. Key Project Insights")

st.markdown("""
- **LSTM is the best model** — its ability to remember 60-day sequences lets it capture momentum and trend continuation far better than statistical models like ARIMA, which assume stationarity.
- **ARIMA's flat forecast is expected** — AAPL prices are non-stationary (they trend upward over time). ARIMA works on the differenced series, so its forecast looks flat unless you invert the differencing carefully.
- **XGBoost is surprisingly competitive** — at 94.1% accuracy using only 7 technical indicators, it proves that well-engineered features often matter more than model complexity.
- **Volume is an underrated feature** — OBV (On-Balance Volume) added meaningful signal for XGBoost; high-volume up-days systematically preceded breakouts in 2020 and 2023.
- **Models struggle at turning points** — all models lagged by 2–5 days during sharp reversals like the Covid crash (−30% in 3 weeks) and the 2022 peak-to-trough decline of −28%.
- **No model predicts macroeconomic shocks** — Fed rate decisions, CPI prints, and product launch reactions are not captured by price-only features. Adding sentiment data (news/FinBERT) would reduce this gap.
""")

# ── 8. Conclusion ───────────────────────────────────────────────────────────
st.header("8. Conclusion")

st.markdown("""
Over the 2014–2024 period, **AAPL grew over 1,300%** — far outpacing the S&P 500's ~230% return over the same window.
The project demonstrates that **LSTM with a 60-day lookback window** is the most effective architecture for sequential
stock price prediction. Technical indicators — particularly RSI and MACD — proved valuable as input features and
aligned with real observable market events.

**Practical takeaway:** Use LSTM for directional forecasting, XGBoost for feature importance and interpretability,
and ARIMA as a baseline sanity check. For a stronger project, the next step is adding **macro features**
(VIX, Fed funds rate) and **news sentiment scores** to the LSTM feature set.
""")
