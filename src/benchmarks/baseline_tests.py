from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.forecasting.theta import ThetaModel
import warnings

warnings.filterwarnings("ignore")

# Optional: TBATS
try:
    from tbats import TBATS
    tbats_available = True
except ImportError:
    tbats_available = False

# Optional: Fourier terms for DHR-ARIMA
try:
    from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess
    dhr_available = True
except ImportError:
    dhr_available = False

# --- Configuration ---

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAIN_CSV = PROJECT_ROOT / "data" / "Training Data" / "bitcoin_2010-07-29_2025-04-25_train.csv"
TEST_CSV = PROJECT_ROOT / "data" / "Testing Data" / "bitcoin_2010-07-29_2025-04-25_test.csv"

test_index = 120  # <- CHOOSE the index of the return you want to predict (e.g. 120 means predict test_log_returns[120])

# --- Load Data ---

train_df = pd.read_csv(TRAIN_CSV, parse_dates=["Date"])
test_df  = pd.read_csv(TEST_CSV,  parse_dates=["Date"])
train_prices = train_df["Close"].astype(float).values
test_prices  = test_df["Close"].astype(float).values

train_log_returns = np.diff(np.log(train_prices))
test_log_returns  = np.diff(np.log(test_prices))
test_dates        = test_df["Date"].iloc[1:].reset_index(drop=True)

# --- Select True Value and History ---

if test_index < 5:
    raise ValueError("Choose a test_index >= 5 for models like OLS and Theta.")

true_log_return = test_log_returns[test_index]
history_returns = test_log_returns[:test_index]
forecast_date   = test_dates[test_index]

print(f"Forecasting 1-step return at index {test_index} → {forecast_date.date()}")
print(f"True log-return: {true_log_return:.6f}\n")

# --- Forecasts ---

results = {}

# 1. SES
try:
    model = SimpleExpSmoothing(history_returns).fit()
    pred = model.forecast(1)[0]
    results["SES"] = pred
except Exception:
    results["SES"] = np.nan

# 2. Theta
try:
    model = ThetaModel(history_returns).fit()
    pred = model.forecast(1)[0]
    results["Theta"] = pred
except Exception:
    results["Theta"] = np.nan

# 3. ETS
try:
    model = ExponentialSmoothing(history_returns, trend='add', seasonal=None).fit()
    pred = model.forecast(1)[0]
    results["ETS"] = pred
except Exception:
    results["ETS"] = np.nan

# 4. ARIMA(1,0,0)
try:
    model = ARIMA(history_returns, order=(1, 0, 0)).fit()
    pred = model.forecast(1)[0]
    results["ARIMA"] = pred
except Exception:
    results["ARIMA"] = np.nan

# 5. TBATS
if tbats_available:
    try:
        estimator = TBATS(seasonal_periods=[7])
        model = estimator.fit(history_returns)
        pred = model.forecast(steps=1)[0]
        results["TBATS"] = pred
    except Exception:
        results["TBATS"] = np.nan

# 6. DHR-ARIMA
if dhr_available:
    try:
        idx_range = pd.date_range(start='2000-01-01', periods=len(history_returns), freq='D')
        y_series = pd.Series(history_returns, index=idx_range)
        fourier = CalendarFourier(freq='A', order=2)
        dp = DeterministicProcess(index=idx_range, constant=True, order=1, additional_terms=[fourier], drop=True)
        X = dp.in_sample()
        model = SARIMAX(y_series, exog=X, order=(1, 0, 0)).fit(disp=False)
        X_fore = dp.out_of_sample(steps=1)
        pred = model.forecast(steps=1, exog=X_fore)[0]
        results["DHR_ARIMA"] = pred
    except Exception:
        results["DHR_ARIMA"] = np.nan

# --- Metrics & Display ---

mae_list = []
rmse_list = []
models = []
preds = []

print("Model Forecasts vs True:")
for model, pred in results.items():
    if np.isnan(pred):
        continue
    mae = abs(pred - true_log_return)
    rmse = np.sqrt((pred - true_log_return) ** 2)
    print(f"{model:<10} → Predicted: {pred:.6f} | MAE: {mae:.6f} | RMSE: {rmse:.6f}")
    models.append(model)
    preds.append(pred)
    mae_list.append(mae)
    rmse_list.append(rmse)

# --- Plotting ---

plt.figure(figsize=(10, 5))
plt.axhline(y=true_log_return, color='black', linestyle='--', label='True Log Return')
for model, pred in zip(models, preds):
    plt.scatter(model, pred, label=f"{model}", s=100)

plt.title(f"Forecasts for Test Index {test_index} ({forecast_date.date()})")
plt.ylabel("Predicted Log Return")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.grid(True)
plt.show()
