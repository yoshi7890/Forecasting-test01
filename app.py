#importing the necessary libraries#
import streamlit as st
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

#adding a title and description#
st.title("Demand Forecasting App")
st.write("This app allows you to perform demand forecasting using various methods.")

#inputs Spare Parts Demand Data From user#
spare_part_ID = st.text_input("Input Spare Part ID:")
spare_part_name = st.text_input("Input Spare Part Name:")

demand = []
for i in range(1, 6):
    demand_value = st.number_input(f"Input Demand year {i}:", value=0.0)
    demand.append(demand_value)

# ฟังก์ชันคำนวณ MSE
def mean_squared_error(actual, forecast):
    return np.mean((np.array(actual) - np.array(forecast))**2)

# ฟังก์ชันสำหรับการคำนวณด้วยวิธี Croston
def croston_forecast(demand, alpha):
    Y = np.zeros(n)  # ค่าเฉลี่ยความต้องการที่ไม่ใช่ศูนย์
    P = np.zeros(n)  # ค่าเฉลี่ยช่วงเวลาระหว่างความต้องการที่ไม่ใช่ศูนย์
    F = np.zeros(n)  # ค่าพยากรณ์

    Y[0] = demand[0] if demand[0] > 0 else 1
    P[0] = 1
    F[0] = Y[0] / P[0]

    last_non_zero_demand = 0

    for t in range(1, n):
        if demand[t] > 0:
            Y[t] = alpha * demand[t] + (1 - alpha) * Y[t-1]
            P[t] = alpha * (t - last_non_zero_demand) + (1 - alpha) * P[t-1]
            last_non_zero_demand = t
        else:
            Y[t] = Y[t-1]
            P[t] = P[t-1]

        F[t] = Y[t] / P[t]

    next_year_forecast = F[-1]
    return next_year_forecast, Y, P, F

# ฟังก์ชันสำหรับการพยากรณ์แบบ Moving Average (3 Periods)
def moving_average_forecast(demand, period=3):
    forecast = []
    for i in range(len(demand)):
        if i < period:
            forecast.append(np.mean(demand[:i+1]))  # ใช้ค่าเฉลี่ยของข้อมูลที่มีอยู่
        else:
            forecast.append(np.mean(demand[i-period:i]))
    return forecast

# ฟังก์ชันสำหรับการพยากรณ์แบบ Bootstrap
def bootstrap_forecasting(demand, n_iterations=1000):
    n_size = len(demand)
    bootstrap_forecasts = []
    for i in range(n_iterations):
        sample = np.random.choice(demand, size=n_size, replace=True)
        forecast = np.mean(sample)
        bootstrap_forecasts.append(forecast)
    return bootstrap_forecasts

# ฟังก์ชันสำหรับคำนวณค่า ADI และ CV^2
def calculate_adi_cv2(demand):
    non_zero_demands = [d for d in demand if d > 0]
    intervals = np.diff([i for i, d in enumerate(demand) if d > 0], prepend=-1)
    adi = np.mean(intervals)
    mean_demand = np.mean(non_zero_demands)
    cv2 = (np.std(non_zero_demands) / mean_demand) ** 2 if mean_demand != 0 else 0
    return adi, cv2

# ฟังก์ชันสำหรับการจัดประเภทความต้องการ
def classify_demand_pattern(adi, cv2):
    if adi < 1.32 and cv2 < 0.49:
        return "Smooth"
    elif adi >= 1.32 and cv2 >= 0.49:
        return "Lumpy"
    elif adi < 1.32 and cv2 >= 0.49:
        return "Erratic"
    else:
        return "Intermittent"

# ฟังก์ชันสำหรับการวิเคราะห์การแจกแจง
def analyze_distribution(demand):
    distributions = ['norm', 'expon', 'gamma', 'poisson']
    best_fit_name = None
    best_fit_params = None
    best_fit_pvalue = 0

    for dist_name in distributions:
        if dist_name == 'poisson':
            lambda_param = np.mean(demand)
            params = (lambda_param,)
            ks_stat, p_value = stats.kstest(demand, dist_name, args=params)
        else:
            dist = getattr(stats, dist_name)
            params = dist.fit(demand)
            ks_stat, p_value = stats.kstest(demand, dist_name, args=params)

        if p_value > best_fit_pvalue:
            best_fit_name = dist_name
            best_fit_params = params
            best_fit_pvalue = p_value

    return best_fit_name, best_fit_params, best_fit_pvalue

#Add a button that, when clicked, runs forecasting calculations#
if st.button("Run Forecasting"):
    # เริ่มต้นคำนวณการพยากรณ์
    best_alpha_exponential = 0
    min_mse_exponential = float('inf')
    best_forecast_exponential = []

    best_alpha_croston = 0
    min_mse_croston = float('inf')
    best_forecast_croston = []

    st.write(f"Forcasting Result For: {spare_part_ID}")
    st.write("===================================")

    for alpha in np.arange(0.0, 1.1, 0.1):
        # Exponential Smoothing
        forecast_exponential = [demand[0]]
        for t in range(1, len(demand)):
            forecast_exponential.append(alpha * demand[t-1] + (1 - alpha) * forecast_exponential[t-1])

        mse_exponential = mean_squared_error(demand, forecast_exponential)
        if mse_exponential < min_mse_exponential:
            min_mse_exponential = mse_exponential
            best_alpha_exponential = alpha
            best_forecast_exponential = forecast_exponential

        # Croston's Method
        forecast_croston, Y, P, F = croston_forecast(demand, alpha)
        mse_croston = mean_squared_error(demand, F)
        if mse_croston < min_mse_croston:
            min_mse_croston = mse_croston
            best_alpha_croston = alpha
            best_forecast_croston = F

    # Moving Average Forecast (3 Periods)
    forecast_moving_average = moving_average_forecast(demand)
    mse_moving_average = mean_squared_error(demand, forecast_moving_average)

    # Bootstrap Forecasting
    bootstrap_forecasts = bootstrap_forecasting(demand)
    mse_bootstrap = mean_squared_error([np.mean(demand)] * len(bootstrap_forecasts), bootstrap_forecasts)
    best_forecast_bootstrap = np.median(bootstrap_forecasts)

    # Calculate ADI and CV^2
    adi, cv2 = calculate_adi_cv2(demand)

    # Classify Demand Pattern
    demand_pattern = classify_demand_pattern(adi, cv2)

    # Analyze Distribution
    best_fit_name, best_fit_params, best_fit_pvalue = analyze_distribution(demand)

    # Display results
    st.write(f"\nExponential Smoothing:\nBest Alpha: {best_alpha_exponential:.1f}, MSE: {min_mse_exponential:.4f}")
    st.write(f"Best forecast: {best_forecast_exponential[-1]}")

    st.write(f"\nCroston's Method:\nBest Alpha: {best_alpha_croston:.1f}, MSE: {min_mse_croston:.4f}")
    st.write(f"Best forecast: {best_forecast_croston[-1]}")

    st.write(f"\nMoving Average (3 Periods):\nMSE: {mse_moving_average:.4f}")
    st.write(f"Best forecast: {forecast_moving_average[-1]}")

    st.write(f"\nBootstrap Method:\nMSE: {mse_bootstrap:.4f}")
    st.write(f"Best Forecast: {best_forecast_bootstrap}")

    # Compare models and choose the best
    min_mse = min(min_mse_exponential, min_mse_croston, mse_moving_average, mse_bootstrap)
    if min_mse == min_mse_exponential:
        best_method = "Exponential Smoothing"
        best_forecast = best_forecast_exponential
    elif min_mse == min_mse_croston:
        best_method = "Croston's Method"
        best_forecast = best_forecast_croston
    elif min_mse == mse_moving_average:
        best_method = "Moving Average (3 Periods)"
        best_forecast = forecast_moving_average
    else:
        best_method = "Bootstrap"
        best_forecast = [best_forecast_bootstrap]  # Show only the best for Bootstrap

    st.write(f"\nDemand Pattern Classification: {demand_pattern}")
    st.write(f"\nDistribution: {best_fit_name}")
    st.write(f"\nBest Forecasting Method: {best_method}")
    st.write(f"Best Forecast: {best_forecast[-1]}")
    st.write(f"Minimum MSE: {min_mse:.4f}")