import os
import io
import base64
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression

# Determine base directory and locate dataset safely for GitHub and cross-platform usage
base_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
csv_path = os.path.join(base_dir, 'data', 'delhi weather data.csv')
if not os.path.exists(csv_path):
    csv_path = os.path.join(base_dir, 'delhi weather data.csv')

if not os.path.exists(csv_path):
    raise FileNotFoundError("Dataset 'delhi weather data.csv' not found in project root or 'data' folder!")

# Load dataset and clean column names
df = pd.read_csv(csv_path, encoding='latin1')
df.columns = df.columns.str.strip()

# Data Preparation and Timestamp conversion
df['timestamp'] = pd.to_datetime(df['time'], dayfirst=True)
df = df.sort_values('timestamp')
df = df.drop_duplicates()

# Drop rows with missing crucial values
df = df.dropna(
    subset=['temp (°C)', 'Relative Humidity (%)', 'Total Precipitation (mm)',
            'Average Wind Speed (km/h)', 'Sea-Level Air Pressure (hPa)']
)
df = df.reset_index(drop=True)

# Create temporal features
df['year'] = df['timestamp'].dt.year
df['month'] = df['timestamp'].dt.month
df['hour'] = df['timestamp'].dt.hour

# Calculate core KPIs
avg_temp = df['temp (°C)'].mean()
min_temp = df['temp (°C)'].min()
max_temp = df['temp (°C)'].max()
avg_humidity = df['Relative Humidity (%)'].mean()
total_rain = df['Total Precipitation (mm)'].sum()

print(f'Average Temperature: {avg_temp:.2f} °C')
print(f'Maximum Temperature: {max_temp:.2f} °C')
print(f'Minimum Temperature: {min_temp:.2f} °C')
print(f'Average Humidity: {avg_humidity:.2f}%')
print(f'Total Precipitation: {total_rain:.2f} mm')

# Helper function to convert matplotlib figure to Base64 string for HTML embedding
def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_str

plots = {}

# 1. Daily Temperature Trend
daily_temp = df.set_index('timestamp')['temp (°C)'].resample('D').mean()
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(daily_temp.index, daily_temp.values, color='#2563eb', linewidth=1.2)
ax.set_title('Daily Temperature Trend', fontsize=14, fontweight='bold', pad=10)
ax.set_xlabel('Date', fontsize=11)
ax.set_ylabel('Temperature (°C)', fontsize=11)
plt.xticks(rotation=45)
plt.grid(True, linestyle='--', alpha=0.3)
plots['chart1'] = fig_to_base64(fig)

# 2. Average Temperature by Month
monthly_temp = df.groupby('month')['temp (°C)'].mean()
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(monthly_temp.index, monthly_temp.values, marker='o', color='#059669', linewidth=2, markersize=6)
ax.set_title('Average Temperature by Month', fontsize=14, fontweight='bold', pad=10)
ax.set_xlabel('Month', fontsize=11)
ax.set_ylabel('Temperature (°C)', fontsize=11)
ax.set_xticks(range(1, 13))
plt.grid(True, linestyle='--', alpha=0.3)
plots['chart2'] = fig_to_base64(fig)

# 3. Temperature Distribution Histogram
fig, ax = plt.subplots(figsize=(10, 4))
ax.hist(df['temp (°C)'], bins=30, color='#d97706', edgecolor='black', alpha=0.85)
ax.set_title('Temperature Distribution', fontsize=14, fontweight='bold', pad=10)
ax.set_xlabel('Temperature (°C)', fontsize=11)
ax.set_ylabel('Frequency', fontsize=11)
plt.grid(True, linestyle='--', alpha=0.3)
plots['chart3'] = fig_to_base64(fig)

# 4. Temperature vs Humidity Scatter Plot
fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(df['temp (°C)'], df['Relative Humidity (%)'], alpha=0.3, color='#dc2626', s=15)
ax.set_title('Relation Between Temperature and Humidity', fontsize=14, fontweight='bold', pad=10)
ax.set_xlabel('Temperature (°C)', fontsize=11)
ax.set_ylabel('Humidity (%)', fontsize=11)
plt.grid(True, linestyle='--', alpha=0.2)
plots['chart4'] = fig_to_base64(fig)

# 5. Top Weather Conditions Bar Chart
weather_counts = df['Weather Condition'].value_counts().head(10)
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(weather_counts.index, weather_counts.values, color='#7c3aed')
ax.set_title('Top Weather Conditions', fontsize=14, fontweight='bold', pad=10)
ax.set_xlabel('Weather Condition', fontsize=11)
ax.set_ylabel('Number of Observations', fontsize=11)
plt.xticks(rotation=30, ha='right')
plt.grid(True, linestyle='--', alpha=0.3)
plots['chart5'] = fig_to_base64(fig)

# Anomaly Detection (Isolation Forest)
features = ['temp (°C)', 'Relative Humidity (%)', 'Total Precipitation (mm)',
            'Average Wind Speed (km/h)', 'Sea-Level Air Pressure (hPa)']
X = df[features]
model = IsolationForest(contamination=0.01, random_state=42)
df['anomaly'] = model.fit_predict(X)

anomalies = df[df['anomaly'] == -1]
anomaly_percent = (len(anomalies) / len(df)) * 100

print(f'Detected anomalies: {len(anomalies):,}')
print(f'Anomaly percentage: {anomaly_percent:.2f}%')

# 6. Temperature Anomalies Chart
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(df['timestamp'], df['temp (°C)'], alpha=0.6, color='#4b5563', label='Temperature', linewidth=1)
ax.scatter(anomalies['timestamp'], anomalies['temp (°C)'], s=25, color='#ef4444', label='Anomaly', zorder=5)
ax.set_title('Temperature Anomalies Detection', fontsize=14, fontweight='bold', pad=10)
ax.set_xlabel('Date', fontsize=11)
ax.set_ylabel('Temperature (°C)', fontsize=11)
ax.legend()
plt.xticks(rotation=45)
plt.grid(True, linestyle='--', alpha=0.3)
plots['chart6'] = fig_to_base64(fig)

# Linear Regression Trend
daily_temp_trend = df.set_index('timestamp').resample('D')['temp (°C)'].mean().dropna()
trend_df = daily_temp_trend.reset_index()
trend_df['time_numeric'] = (trend_df['timestamp'] - trend_df['timestamp'].min()).dt.days

X_trend = trend_df[['time_numeric']]
y_trend = trend_df['temp (°C)']
trend_model = LinearRegression()
trend_model.fit(X_trend, y_trend)

daily_slope = trend_model.coef_[0]
annual_trend = daily_slope * 365
print(f'Estimated Annual Trend: {annual_trend:.4f} °C/year')

# Final Summary DataFrame
final_summary = pd.DataFrame({
    'Metric': [
        'Average Temperature', 'Minimum Temperature', 'Maximum Temperature',
        'Average Humidity', 'Total Precipitation', 'Anomaly Percentage', 'Annual Temperature Trend'
    ],
    'Value': [
        f"{avg_temp:.2f} °C", f"{min_temp:.2f} °C", f"{max_temp:.2f} °C",
        f"{avg_humidity:.2f}%", f"{total_rain:.2f} mm", f"{anomaly_percent:.2f}%", f"{annual_trend:.4f} °C/year"
    ]
})

# Generate HTML Report
html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Delhi Weather Analysis & Anomaly Detection Dashboard</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; background-color: #f8fafc; color: #1e293b; margin: 0; padding: 20px; }}
        .container {{ max-width: 1100px; margin: 0 auto; background: #ffffff; padding: 40px; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.05); }}
        h1 {{ text-align: center; color: #0f172a; margin-bottom: 5px; font-size: 28px; }}
        .subtitle {{ text-align: center; color: #64748b; margin-bottom: 35px; font-size: 15px; }}
        .kpi-container {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 20px; margin-bottom: 40px; }}
        .kpi-card {{ background: #f1f5f9; padding: 20px; border-radius: 8px; text-align: center; border-left: 5px solid #2563eb; }}
        .kpi-title {{ font-size: 13px; color: #475569; text-transform: uppercase; font-weight: 600; margin-bottom: 8px; }}
        .kpi-value {{ font-size: 20px; font-weight: 700; color: #0f172a; }}
        h2 {{ border-bottom: 2px solid #e2e8f0; padding-bottom: 8px; margin-top: 45px; color: #1e293b; font-size: 20px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0 40px 0; }}
        th, td {{ padding: 12px 16px; border: 1px solid #e2e8f0; text-align: left; }}
        th {{ background-color: #f8fafc; font-weight: 600; color: #334155; }}
        tr:nth-child(even) {{ background-color: #f8fafc; }}
        .chart-box {{ text-align: center; margin: 30px 0; background: #ffffff; padding: 15px; border: 1px solid #e2e8f0; border-radius: 8px; }}
        .chart-box img {{ max-width: 100%; height: auto; border-radius: 6px; }}
        .footer {{ text-align: center; margin-top: 50px; font-size: 13px; color: #94a3b8; border-top: 1px solid #e2e8f0; padding-top: 20px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Delhi Weather Analysis & Anomaly Detection Dashboard</h1>
        <div class="subtitle">Comprehensive Exploratory Data Analysis, Machine Learning Anomaly Detection, and Trend Forecasting</div>
        
        <h2>📌 Key Performance Indicators (KPIs)</h2>
        <div class="kpi-container">
            <div class="kpi-card">
                <div class="kpi-title">Avg Temperature</div>
                <div class="kpi-value">{avg_temp:.2f} °C</div>
            </div>
            <div class="kpi-card" style="border-left-color: #059669;">
                <div class="kpi-title">Min Temperature</div>
                <div class="kpi-value">{min_temp:.2f} °C</div>
            </div>
            <div class="kpi-card" style="border-left-color: #dc2626;">
                <div class="kpi-title">Max Temperature</div>
                <div class="kpi-value">{max_temp:.2f} °C</div>
            </div>
            <div class="kpi-card" style="border-left-color: #d97706;">
                <div class="kpi-title">Avg Humidity</div>
                <div class="kpi-value">{avg_humidity:.2f}%</div>
            </div>
            <div class="kpi-card" style="border-left-color: #7c3aed;">
                <div class="kpi-title">Total Precipitation</div>
                <div class="kpi-value">{total_rain:.2f} mm</div>
            </div>
        </div>

        <h2>📋 Summary Metrics Table</h2>
        <table>
            <thead>
                <tr><th>Metric</th><th>Value</th></tr>
            </thead>
            <tbody>
                {"".join([f"<tr><td>{row['Metric']}</td><td><b>{row['Value']}</b></td></tr>" for _, row in final_summary.iterrows()])}
            </tbody>
        </table>

        <h2>📊 Analytical Charts & Visualizations</h2>
        
        <div class="chart-box">
            <h3>Daily Temperature Trend</h3>
            <img src="data:image/png;base64,{plots['chart1']}" alt="Daily Temperature Trend">
        </div>

        <div class="chart-box">
            <h3>Average Temperature by Month</h3>
            <img src="data:image/png;base64,{plots['chart2']}" alt="Average Temperature by Month">
        </div>

        <div class="chart-box">
            <h3>Temperature Distribution</h3>
            <img src="data:image/png;base64,{plots['chart3']}" alt="Temperature Distribution">
        </div>

        <div class="chart-box">
            <h3>Relation Between Temperature and Humidity</h3>
            <img src="data:image/png;base64,{plots['chart4']}" alt="Temperature vs Humidity">
        </div>

        <div class="chart-box">
            <h3>Top Weather Conditions</h3>
            <img src="data:image/png;base64,{plots['chart5']}" alt="Top Weather Conditions">
        </div>

        <div class="chart-box">
            <h3>Temperature Anomalies Detection (Isolation Forest)</h3>
            <img src="data:image/png;base64,{plots['chart6']}" alt="Temperature Anomalies">
        </div>

        <div class="footer">
            Generated automatically via Python &bull; Weather Data Analysis Project &bull; Standard Portfolio Edition
        </div>
    </div>
</body>
</html>
"""

report_filename = "Weather_Report.html"
with open(report_filename, "w", encoding="utf-8") as f:
    f.write(html_content)

print(f"✅ Single-page HTML report successfully created: {report_filename}")

# Automatically open in the default web browser
import webbrowser
webbrowser.open(os.path.abspath(report_filename))
