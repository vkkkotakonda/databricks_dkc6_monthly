# Databricks notebook source
# MAGIC %md
# MAGIC Plan:
# MAGIC 1. Load some data
# MAGIC 2. Run some transformations on the data
# MAGIC 3. Develop a model
# MAGIC 4. Register the model as an artefact
# MAGIC 5. Create a forecast
# MAGIC 6. Save the forecast to the databricks
# MAGIC 7. Orchestrate the process using databricks workflows
# MAGIC 8. version control the code using Git
# MAGIC 9. Iterate till more sophisticated
# MAGIC

# COMMAND ----------

# Databricks notebook source
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import mlflow
import mlflow.sklearn
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# COMMAND ----------

# Load the data
data = spark.read.csv("dbfs:/FileStore/ons/dkc6_monthly.csv", header=True, inferSchema=True)
df = data.toPandas()

# COMMAND ----------

print(df.dtypes)

# COMMAND ----------

# Prepare the data
df['Period'] = pd.to_datetime(df['Period'], format='%Y %b')
df = df.set_index('Period')
df['Value'] = df['Value'].astype(float)

# COMMAND ----------

# Split the data into train and test sets
train = df[:'2022-12-01']
test = df['2023-01-01':]

# COMMAND ----------

# Define the ARIMA model
def create_arima_model(order):
    return ARIMA(train['Value'], order=order)

# COMMAND ----------

# Train the model and make predictions
def train_and_predict(model, steps):
    result = model.fit()
    forecast = result.forecast(steps=steps)
    return result, forecast

# COMMAND ----------

# Evaluate the model
def evaluate_model(actual, predicted):
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    return mse, rmse

# COMMAND ----------


# Set MLflow experiment
mlflow.set_experiment("/Shared/DKC6_Monthly_Forecasting/")

# COMMAND ----------

with mlflow.start_run(run_name="ARIMA_model"):
    # Set model parameters
    order = (1, 1, 1)  # (p, d, q) for ARIMA model
    
    # Log parameters
    mlflow.log_param("p", order[0])
    mlflow.log_param("d", order[1])
    mlflow.log_param("q", order[2])
    
    # Create and train the model
    model = create_arima_model(order)
    result, forecast = train_and_predict(model, steps=len(test) + 18)  # 18 months ahead
    
    # Evaluate the model
    mse, rmse = evaluate_model(test['Value'], forecast[:len(test)])
    
    # Log metrics
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("rmse", rmse)

    # Set tags
    mlflow.set_tag("model", "arima")
    mlflow.set_tag("created_by", "kiran")
    
    # Log the model
    mlflow.sklearn.log_model(result, "arima_model")
    
    # Create forecasts for the next 18 months
    future_dates = pd.date_range(start=df.index[-1] + timedelta(days=1), periods=18, freq='MS')
    future_forecast = pd.Series(forecast[len(test):], index=future_dates)
    
    # Combine historical data and forecasts
    full_series = pd.concat([df['Value'], future_forecast])
    
    # Plot the results
    plt.figure(figsize=(12, 6))
    plt.plot(train.index, train['Value'], label='Historical Data (Train)')
    plt.plot(test.index, test['Value'], label='Historical Data (Test)')
    plt.plot(future_dates, future_forecast, label='Forecast')
    plt.title('Time Series Forecast')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    
    # Save the plot
    plt.savefig("/dbfs/FileStore/time_series_forecast.png")
    mlflow.log_artifact("/dbfs/FileStore/time_series_forecast.png")

# COMMAND ----------

# Print the forecast for the next 18 months
print(future_forecast)

# COMMAND ----------

# Print the forecast for the next 18 months
print(full_series)

# COMMAND ----------

# Combine historical data and forecasts
full_series = pd.concat([df['Value'], future_forecast])

# COMMAND ----------

# Assuming full_series is a pandas DataFrame
file_path = "/dbfs/FileStore/ons/dkc6_monthly_forecast.csv"
full_series.to_csv(file_path, index=True)

# COMMAND ----------

print(full_series.head())
