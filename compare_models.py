import numpy as np
import pandas as pd

def compare_models(rf_metrics, linear_metrics, lstm_metrics, ridge_metrics):
    """
    
    """
    comparison = pd.DataFrame({
        "Model": ["Random Forest", "Linear Regression", "LSTM", "Ridge Regression"],
        "MAE": [rf_metrics["MAE"], linear_metrics["MAE"], lstm_metrics["MAE"], ridge_metrics["MAE"]],
        "RMSE": [rf_metrics["RMSE"], linear_metrics["RMSE"], lstm_metrics["RMSE"], ridge_metrics["RMSE"]],
        "R2": [rf_metrics["R2"], linear_metrics["R2"], lstm_metrics["R2"], ridge_metrics["R2"]],

    })


    return comparison