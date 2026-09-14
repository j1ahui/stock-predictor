import pandas as pd

def compare_models(rfr_metrics, linear_metrics, lstm_metrics, ridge_metrics):
    """
    
    """
    comparison = pd.DataFrame({
        "Model": ["Random Forest Regression", "Linear Regression", "LSTM", "Ridge Regression"],
        "MAE": [rfr_metrics["MAE"], linear_metrics["MAE"], lstm_metrics["MAE"], ridge_metrics["MAE"]],
        "RMSE": [rfr_metrics["RMSE"], linear_metrics["RMSE"], lstm_metrics["RMSE"], ridge_metrics["RMSE"]],
        "R2": [rfr_metrics["R2"], linear_metrics["R2"], lstm_metrics["R2"], ridge_metrics["R2"]],

    })


    return comparison