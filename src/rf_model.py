from sklearn.ensemble import RandomForestClassifier     # AI classification algo imported from py lib used for ml 
from sklearn.model_selection import train_test_split    
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score             # evaluation functions
from sklearn.dummy import DummyClassifier               # .dummy is submodule/package 

import pandas as pd
import numpy as np  
import joblib, os

# train = teaches AI. test = evals AI

FEATURES = [                        # input features (col names). gives model multiple indicators describing current state of stock
    "MA_10",
    "MA_50",
    "Daily_Return",
    "Volume_Ratio",
    "Volatility",
    "Momentum_5",
    "Momentum_10",
    "Dist_MA_10",
    "Dist_MA_50",
    "RSI",
    "MACD",
]

def train_model(df: pd.DataFrame, target_column: str, model_path: str) -> tuple[RandomForestClassifier, dict, pd.DataFrame, np.ndarray, np.ndarray, pd.DataFrame]:                 # ndarray = N dimensional array

    df = df.dropna().copy()                     # drops missing values 

    X = df[FEATURES]                            # X = inputs in ML ("from df, select [item] in features"). X = NEW DATAFRAME !!!!

    y = df[target_column]                       # y = outputs (contains correct answers/labels)

    X_train, X_test, y_train, y_test = train_test_split(            # out of sample testing (how well does model perform on data it has never seen before)
        X,
        y,
        test_size=0.2,      # 80% training data, 20% testing data 
        shuffle=False       # order matters for market/time series data
    )

    baseline = DummyClassifier(strategy="most_frequent")                 # creating object (baseline - creates a very simple model that doesnt actually learn meaningful relationships between features and stock movements). most_frequent = always predict whichever class appeared most frequently
    baseline.fit(X_train, y_train)
    baseline_predictions = baseline.predict(X_test)                      # baseline_predictions returns an array of predictions [1, 1, 1 ..]
    baseline_accuracy = accuracy_score(y_test, baseline_predictions)

    model = RandomForestClassifier(n_estimators=200, random_state=67)    # random forest = collection of many decision trees. each tree makes a prediction (up or down). forest uses majority vote 

    model.fit(X_train, y_train)                                          # rf builds decision trees. tree learns decision tree rules from training data (features + the answers -> then identifies patterns to separate the two classes)

    predictions = model.predict(X_test)                                  # predicting unseen data 
    probabilities = model.predict_proba(X_test)                          # [[0.27, 0.73]] - 27% down, 73% up

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),                 # calc accuracy. compares real answers (y_test) and forest predictions. measures what percentage of predictions were correct
        "precision": precision_score(y_test, predictions),               # measures when model predicted up, how often was it actually up?
        "recall": recall_score(y_test, predictions),                     # measures when stock was up, how many did model identify correctly?
        "f1": f1_score(y_test, predictions),                             # combines precision and recall into one score  
        "roc_auc": roc_auc_score(y_test, probabilities[:, 1]),           # evaluates how well the model can distinguish between two classes across different probability thresholds
    
    }

    improvement = (metrics["accuracy"] - baseline_accuracy)

    metrics["baseline_accuracy"] = baseline_accuracy
    metrics["baseline_improvement"] = improvement

    latest = df.iloc[-1:]                      # models prediction for stock as of right now
    X_latest = latest[FEATURES]

    # os.makedirs("models", exist_ok=True)                # creating a folder, saving trained model to my computer (prevents retraining)
    # joblib.dump(model, "models/random_forest.pkl")      # takes model and save to file (in string)

    model_data = {
        "model": model,
        "metrics": metrics,
        "accuracy": metrics["accuracy"],
        "X_test": X_test,
        "predictions": predictions,
        "probabilities": probabilities,

    }

    joblib.dump(model_data, model_path)

    return model, metrics, X_test, predictions, probabilities, X_latest