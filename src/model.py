from sklearn.ensemble import RandomForestClassifier     # AI classification algo imported from py lib used for ml 
from sklearn.model_selection import train_test_split    
from sklearn.metrics import accuracy_score              # measures prediction accuracy 

import pandas as pd
import numpy as np
import joblib, os

# train = teaches AI. test = evals AI

def train_model(df: pd.DataFrame, target_column: str, model_path: str) -> tuple[RandomForestClassifier, float, pd.DataFrame, np.ndarray, np.ndarray, pd.DataFrame]:                 # ndarray = N dimensional array

    df = df.dropna()                    # drops missing values 

    features = [                        # input features (col names). gives model multiple indicators describing current state of stock
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

    X = df[features]    # X = inputs in ML ("from df, select [item] in features"). X = NEW DATAFRAME !!!!

    y = df[target_column]    # y = outputs (contains correct answers/labels)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,      # 80% training data, 20% testing data 
        shuffle=False       # order matters for market/time series data
    )

    model = RandomForestClassifier(n_estimators=200, random_state=67)    # random forest = collection of many decision trees. each tree makes a prediction (up or down). forest uses majority vote 

    model.fit(X_train, y_train)                                          # rf builds decision trees. tree learns decision tree rules from training data (features + the answers -> then identifies patterns to separate the two classes)

    predictions = model.predict(X_test)                                  # predicting unseen data 
    probabilities = model.predict_proba(X_test)                          # [[0.27, 0.73]] - 27% down, 73% up

    accuracy = accuracy_score(y_test, predictions)      # calc accuracy. compares real answers (y_test) and forest predictions

    latest = df.dropna().iloc[-1:]                      # models prediction for stock as of right now
    X_latest = latest[features]

    # os.makedirs("models", exist_ok=True)                # creating a folder, saving trained model to my computer (prevents retraining)
    # joblib.dump(model, "models/random_forest.pkl")      # takes model and save to file (in string)

    model_data = {
        "model": model,
        "accuracy": accuracy,
        "X_test": X_test,
        "predictions": predictions,
        "probabilities": probabilities,

    }

    joblib.dump(model_data, model_path)

    return model, accuracy, X_test, predictions, probabilities, X_latest