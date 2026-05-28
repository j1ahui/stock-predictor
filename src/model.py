from sklearn.ensemble import RandomForestClassifier     # AI classification algo imported from py lib used for ml 
from sklearn.model_selection import train_test_split    
from sklearn.metrics import accuracy_score      # measures prediction accuracy 

# train = teaches AI. test = evals AI

def train_model(df):

    df = df.dropna()    # drops missing values 

    features = [        # input features (col names). equiv to df[["MA_10", "MA_50", "Daily_Return"]]
        "MA_10",
        "MA_50",
        "Daily_Return"
    ]

    X = df[features]    # X = inputs in ML ("from df, select [item] in features")

    y = df["Target"]    # y = outputs (contains correct answers/labels)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,      # 80% training data, 20% testing data 
        shuffle=False       # order matters for market/time series data
    )

    model = RandomForestClassifier()    # random forest = collection of many decision trees. each tree makes a prediction (up or down). forest votes on final answer 

    model.fit(X_train, y_train)     # training 

    predictions = model.predict(X_test)     # AI tries predicting unseen data 

    accuracy = accuracy_score(y_test, predictions)      # calc accuracy. compares real answers (y_test) and AI predictions (predictions)

    return model, accuracy