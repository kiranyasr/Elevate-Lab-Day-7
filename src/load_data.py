import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess(path="data/breast-cancer.csv", target_col="diagnosis"):
    # Load dataset
    data = pd.read_csv(path)

    # If labels are strings (M/B), convert to 0/1
    if data[target_col].dtype == "object":
        data[target_col] = data[target_col].map({"M": 1, "B": 0})

    # Features & Target
    X = data.drop(columns=[target_col])
    y = data[target_col]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test
