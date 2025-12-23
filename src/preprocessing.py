import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_features(df, target_column):
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    X = pd.get_dummies(X, drop_first=True)
    return X, y

def scale_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled
