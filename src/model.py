from sklearn.ensemble import RandomForestClassifier

def create_model(n_estimators=100, random_state=42, max_depth=None, min_samples_split=2):
    model = RandomForestClassifier(n_estimators=n_estimators,
                                   max_depth=max_depth,
                                   min_samples_split=min_samples_split,
                                   random_state=random_state)
    return model

def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model

def predict(model, X):
    return model.predict(X)
