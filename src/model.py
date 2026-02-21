from sklearn.ensemble import RandomForestRegressor
import joblib
from .config import MODEL_PATH

def train_model(X_train, y_train):

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=18,
        min_samples_split=4,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    joblib.dump(model, MODEL_PATH)

    return model

def load_model():
    return joblib.load(MODEL_PATH)
