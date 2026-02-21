from sklearn.model_selection import train_test_split
from .config import TEST_SIZE, RANDOM_STATE
from .model import train_model
from .evaluator import evaluate

def run_pipeline(df, feature_cols):

    X = df[feature_cols]
    y = df['units_sold']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        shuffle=False,
        random_state=RANDOM_STATE
    )

    model = train_model(X_train, y_train)

    predictions = model.predict(X_test)

    mae, rmse, r2 = evaluate(y_test, predictions)

    return X_test, y_test, predictions, mae, rmse, r2
