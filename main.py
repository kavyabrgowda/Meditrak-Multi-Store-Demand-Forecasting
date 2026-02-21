import os
import logging
import pandas as pd

from src.data_loader import load_data
from src.preprocessing import clean_data
from src.feature_engineering import create_features
from src.pipeline import run_pipeline
from src.inventory_optimizer import generate_reorder_plan
from src.config import (
    PROCESSED_DATA_PATH,
    PREDICTION_OUTPUT,
    EVALUATION_REPORT
)

logging.basicConfig(level=logging.INFO)


def ensure_directories():
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)


def main():

    print("\n Starting Meditrak Store-wise Forecasting...\n")

    ensure_directories()

    df = load_data(
        "data/raw/sales.csv",
        "data/raw/calendar.csv",
        "data/raw/items.csv"
    )

    df = clean_data(df)
    df = create_features(df)
    df = df.dropna()

    df.to_csv(PROCESSED_DATA_PATH, index=False)

    feature_cols = [
        'store_encoded',
        'item_encoded',
        'day_of_week',
        'month',
        'week_of_year',
        'is_weekend',
        'is_holiday',
        'promotion_flag',
        'trend_index',
        'lag_7',
        'rolling_mean',
        'volatility_index'
    ]

    all_predictions = []
    evaluation_results = []

    # TRAIN MODEL FOR EACH STORE SEPARATELY
    for store in df['store_id_original'].unique():

        print(f"Training model for Store {store}...")

        store_df = df[df['store_id_original'] == store]

        X_test, y_test, predictions, mae, rmse, r2 = run_pipeline(
            store_df,
            feature_cols
        )

        reorder_plan = generate_reorder_plan(X_test.copy(), predictions)

        reorder_plan['store_id'] = store
        reorder_plan['item_id'] = store_df.loc[X_test.index, 'item_id_original'].values

        all_predictions.append(reorder_plan)

        evaluation_results.append({
            "store_id": store,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2
        })

        print(f"Store {store} -> MAE: {round(mae,2)}, RMSE: {round(rmse,2)}, R2: {round(r2,2)}")

    final_predictions = pd.concat(all_predictions, ignore_index=True)
    final_predictions.to_csv(PREDICTION_OUTPUT, index=False)

    evaluation_df = pd.DataFrame(evaluation_results)
    evaluation_df.to_csv("outputs/store_evaluation.csv", index=False)

    print("\n Store-wise forecasting completed successfully!\n")


if __name__ == "__main__":
    main()
