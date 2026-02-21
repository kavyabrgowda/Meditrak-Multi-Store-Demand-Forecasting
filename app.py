from flask import Flask, render_template, request, send_file
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import datetime

app = Flask(__name__)

medicine_master = {
    "M101": "Paracetamol",
    "M102": "Amoxicillin",
    "M103": "Cough Syrup",
    "M104": "Vitamin D",
    "M105": "Insulin"
}


def generate_store_data(store_id):
    np.random.seed(hash(store_id) % 10000)

    days = 90
    data = []

    for day in range(days):
        date = datetime.date(2025, 1, 1) + datetime.timedelta(days=day)

        for item_id in medicine_master.keys():

            base = {
                "S1": 50,
                "S2": 70,
                "S3": 40,
                "S4": 90,
                "S5": 30
            }[store_id]

            # Different demand per medicine
            medicine_factor = {
                "M101": 1.0,
                "M102": 1.2,
                "M103": 0.9,
                "M104": 0.7,
                "M105": 1.5
            }[item_id]

            variation = np.random.randint(0, 30)
            trend = day * np.random.uniform(0.1, 0.4)

            units = int((base * medicine_factor) + variation + trend)

            data.append([date, store_id, item_id, units])

    df = pd.DataFrame(data, columns=["Date", "Store", "Item_ID", "Units_Sold"])
    return df


def forecast_demand(df):

    forecast_list = []

    for item in df["Item_ID"].unique():

        item_df = df[df["Item_ID"] == item].copy()
        item_df["Day"] = pd.to_datetime(item_df["Date"]).dt.dayofyear

        X = item_df[["Day"]]
        y = item_df["Units_Sold"]

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)

        future_days = pd.DataFrame({
            "Day": [item_df["Day"].max() + i for i in range(1, 8)]
        })

        predicted = model.predict(future_days).mean()

        # Business logic
        safety_stock = 20
        current_stock = np.random.randint(30, 100)

        reorder_quantity = max(
            round(predicted + safety_stock - current_stock, 2),
            0
        )

        forecast_list.append([
            item,
            round(predicted, 2),
            current_stock,
            safety_stock,
            reorder_quantity
        ])

    forecast_df = pd.DataFrame(
        forecast_list,
        columns=[
            "Item_ID",
            "Predicted_Demand",
            "Current_Stock",
            "Safety_Stock",
            "Reorder_Quantity"
        ]
    )

    forecast_df["Medicine_Name"] = forecast_df["Item_ID"].map(medicine_master)

    forecast_df["Risk"] = np.select(
        [
            forecast_df["Predicted_Demand"] > 90,
            forecast_df["Predicted_Demand"] > 70
        ],
        ["High", "Medium"],
        default="Low"
    )

    return forecast_df


@app.route("/")
def dashboard():

    store = request.args.get("store", "S1")

    df = generate_store_data(store)

    forecast_df = forecast_demand(df)


    daily_sales = (
        df.groupby("Item_ID")["Units_Sold"]
        .mean()
        .reset_index()
    )
    daily_sales["Medicine_Name"] = daily_sales["Item_ID"].map(medicine_master)

    daily_analysis = df.copy()
    daily_analysis["Medicine_Name"] = daily_analysis["Item_ID"].map(medicine_master)

    daily_analysis = daily_analysis.sort_values(["Date", "Item_ID"])


    top_items = (
        df.groupby("Item_ID")["Units_Sold"]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )
    top_items["Medicine_Name"] = top_items["Item_ID"].map(medicine_master)

    high_risk = forecast_df[forecast_df["Risk"] == "High"]

    return render_template(
        "dashboard.html",
        store=store,
        total_days=df["Date"].nunique(),
        items_predicted=len(forecast_df),
        high_risk_count=len(high_risk),
        high_risk_names=", ".join(high_risk["Medicine_Name"].tolist()),
        forecast=forecast_df.to_dict(orient="records"),
        daily_sales=daily_sales.to_dict(orient="records"),
        daily_analysis=daily_analysis.to_dict(orient="records"),  
        top_items=top_items.to_dict(orient="records")
    )



@app.route("/download")
def download():
    store = request.args.get("store", "S1")
    df = generate_store_data(store)
    forecast_df = forecast_demand(df)

    filename = f"forecast_{store}.csv"
    forecast_df.to_csv(filename, index=False)

    return send_file(filename, as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True)
