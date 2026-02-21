import pandas as pd
import numpy as np
import os

np.random.seed()

os.makedirs("data/raw", exist_ok=True)


dates = pd.date_range("2025-01-01", "2025-03-31")

stores = ["S1", "S2", "S3", "S4", "S5"]

items = {
    "M101": ("Paracetamol", 2.5, 45, 3),
    "M102": ("Amoxicillin", 4.0, 35, 4),
    "M103": ("Cough Syrup", 3.5, 28, 3),
    "M104": ("Vitamin D", 15.0, 20, 2),
    "M105": ("Insulin", 6.0, 25, 2),
    "M106": ("Aspirin", 2.0, 30, 3),
    "M107": ("Metformin", 5.0, 22, 2),
    "M108": ("Zinc Tablets", 3.0, 26, 3),
}

sales_data = []
calendar_data = []

for date in dates:

    is_weekend = 1 if date.weekday() >= 5 else 0
    is_holiday = 1 if date.strftime("%m-%d") in ["01-01", "01-26"] else 0
    promotion_flag = 1 if (date.day % 10 == 0) else 0

    calendar_data.append([date, is_holiday, is_weekend, promotion_flag])

    for store in stores:

        store_multiplier = {
            "S1": 0.95,
            "S2": 1.10,
            "S3": 1.05,
            "S4": 0.85,
            "S5": 1.30
        }[store]

        for item_id, (name, price, base_demand, volatility) in items.items():

            demand = base_demand

            weekly_pattern = [1.0, 0.95, 1.0, 1.05, 1.1, 1.25, 1.30]
            demand *= weekly_pattern[date.weekday()]

            if is_holiday:
                demand *= 1.4

            if promotion_flag:
                demand *= 1.3

            month_growth = 1 + (date.month - 1) * 0.04
            demand *= month_growth
            demand *= store_multiplier

            demand += np.random.normal(0, volatility)

            sales_data.append([
                date,
                store,
                item_id,
                max(0, int(demand))
            ])

sales_df = pd.DataFrame(
    sales_data,
    columns=["date", "store_id", "item_id", "units_sold"]
)

calendar_df = pd.DataFrame(
    calendar_data,
    columns=["date", "is_holiday", "is_weekend", "promotion_flag"]
)

items_df = pd.DataFrame(
    [(k, v[0], v[1]) for k, v in items.items()],
    columns=["item_id", "medicine_name", "base_price"]
)

sales_df.to_csv("data/raw/sales.csv", index=False)
calendar_df.to_csv("data/raw/calendar.csv", index=False)
items_df.to_csv("data/raw/items.csv", index=False)

print(" Enterprise Dataset Generated Successfully!")
