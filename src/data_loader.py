import pandas as pd

def load_data(sales_path, calendar_path, items_path):
    sales = pd.read_csv(sales_path, parse_dates=['date'])
    calendar = pd.read_csv(calendar_path, parse_dates=['date'])
    items = pd.read_csv(items_path)

    df = sales.merge(calendar, on='date', how='left')
    df = df.merge(items, on='item_id', how='left')

    return df
