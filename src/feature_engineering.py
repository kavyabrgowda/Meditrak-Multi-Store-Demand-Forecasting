import numpy as np
from sklearn.preprocessing import LabelEncoder

def create_features(df):

    # Time Features
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)

    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

    df['trend_index'] = (df['date'] - df['date'].min()).dt.days

    # Rolling Features
    df['rolling_mean'] = df.groupby(
        ['store_id', 'item_id']
    )['units_sold'].transform(lambda x: x.rolling(7, min_periods=1).mean())

    df['rolling_std'] = df.groupby(
        ['store_id', 'item_id']
    )['units_sold'].transform(lambda x: x.rolling(7, min_periods=1).std())

    df['volatility_index'] = df['rolling_std'] / (df['rolling_mean'] + 1)

    df['lag_7'] = df.groupby(
        ['store_id', 'item_id']
    )['units_sold'].shift(7)

    df['store_id_original'] = df['store_id']
    df['item_id_original'] = df['item_id']

    le_store = LabelEncoder()
    le_item = LabelEncoder()

    df['store_encoded'] = le_store.fit_transform(df['store_id'])
    df['item_encoded'] = le_item.fit_transform(df['item_id'])

    return df
