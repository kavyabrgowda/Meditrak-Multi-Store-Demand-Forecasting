def clean_data(df):
    df['units_sold'] = df['units_sold'].fillna(0)
    df = df[df['units_sold'] >= 0]

    # Remove extreme outliers
    upper_limit = df['units_sold'].quantile(0.99)
    df = df[df['units_sold'] <= upper_limit]

    return df
