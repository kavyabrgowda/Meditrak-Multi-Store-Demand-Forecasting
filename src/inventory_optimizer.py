def generate_reorder_plan(df_test, predictions):

    df_test['predicted_demand'] = predictions

    df_test['safety_stock'] = (
        df_test['predicted_demand'] * 0.3 +
        df_test['volatility_index'] * 10
    )

    df_test['reorder_point'] = (
        df_test['predicted_demand'] +
        df_test['safety_stock']
    )

    return df_test[
        ['predicted_demand', 'safety_stock', 'reorder_point']
    ]
