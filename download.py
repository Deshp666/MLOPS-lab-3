import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

def download_data():
    df = pd.read_csv('https://raw.githubusercontent.com/Deshp666/cars_dataset/refs/heads/main/cars.csv', delimiter = ',')
    df.to_csv("cars.csv", index = False)
    return df

def clear_data(path2df, drop_duplicates=False, use_statistic_method=True):
    df = pd.read_csv(path2df)
    df = df.drop(['torque'], axis=1)
    df['engine'] = df['engine'].str.replace('CC', '').apply(pd.to_numeric, errors='coerce')
    df['mileage'] = df['mileage'].str.replace('kmpl', '').apply(pd.to_numeric, errors='coerce')
    df['max_power'] = df['max_power'].str.replace('bhp', '').apply(pd.to_numeric, errors='coerce')

    df = df.dropna()

    if drop_duplicates:
        df = df.drop_duplicates()

    # Используем статистический метод для обработки выбросов.
    df_col_to_prepare = ['year', 'mileage', 'engine', 'max_power',
                         'km_driven', 'selling_price']
    if use_statistic_method:
        for col in df_col_to_prepare:
            lower_border, upper_border = calculate_outliers(df[col])
            df.loc[df[col] > upper_border, col] = upper_border
            df.loc[df[col] < lower_border, col] = lower_border

    cat_columns = ['name', 'fuel', 'transmission', 'owner', 'seller_type']
    ordinal = OrdinalEncoder()
    ordinal.fit(df[cat_columns]);
    Ordinal_encoded = ordinal.transform(df[cat_columns])
    df_ordinal = pd.DataFrame(Ordinal_encoded, columns=cat_columns)
    df[cat_columns] = df_ordinal[cat_columns]
    df.to_csv('df_clear.csv')
    return True

def calculate_outliers(column):
    inter_quantile = column.quantile(0.75) - column.quantile(0.25)
    lower_border = column.quantile(0.25) - inter_quantile * 1.5
    upper_border = inter_quantile * 1.5 + column.quantile(0.75)
    return round(lower_border), round(upper_border)

download_data()
clear_data("cars.csv")
