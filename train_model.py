from os import name
from sklearn.preprocessing import StandardScaler, PowerTransformer
import pandas as pd
from sklearn.model_selection import train_test_split
import mlflow
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from mlflow.models import infer_signature
import joblib

def scale_frame(frame):
    df = frame.copy()
    X,y = df.drop(columns = ['selling_price']), df['selling_price']
    scaler = StandardScaler()
    power_trans = PowerTransformer()
    X_scale = scaler.fit_transform(X.values)
    Y_scale = power_trans.fit_transform(y.values.reshape(-1,1))
    return X_scale, Y_scale, power_trans

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


if __name__ == "__main__":
    df = pd.read_csv("./df_clear.csv")
    X, Y, power_trans = scale_frame(df)
    X_train, X_val, y_train, y_val = train_test_split(X, Y,
                                                      test_size=0.2,
                                                      random_state=42)

    params = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30],
        'max_features': ['log2', 'sqrt'],
        'min_samples_leaf': [1, 2, 4],
        'min_samples_split': [2, 5, 10]
    }

    mlflow.set_experiment("randomforest model cars")
    with mlflow.start_run():
        rf = RandomForestRegressor(random_state=42)
        clf = GridSearchCV(rf, params, cv=3, n_jobs=-1)
        clf.fit(X_train, y_train.reshape(-1))

        best = clf.best_estimator_
        y_pred = best.predict(X_val)
        y_price_pred = power_trans.inverse_transform(y_pred.reshape(-1, 1))

        (rmse, mae, r2) = eval_metrics(power_trans.inverse_transform(y_val), y_price_pred)

        mlflow.log_param("n_estimators", best.n_estimators)
        mlflow.log_param("max_depth", best.max_depth)
        mlflow.log_param("max_features", best.max_features)
        mlflow.log_param("min_samples_leaf", best.min_samples_leaf)
        mlflow.log_param("min_samples_split", best.min_samples_split)

        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        predictions = best.predict(X_train)
        signature = infer_signature(X_train, predictions)
        mlflow.sklearn.log_model(best, "model", signature=signature)
        with open("rf_cars.pkl", "wb") as file:
            joblib.dump(rf, file)

dfruns = mlflow.search_runs()
path2model = dfruns.sort_values("metrics.r2", ascending=False).iloc[0]['artifact_uri'].replace("file://",
                                                                                               "") + '/model'  # путь до эксперимента с лучшей моделью
print(path2model)