import argparse
import io
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from minio import Minio
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

import mlflow.sklearn

from cs_ubb_eduml_app_ma.config import Settings
from cs_ubb_eduml_app_ma.mlflow.wrappers import sklearn_model

settings = Settings.from_env()
warnings.filterwarnings("ignore")
np.random.seed(40)


ROOT_DIR = Path(__file__).parent.parent.parent


def load_data() -> pd.DataFrame:
    if settings.minio.enabled:
        minio = Minio(
            settings.minio.uri,
            access_key=settings.minio.user,
            secret_key=settings.minio.password,
            secure=False
        )
        response = minio.get_object(settings.minio.bucket, settings.minio.path)
        try:
            return pd.read_csv(io.StringIO(response.data.decode()))
        finally:
            response.close()
            response.release_conn()
    else:
        local_path = ROOT_DIR / "data" / "dataset.csv"
        with open(local_path, "r"):
            result = pd.read_csv(local_path)
        return result


def eval_metrics(actual, pred) -> dict:
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return {
        "rmse": rmse,
        "r2": r2,
        "mae": mae,
    }


@sklearn_model(settings.mlflow.enabled, settings.mlflow.tracking_uri, settings.mlflow.experiment_name)
def fit_predict_wine_quality(a: float, l1: float):
    wine_quality_df = load_data()
    train, test = train_test_split(wine_quality_df)

    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]

    lr = ElasticNet(alpha=a, l1_ratio=l1, random_state=42)
    lr.fit(train_x, train_y)

    predicted_qualities = lr.predict(test_x)
    return (
        test_x,
        predicted_qualities,
        lr,
        eval_metrics(test_y, predicted_qualities),
    )


# Split the data into training and test sets. (0.75, 0.25) split.
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha")
    parser.add_argument("--l1-ratio")
    args = parser.parse_args()
    alpha = float(args.alpha)
    l1_ratio = float(args.l1_ratio)
    print("parsed args alpha", alpha, "and l1 ratio", l1_ratio)

    fit_predict_wine_quality(alpha, l1_ratio)
