import os
from dataclasses import dataclass, field

from dotenv import load_dotenv


@dataclass
class MlflowSettings:
    enabled: bool = False
    tracking_uri: str = ""
    experiment_name: str = "edu-ubb-gpu-experiment"


@dataclass
class MinioSettings:
    enabled: bool = False
    uri: str = "localhost:9000"
    bucket: str = "test-bucket"
    path: str = "edu-ubb-gpu/data.csv"
    user: str | None = None
    password: str | None = None


@dataclass
class Settings:
    __TRUTH_VALUES = {"t", "true", "1", "yes", "y"}

    mlflow: MlflowSettings = field(repr=False, hash=False, compare=False)
    minio: MinioSettings = field(repr=False, hash=False, compare=False)

    @classmethod
    def read_bool_from_env(cls, name: str, default: bool = False) -> bool:
        value = os.getenv(name)
        if value is None:
            return default
        return str(value) in cls.__TRUTH_VALUES

    @staticmethod
    def read_str_from_env(name: str, default: str = "") -> str:
        value = os.getenv(name)
        if value is None:
            return default
        return str(value)

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        mlflow_settings = MlflowSettings(
            enabled=cls.read_bool_from_env("MLFLOW_TRACKING_ENABLED"),
            tracking_uri=cls.read_str_from_env("MLFLOW_TRACKING_URI"),
            experiment_name=cls.read_str_from_env("MLFLOW_EXPERIMENT_NAME", "edu-ubb-gpu-experiment")
        )
        minio_settings = MinioSettings(
            enabled=cls.read_bool_from_env("MINIO_ENABLED"),
            uri=cls.read_str_from_env("MINIO_URI"),
            bucket=cls.read_str_from_env("MINIO_BUCKET"),
            path=cls.read_str_from_env("MINIO_OBJECT_PATH", "edu-ubb-gpu/data.csv"),
            user=cls.read_str_from_env("MINIO_USERNAME"),
            password=cls.read_str_from_env("MINIO_PASSWORD"),
        )
        return Settings(mlflow=mlflow_settings, minio=minio_settings)
