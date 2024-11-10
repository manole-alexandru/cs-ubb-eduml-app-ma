from typing import Any

import mlflow.sklearn
from mlflow.models import ModelSignature

from edu_ubb_gpu.mlflow.wrappers._base import mlflow_decorator


class sklearn_model(mlflow_decorator):
    @classmethod
    def _unsafe_optional_import(cls) -> None:
        import mlflow.sklearn

    def _log_model(self, model: Any, sig: ModelSignature) -> None:
        mlflow.sklearn.log_model(
            model, self._model_path, **self._extra_log_model_args(sig)
        )

    @property
    def _model_path(self) -> str:
        return "sklearn"
