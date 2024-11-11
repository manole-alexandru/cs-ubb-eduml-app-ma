from typing import Any

import mlflow.sklearn
from mlflow.models import ModelSignature

from cs_ubb_eduml_app_ma.mlflow.wrappers._base import mlflow_decorator


class torch_model(mlflow_decorator):
    @classmethod
    def _unsafe_optional_import(cls) -> None:
        import mlflow.pytorch

    def _log_model(self, model: Any, sig: ModelSignature) -> None:
        mlflow.pytorch.log_model(
            model, self._model_path, **self._extra_log_model_args(sig)
        )

    @property
    def _model_path(self) -> str:
        return "pytorch"
