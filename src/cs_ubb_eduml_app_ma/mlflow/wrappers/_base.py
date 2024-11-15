import functools
from abc import ABCMeta, abstractmethod
from typing import Callable, Any, Optional

import mlflow
from mlflow.models import infer_signature, ModelSignature


class mlflow_decorator(metaclass=ABCMeta):
    def __new__(cls, *args, **kwargs):
        cls.__IMPORT_SUCCESS = True
        try:
            cls._unsafe_optional_import()
        except ImportError:
            cls.__IMPORT_SUCCESS = False
        return super().__new__(cls)

    def __init__(self, enabled: bool, tracking_uri: str, experiment: Optional[str] = None) -> None:
        if enabled:
            mlflow.set_tracking_uri(tracking_uri)
        self._experiment = experiment

    def __ensure_experiment_id(self):
        if not self._experiment:
            return None
        exp = mlflow.get_experiment_by_name(self._experiment)
        if exp:
            return exp.experiment_id
        return mlflow.create_experiment(self._experiment)

    @classmethod
    @abstractmethod
    def _unsafe_optional_import(cls) -> None:
        pass

    @abstractmethod
    def _log_model(self, model: Any, sig: ModelSignature) -> None:
        pass

    @property
    @abstractmethod
    def _model_path(self) -> str:
        return ""

    def _extra_log_model_args(self, sig: ModelSignature) -> dict:
        result = {"signature": sig}
        if self._experiment is not None:
            result["registered_model_name"] = f"{self._experiment}-{self._model_path}"
        return result

    @staticmethod
    def _process_wrapped_func_result(result: Any) -> tuple:
        if result is None:
            raise TypeError("model training function returned NoneType")
        if not isinstance(result, tuple):
            raise TypeError("model training function did not return tuple")
        if len(result) != 4:
            raise ValueError("expected training to return tuple with 4 values")
        return result

    def __call__(self, wrapped: Callable[..., ModelSignature]) -> Callable:
        @functools.wraps(wrapped)
        def wrapper(*args, **kwargs) -> Any:
            run_finish_status = "FINISHED"
            try:
                run_args = {}
                experiment_id = self.__ensure_experiment_id()
                if experiment_id:
                    run_args["experiment_id"] = experiment_id

                run = mlflow.start_run(**run_args)
                model_params = {
                    f"param_{idx+1}": arg
                    for idx, arg in enumerate(args)
                }
                model_params.update(**kwargs)
                mlflow.log_params(model_params)

                result = wrapped(*args, **kwargs)
                data_in, model_out, model, metrics = self._process_wrapped_func_result(result)

                if self.__IMPORT_SUCCESS:
                    self._log_model(model, infer_signature(data_in, model_out, model_params))
                mlflow.log_metrics(metrics, run_id=run.info.run_id)
                return result
            except Exception as exc:
                run_finish_status = "FAILED"
                raise exc
            finally:
                mlflow.end_run(run_finish_status)

        return wrapper
