from kedro.context import KedroContext
from kedex.context.flexible_context import FlexibleContext
from datetime import datetime, timedelta
from mlflow import log_artifact, log_metric, log_param
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Union  # NOQA
import logging

log = logging.getLogger(__name__)


class MLflowContext(KedroContext):
    logging_artifacts = []  # type: Iterable[str]
    offset_hours = 0  # type: int

    def __init__(
        self,
        *args,  # type: Any
        logging_artifacts=[],  # type: Iterable[str]
        offset_hours=0,  # type: int
        **kwargs  # type: Any
    ):
        super().__init__(*args, **kwargs)
        self.logging_artifacts = logging_artifacts or self.logging_artifacts
        self.offset_hours = offset_hours or self.offset_hours

    def _format_kedro_dataset(self, ds_name, ds_dict):
        ds_name, ds_dict = self._set_filepath(ds_name, ds_dict)
        ds_name, ds_dict = self._get_mlflow_logging_flag(ds_name, ds_dict)
        ds_name, ds_dict = self._enable_caching(ds_name, ds_dict)
        return ds_name, ds_dict

    def _get_mlflow_logging_flag(self, ds_name, ds_dict):
        if "mlflow_logging" in ds_dict:
            mlflow_logging = ds_dict.pop("mlflow_logging")
            if mlflow_logging and ds_name not in self.logging_artifacts:
                self.logging_artifacts.append(ds_name)
        return ds_name, ds_dict

    def run(
        self,
        *args,  # type: Any
        **kwargs  # type: Any
    ):
        parameters = self.catalog._data_sets["parameters"].load()
        mlflow_logging_params = parameters.get("mlflow_logging_params")
        if mlflow_logging_params:
            self.offset_hours = (
                mlflow_logging_params.get("offset_hours") or self.offset_hours
            )
            self.logging_artifacts = (
                mlflow_logging_params.get("logging_artifacts") or self.logging_artifacts
            )

        conf_path = Path(self.config_loader.conf_paths[0]) / "parameters.yml"
        log_artifact(conf_path)

        log_metric("__t0", get_timestamp_int(offset_hours=self.offset_hours))
        log_param("time_begin", get_timestamp(offset_hours=self.offset_hours))

        nodes = super().run(*args, **kwargs)

        log_metric("__t1", get_timestamp_int(offset_hours=self.offset_hours))
        log_param("time_end", get_timestamp(offset_hours=self.offset_hours))

        for d in self.logging_artifacts:
            ds = getattr(self.catalog.datasets, d, None)
            if ds:
                fp = getattr(ds, "_filepath", None)
                if not fp:
                    low_ds = getattr(ds, "_dataset", None)
                    if low_ds:
                        fp = getattr(low_ds, "_filepath", None)
                if fp:
                    log_artifact(fp)
                    log.info("'{}' was logged by MLflow.".format(fp))
                else:
                    log.warning("_filepath of '{}' could not be found.".format(d))

        return nodes


def get_timestamp(offset_hours=0, fmt="%Y-%m-%dT%H:%M:%S"):
    return (datetime.now() + timedelta(hours=offset_hours)).strftime(fmt)


def get_timestamp_int(offset_hours=0):
    return int(get_timestamp(offset_hours=offset_hours, fmt="%Y%m%d%H%M"))


class MLflowFlexibleContext(MLflowContext, FlexibleContext):
    pass
