from typing import Dict, Iterable, Type
from kedro.context import KedroContext, KedroContextError
from kedro.pipeline import Pipeline  # NOQA

from kedex.hatch_dict.hatch_dict import HatchDict


class PipelinesInParametersContext(KedroContext):
    def _get_pipelines(self) -> Dict[str, Pipeline]:
        parameters = self.catalog._data_sets["parameters"].load()
        pipelines = HatchDict(parameters).get("PIPELINES")
        assert pipelines
        return pipelines

    def run(self, *args, **kwargs):
        parameters = self.catalog._data_sets["parameters"].load()
        run_dict = HatchDict(parameters).get("RUN_CONFIG", dict())
        run_dict.update(kwargs)
        return super().run(*args, **run_dict)
