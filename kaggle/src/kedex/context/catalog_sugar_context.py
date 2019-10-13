import copy
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Union  # NOQA

from kedro.context import KedroContext, KedroContextError
from kedro.io import DataCatalog

log = logging.getLogger(__name__)


class CatalogSyntacticSugarContext(KedroContext):
    """ Convert Kedrex's Syntactic Sugar to pure Kedro Catalog. """

    def _create_catalog(  # pylint: disable=no-self-use,too-many-arguments
        self,
        conf_catalog: Dict[str, Any],
        conf_creds: Dict[str, Any],
        save_version: str = None,
        journal: Any = None,
        load_versions: Dict[str, str] = None,
    ) -> DataCatalog:
        """DataCatalog instantiation.
        Allow whether to apply CachedDataSet using `cached` key.
        Returns:
            DataCatalog defined in `catalog.yml`.
        """
        conf_catalog = self._format_kedro_catalog(conf_catalog)
        return DataCatalog.from_config(
            conf_catalog,
            conf_creds,
            save_version=save_version,
            load_versions=load_versions,
        )

    def _format_kedro_catalog(self, conf_catalog):
        default_dict = {}
        if "/" in conf_catalog:
            default_dict = conf_catalog.pop("/")

        conf_catalog_processed = {}

        for ds_name, ds_dict_ in conf_catalog.items():
            ds_dict = copy.deepcopy(default_dict)
            if isinstance(ds_dict_, dict):
                ds_dict.update(ds_dict_)
            ds_name, ds_dict = self._format_kedro_dataset(ds_name, ds_dict)
            conf_catalog_processed[ds_name] = ds_dict
        return conf_catalog_processed

    def _format_kedro_dataset(self, ds_name, ds_dict):
        ds_name, ds_dict = self._set_filepath(ds_name, ds_dict)
        ds_name, ds_dict = self._enable_caching(ds_name, ds_dict)
        return ds_name, ds_dict

    def _set_filepath(self, ds_name, ds_dict):
        if "filepath" not in ds_dict:
            ds_dict["filepath"] = ds_name
            ds_name = Path(ds_name).stem
        return ds_name, ds_dict

    def _enable_caching(self, ds_name, ds_dict):
        cached = False
        if "cached" in ds_dict:
            cached = ds_dict.pop("cached")
        if cached and (ds_dict.get("type") != "kedro.contrib.io.cached.CachedDataSet"):
            ds_dict = {
                "type": "kedro.contrib.io.cached.CachedDataSet",
                "dataset": ds_dict,
            }
        return ds_name, ds_dict
