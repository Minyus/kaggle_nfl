import copy
from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np
from PIL import Image

from kedro.contrib.io import DefaultArgumentsMixIn
from kedro.io.core import AbstractVersionedDataSet, DataSetError, Version
import logging

log = logging.getLogger(__name__)


class ImagesLocalDataSet(DefaultArgumentsMixIn, AbstractVersionedDataSet):
    def __init__(
        self,
        filepath: str,
        load_args: Dict[str, Any] = None,
        save_args: Dict[str, Any] = None,
        version: Version = None,
    ) -> None:

        super().__init__(
            filepath=Path(filepath),
            load_args=load_args,
            save_args=save_args,
            version=version,
        )

    def _load(self) -> Any:

        load_path = Path(self._get_load_path())

        load_args = copy.deepcopy(self._load_args)
        load_args = load_args or dict()

        as_numpy = load_args.pop("as_numpy", None)

        if load_path.is_dir():
            images_dict = {}
            for p in load_path.glob("*"):
                with p.open("r") as local_file:
                    img = Image.open(local_file, **load_args)
                    images_dict[p.stem] = img
            if as_numpy:
                images = list(images_dict.values())
                names = list(images_dict.keys())

                images = [np.asarray(img) for img in images]
                images = np.stack(images, axis=0)
                images_dict = dict(images=images, names=names)

            return images_dict

        else:
            with load_path.open("r") as local_file:
                img = Image.open(local_file, **load_args)
                if as_numpy:
                    img = np.asarray(img)
                return img

    def _save(self, data: Union[np.ndarray, dict, type(Image)]) -> None:
        save_path = Path(self._get_save_path())
        p = Path(save_path)

        save_args = copy.deepcopy(self._save_args)
        save_args = save_args or dict()
        mode = save_args.pop("mode", None)

        save_path.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(data, type(Image)):
            data.save(p, **save_args)
            return None

        elif isinstance(data, list):
            p.mkdir(parents=True, exist_ok=True)
            for i, img in enumerate(data):
                assert isinstance(img, type(Image))
                s = p / "{}_{:05d}{}".format(p.stem, i, p.suffix)
                img.save(s, **save_args)
                return None

        elif isinstance(data, np.ndarray):
            images = data

        elif isinstance(data, dict):
            images = data.get("images")
            assert isinstance(images, np.ndarray)
            names = data.get("names")

        else:
            raise ValueError("Unsupported data type: {}".format(type(data)))

        if images.ndim in {2, 3}:
            img = Image.fromarray(images, mode=mode)
            img.save(p, **save_args)
            return None

        elif images.ndim == 4:
            p.mkdir(parents=True, exist_ok=True)
            total = images.shape[0]

            for i in range(total):
                name = names[i] if names else "{:05d}".format(i)
                s = p / "{}_{}{}".format(p.stem, name, p.suffix)
                img = Image.fromarray(images[i, :, :, :], mode=mode)
                img.save(s, **save_args)
            return None

        else:
            raise ValueError("Unsupported number of dimensions: {}".format(images.ndim))

    def _describe(self) -> Dict[str, Any]:
        return dict(
            filepath=self._filepath,
            load_args=self._save_args,
            save_args=self._save_args,
            version=self._version,
        )

    def _exists(self) -> bool:
        try:
            path = self._get_load_path()
        except DataSetError:
            return False
        return Path(path).exists()
