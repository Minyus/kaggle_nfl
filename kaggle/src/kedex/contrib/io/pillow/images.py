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
        upper = load_args.pop("upper", None)
        lower = load_args.pop("lower", None)
        to_scale = upper or lower

        if load_path.is_dir():
            images_dict = {}
            for p in load_path.glob("*"):
                with p.open("r") as local_file:
                    img = Image.open(local_file, **load_args)
                    images_dict[p.stem] = img
            if as_numpy or to_scale:
                images = list(images_dict.values())
                names = list(images_dict.keys())

                images = [np.asarray(img) for img in images]
                images = np.stack(images, axis=0)

                images = scale(images, lower=lower, upper=upper)

                if as_numpy:
                    images_dict = dict(images=images, names=names)

                if not as_numpy:
                    for i in range(images.shape[0]):
                        img = Image.fromarray(images[i, :, :, :])
                        images_dict[names[i]] = img

            return images_dict

        else:
            with load_path.open("r") as local_file:
                img = Image.open(local_file, **load_args)
                if as_numpy or to_scale:
                    img = np.asarray(img)
                    img = scale(img, lower=lower, upper=upper)

                    if not as_numpy:
                        img = Image.fromarray(img)

                return img

    def _save(self, data: Union[np.ndarray, dict, type(Image)]) -> None:
        save_path = Path(self._get_save_path())
        save_path.parent.mkdir(parents=True, exist_ok=True)
        p = save_path

        save_args = copy.deepcopy(self._save_args)
        save_args = save_args or dict()
        mode = save_args.pop("mode", None)
        upper = save_args.pop("upper", None)
        lower = save_args.pop("lower", None)
        to_scale = upper or lower

        if isinstance(data, dict):
            images = data.get("images")
            names = data.get("names")
        else:
            images = data

        if isinstance(images, type(Image)):
            if not to_scale:
                img = images
                img.save(p, **save_args)
                return None
            else:
                images = np.asarray(images)

        if isinstance(images, np.ndarray):
            if images.ndim in {2, 3}:
                img = images
                img = scale(img, lower=lower, upper=upper)
                img = Image.fromarray(img, mode=mode)
                img.save(p, **save_args)
                return None
            elif images.ndim in {4}:
                images = scale(lower=lower, upper=upper)(images)
                dataset = Np3DArrDataset(images)
            else:
                raise ValueError(
                    "Unsupported number of dimensions: {}".format(images.ndim)
                )
        elif hasattr(images, "__getitem__") and hasattr(images, "__len__"):
            if not to_scale:
                p.mkdir(parents=True, exist_ok=True)
                for i, img in enumerate(images):
                    if isinstance(img, np.ndarray):
                        img = Image.fromarray(img)
                    assert isinstance(img, type(Image))
                    s = p / "{}_{:05d}{}".format(p.stem, i, p.suffix)
                    img.save(s, **save_args)
                return None
            else:
                dataset = Np3DArrDatasetFromList(
                    images, transform=scale(lower=lower, upper=upper)
                )
        else:
            raise ValueError("Unsupported data type: {}".format(type(images)))

        p.mkdir(parents=True, exist_ok=True)
        for i in range(len(dataset)):
            img = dataset[i]
            if isinstance(img, (tuple, list)):
                img = img[0]
            img = Image.fromarray(img, mode=mode)
            name = names[i] if names else "{:05d}".format(i)
            s = p / "{}_{}{}".format(p.stem, name, p.suffix)
            img.save(s, **save_args)
        return None

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


def scale(**kwargs):
    def _scale(a):
        lower = kwargs.get("lower")
        upper = kwargs.get("upper")
        if lower or upper:
            max_val = a.max()
            min_val = a.min()
            stat_dict = dict(max_val=max_val, min_val=min_val)
            log.info(stat_dict)
            upper = upper or max_val
            lower = lower or min_val
            a = (
                ((a - min_val) / (max_val - min_val)) * (upper - lower) + lower
            ).astype(np.uint8)
        return a

    return _scale


class Np3DArrDataset:
    def __init__(self, a):
        self.a = a

    def __getitem__(self, index):
        return self.a[index, ...]

    def __len__(self):
        return len(self.a)


class Np3DArrDatasetFromList:
    def __init__(self, a, transform=None):
        self.a = a
        self.transform = transform

    def __getitem__(self, index):
        item = np.asarray(self.a[index])
        if self.transform:
            item = self.transform(item)
        return item

    def __len__(self):
        return len(self.a)
