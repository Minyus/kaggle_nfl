""" kaggle_nfl.py """

import pandas as pd
import numpy as np
import random
import sys
import os
from pathlib import Path
from collections import OrderedDict

import torch
from torchvision.transforms import ToTensor, Compose
import math


import logging

log = logging.getLogger(__name__)


def preprocess(df, parameters=None):
    """ Reference:
    https://www.kaggle.com/statsbymichaellopez/nfl-tracking-initial-wrangling-voronoi-areas
    """
    df["ToLeft"] = df["PlayDirection"] == "left"
    df["IsBallCarrier"] = df["NflId"] == df["NflIdRusher"]

    team_abbr_dict = {"ARI": "ARZ", "BAL": "BLT", "CLE": "CLV", "HOU": "HST"}
    df["VisitorTeamAbbr"] = df["VisitorTeamAbbr"].replace(team_abbr_dict)
    df["HomeTeamAbbr"] = df["HomeTeamAbbr"].replace(team_abbr_dict)

    home_dict = {True: "home", False: "away"}
    df["TeamOnOffense"] = (df["PossessionTeam"] == df["HomeTeamAbbr"]).replace(
        home_dict
    )

    df["IsOnOffense"] = df["Team"] == df["TeamOnOffense"]
    df["YardsFromOwnGoal"] = -df["YardLine"] + 100
    df.loc[
        (df["FieldPosition"].astype(str) == df["PossessionTeam"]), "YardsFromOwnGoal"
    ] = df["YardLine"]

    df["X_std"] = df["X"]
    df.loc[df["ToLeft"], "X_std"] = -df["X"] + 120
    df["X_std"] = df["X_std"] - 10

    df["Y_std"] = df["Y"]
    df.loc[df["ToLeft"], "Y_std"] = -df["Y"] + 160 / 3

    """ """
    df["RelativeDefenceMeanYards"] = df["X_std"]
    df.loc[df["IsOnOffense"], "RelativeDefenceMeanYards"] = np.nan
    df["RelativeDefenceMeanYards"] = (
        df.groupby(["PlayId"])["RelativeDefenceMeanYards"].transform("mean")
        - df["YardsFromOwnGoal"]
    )
    """ """
    df["PlayerCategory"] = df["IsOnOffense"].astype(np.uint8)
    df.loc[df["IsBallCarrier"], "PlayerCategory"] = 2
    # df["X_int"] = np.floor(df["X_std"] + 10).clip(lower=0, upper=119).astype(np.uint8)
    len_x = 30
    df["X_int"] = (
        np.floor(df["X_std"] - df["YardsFromOwnGoal"] + 10)
        .clip(lower=0, upper=(len_x - 1))
        .astype(np.uint8)
    )
    len_y = 60
    df["Y_int"] = (
        np.floor(df["Y_std"]).clip(lower=0, upper=(len_y - 1)).astype(np.uint8)
    )

    cols = [
        # "GameId",
        "PlayId",
        # 'Team',
        # 'X',
        # 'Y',
        # 'S',
        # 'A',
        # 'Dis',
        # 'Orientation',
        # 'Dir',
        "NflId",
        # 'DisplayName',
        # 'JerseyNumber',
        "Season",
        "YardLine",
        "Quarter",
        # 'GameClock',
        # 'PossessionTeam',
        "Down",
        # "Distance",
        "FieldPosition",
        "HomeScoreBeforePlay",
        "VisitorScoreBeforePlay",
        # 'NflIdRusher',
        "OffenseFormation",
        "OffensePersonnel",
        "DefendersInTheBox",
        "DefensePersonnel",
        # 'PlayDirection',
        # 'TimeHandoff',
        # 'TimeSnap',
        "Yards",
        "PlayerHeight",
        "PlayerWeight",
        # 'PlayerBirthDate',
        # 'PlayerCollegeName',
        "Position",
        # "HomeTeamAbbr",
        # "VisitorTeamAbbr",
        "Week",
        # 'Stadium',
        # 'Location',
        # 'StadiumType',
        "Turf",
        "GameWeather",
        "Temperature",
        "Humidity",
        # "WindSpeed",
        # "WindDirection",
        # 'ToLeft',
        # 'IsBallCarrier',
        # 'TeamOnOffense',
        "IsOnOffense",
        "YardsFromOwnGoal",
        # "X_std",
        # "Y_std",
        "RelativeDefenceMeanYards",
        "PlayerCategory",
        "X_int",
        "Y_int",
    ]
    df = df.filter(items=cols)

    return df


def _relative_values(abs_sr, comp_sr, offset=101, transform_func=None):
    transform_func = np.log10
    if len(comp_sr) != len(abs_sr):
        comp_sr = comp_sr.iloc[0]
    denominator_sr = comp_sr + offset
    assert (
        (denominator_sr > 0.0)
        if isinstance(denominator_sr, float)
        else (denominator_sr > 0.0).all()
    )

    numerator_sr = abs_sr + offset
    values_sr = numerator_sr / denominator_sr
    assert not values_sr.isna().any()
    if transform_func and callable(transform_func):
        values_sr = transform_func(values_sr)
    return values_sr


class PlayDfsDataset:
    def __init__(self, df, transform=None):
        self.play_id_list = df["PlayId"].drop_duplicates().to_list()
        self.df = df.set_index("PlayId", inplace=False)
        self.transform = transform

    def __getitem__(self, index):
        play_id = self.play_id_list[index]
        item = self.df.xs(key=play_id, drop_level=False).reset_index()
        if self.transform:
            item = self.transform(item)
        return item

    def __len__(self):
        return len(self.play_id_list)


class FieldImagesDataset:
    def __init__(
        self,
        df,
        dim_cols=["PlayerCategory", "X_int", "Y_int"],
        dim_sizes=[3, 30, 60],
        float_scale=None,
        to_pytorch_tensor=False,
        store_as_sparse_tensor=False,
        random_horizontal_flip=dict(p=0),
        random_horizontal_shift=dict(p=0, max_shift=1),
        transform=None,
        target_transform=None,
    ):

        if "Yards" not in df.columns:
            df["Yards"] = np.nan

        play_target_df = (
            df[["PlayId", "Yards"]].drop_duplicates().reset_index(drop=True)
        )
        self.len = len(play_target_df)
        self.target_dict = play_target_df["Yards"].to_dict()

        play_id_dict = play_target_df["PlayId"].to_dict()
        self.play_id_dict = play_id_dict
        play_index_dict = {v: k for k, v in play_id_dict.items()}
        df["PlayIndex"] = df["PlayId"].map(play_index_dict)

        count_df = df.groupby(
            ["PlayIndex", "PlayerCategory", "X_int", "Y_int"], as_index=False
        )["NflId"].count()

        if (float_scale is None and to_pytorch_tensor) or float_scale == True:
            float_scale = 1.0 / 255

        if float_scale:
            count_df.loc[:, "NflId"] = count_df["NflId"].astype(
                np.float32
            ) * np.float32(float_scale)
        else:
            count_df.loc[:, "NflId"] = count_df["NflId"].astype(np.uint8)

        if to_pytorch_tensor:

            count_df.set_index("PlayIndex", inplace=True)
            count_df.loc[:, dim_cols] = count_df[dim_cols].astype(np.int64)
            f = torch.sparse_coo_tensor if store_as_sparse_tensor else dict
            coo_dict = dict()
            for pi in play_id_dict.keys():
                play_df = count_df.xs(pi)
                coo_3d = f(
                    values=torch.from_numpy(play_df["NflId"].values),
                    indices=torch.from_numpy(play_df[dim_cols].values.transpose()),
                    size=dim_sizes,
                )
                coo_dict[pi] = coo_3d

        else:
            count_df.set_index(["PlayIndex", dim_cols[0]], inplace=True)
            from scipy.sparse import coo_matrix

            coo_dict = dict()
            for pi in play_id_dict.keys():
                play_coo_2d_dict = dict()
                for ci in range(3):
                    ch_df = count_df.xs([pi, ci])
                    coo_2d = coo_matrix(
                        (
                            ch_df["NflId"].values,
                            (ch_df[dim_cols[1]].values, ch_df[dim_cols[2]].values),
                        ),
                        shape=tuple(dim_sizes[1:]),
                    )
                    play_coo_2d_dict[ci] = coo_2d
                coo_dict[pi] = play_coo_2d_dict

        self.coo_dict = coo_dict

        self.to_pytorch_tensor = to_pytorch_tensor
        self.store_as_sparse_tensor = store_as_sparse_tensor

        self.random_horizontal_flip = random_horizontal_flip or dict()
        self.random_horizontal_shift = random_horizontal_shift or dict()

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):

        play_coo_3d = self.coo_dict[index]

        if self.to_pytorch_tensor:
            if not self.store_as_sparse_tensor:
                _random_horizontal_flip(
                    indices=play_coo_3d["indices"],
                    size=play_coo_3d.get("size"),
                    p=self.random_horizontal_flip.get("p"),
                )
                _random_horizontal_shift(
                    indices=play_coo_3d["indices"],
                    size=play_coo_3d.get("size"),
                    p=self.random_horizontal_shift.get("p"),
                    max_shift=self.random_horizontal_shift.get("max_shift"),
                )
                play_coo_3d = torch.sparse_coo_tensor(**play_coo_3d)
            img = play_coo_3d.to_dense()
        else:
            play_coo_dict = play_coo_3d
            img_ch_2darr_list = [play_coo_dict[ci].toarray() for ci in range(3)]
            img = np.stack(img_ch_2darr_list, axis=2)

        target = self.target_dict[index]

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return self.len


def _random_cond(p=None):
    return p and random.random() < p


def _random_flip(indices, size, p, dim):
    if _random_cond(p):
        indices[dim, :].mul_(-1).add_(size[dim] - 1)


def _random_horizontal_flip(indices, size, p=None):
    _random_flip(indices, size, p, dim=2)


def _random_vertical_flip(indices, size, p=None):
    _random_flip(indices, size, p, dim=1)


def _random_shift(indices, size, p, max_shift, dim):
    if _random_cond(p):
        shift = int(max_shift * random.uniform(-1, 1))
        indices[dim, :].add_(shift).clamp_(min=0, max=size[dim] - 1)


def _random_horizontal_shift(indices, size, p=None, max_shift=1):
    _random_shift(indices, size, p, max_shift, dim=2)


def _random_vertical_shift(indices, size, p=None, max_shift=1):
    _random_shift(indices, size, p, max_shift, dim=1)


class AugFieldImagesDataset(FieldImagesDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            random_horizontal_flip=dict(p=0.5),
            random_horizontal_shift=dict(p=1, max_shift=1),
            **kwargs
        )


def generate_datasets(df, parameters=None):

    if "Validation" in df.columns:
        fit_df = df.query("Validation == 0").drop(columns=["Validation"])
        vali_df = df.query("Validation == 1").drop(columns=["Validation"])
    else:
        fit_df = df
        vali_df = df

    log.info("Setting up train_dataset from df shape: {}".format(fit_df.shape))
    train_dataset = AugFieldImagesDataset(fit_df, to_pytorch_tensor=True)
    # train_dataset = FieldImagesDataset(fit_df, transform=ToTensor())

    log.info("Setting up val_dataset from df shape: {}".format(vali_df.shape))
    val_dataset = AugFieldImagesDataset(vali_df, to_pytorch_tensor=True)
    # val_dataset = FieldImagesDataset(vali_df, transform=ToTensor())

    return train_dataset, val_dataset


def generate_field_images(df, parameters=None):

    field_images = FieldImagesDataset(df)
    play_id_dict = field_images.play_id_dict

    total = len(field_images)
    use_tqdm = True
    if use_tqdm:
        from tqdm import trange

        play_range = trange(total)
    else:
        play_range = range(total)

    img_3darr_list = []
    yards_list = []
    play_id_list = []
    for i in play_range:
        field_image, yards = field_images[i]
        img_3darr_list.append(field_image)
        yards_list.append(yards)
        play_id_list.append(play_id_dict[i])

    names = ["{}_{}".format(p, y) for p, y in zip(play_id_list, yards_list)]

    img_4darr = np.stack(img_3darr_list, axis=0)
    images = dict(images=img_4darr, names=names)
    return images


def fit_base_model(df, parameters=None):

    from mlflow import log_metrics, log_params

    from scaler.skew_scaler import SkewScaler

    model = SkewScaler()

    if "Validation" in df.columns:
        fit_df = df.query("Validation == 0").drop(columns=["Validation"])
        log.info("Fitting model using data shape: {}".format(fit_df.shape))
    else:
        fit_df = df

    fit_df = preprocess(fit_df)
    fit_play_df = fit_df.drop_duplicates(subset="PlayId")
    outcome_sr = _relative_values(
        fit_play_df["Yards"], fit_play_df["RelativeDefenceMeanYards"]
    )
    model.fit(outcome_sr)

    if "Validation" in df.columns:
        log.info("Validating...")
        vali_df = df.query("Validation == 1").drop(columns=["Validation"])

        play_crps_dict = {}

        play_dfs = PlayDfsDataset(vali_df)
        total = len(play_dfs)

        vali_df.set_index("PlayId", inplace=True)

        use_tqdm = True
        if use_tqdm:
            from tqdm import trange

            play_index_iter = trange(total)
        else:
            play_index_iter = range(total)
        for i in play_index_iter:
            play_df = play_dfs[i]
            last = i == total - 1

            play_id = play_df["PlayId"].iloc[0]
            y_true = play_df["Yards"].iloc[0]
            cdf_arr = _predict_cdf(play_df, model)

            h_arr = np.ones(199)
            h_arr[: (99 + y_true)] = 0
            play_crps = ((cdf_arr - h_arr) ** 2).mean()
            play_crps_dict[play_id] = play_crps

            if (not (i % 100)) or last:
                play_crps_arr = np.array(list(play_crps_dict.values()))
                metrics_orddict = OrderedDict(
                    [
                        ("crps_mean", play_crps_arr.mean()),
                        ("crps_std", play_crps_arr.std()),
                        ("crps_max", play_crps_arr.max()),
                    ]
                )

                crps_max_play_id = max(play_crps_dict, key=play_crps_dict.get)
                crps_max_play_df = vali_df.xs(
                    key=crps_max_play_id, drop_level=False
                ).reset_index()
                crps_max_play_orddict = (
                    crps_max_play_df.query("NflIdRusher == NflId")
                    .astype(str)
                    .to_dict(orient="records", into=OrderedDict)[0]
                )

                report_orddict = OrderedDict([])
                report_orddict.update(metrics_orddict)
                report_orddict.update(crps_max_play_orddict)
                if hasattr(play_index_iter, "set_postfix"):
                    play_index_iter.set_postfix(
                        ordered_dict=report_orddict, refresh=True
                    )
                else:
                    print(report_orddict)
            assert not np.isnan(play_crps)

        log.info(metrics_orddict)
        log.info(crps_max_play_orddict)
        log_metrics(dict(metrics_orddict))
        log_params(dict(crps_max_play_orddict))

    return model


# def _predict_cdf(test_df, model):
#
#     yards_abs = test_df["YardsFromOwnGoal"].iloc[0]
#
#     pred_arr = np.zeros(199)
#     pred_arr[-100:] = np.ones(100)
#
#     target_yards_sr = pd.Series(np.arange(-yards_abs, 100 - yards_abs, 1))
#     outcome_sr = _relative_values(target_yards_sr, test_df["RelativeDefenceMeanYards"])
#     assert not outcome_sr.isna().any()
#     cdf_arr = model.cdf(outcome_sr)
#     assert not np.isnan(cdf_arr).any()
#     cdf_arr = np.maximum.accumulate(cdf_arr)
#     # for i in range(len(cdf_arr) - 1):
#     #     cdf_arr[i + 1] = max(cdf_arr[i + 1], cdf_arr[i])
#
#     pred_arr[(99 - yards_abs) : (199 - yards_abs)] = cdf_arr
#     return pred_arr


def _predict_cdf(test_df, pytorch_model):
    yards_abs = test_df["YardsFromOwnGoal"].iloc[0]

    imgs_3dtt, _ = FieldImagesDataset(test_df, to_pytorch_tensor=True)[0]
    # img_3darr, _ = FieldImagesDataset(test_df, to_pytorch_tensor=True)[0]
    # imgs_3dtt = ToTensor()(img_3darr)

    pytorch_model.eval()
    with torch.no_grad():
        imgs_4dtt = torch.unsqueeze(imgs_3dtt, 0)  # instead of DataLoader
        out_2dtt = pytorch_model(imgs_4dtt)
        pred_arr = torch.squeeze(out_2dtt).numpy()

    pred_arr = np.maximum.accumulate(pred_arr)
    pred_arr[: (99 - yards_abs)] = 0.0
    pred_arr[(199 - yards_abs) :] = 1.0
    return pred_arr


def crps_loss(input, target, target_to_index=None, reduction="mean"):
    index_1dtt = target_to_index(target) if target_to_index else target
    h_1dtt = torch.arange(input.shape[1])

    h_2dtt = (h_1dtt.reshape(1, -1) >= index_1dtt.reshape(-1, 1)).type(
        torch.FloatTensor
    )

    ret = (input - h_2dtt) ** 2
    if reduction != "none":
        ret = torch.mean(ret) if reduction == "mean" else torch.sum(ret)
    return ret


def yards_to_index(y_1dtt):
    return y_1dtt + 99


def nfl_crps_loss(input, target):
    return crps_loss(input, target, target_to_index=yards_to_index)


class NflCrpsLossFunc:
    def __init__(self, min=-15, max=25):
        self.min = min
        self.max = max

    def __call__(self, input, target):
        target = torch.clamp(target, min=self.min, max=self.max)
        return nfl_crps_loss(input, target)


class PytorchLogNormalCDF(torch.nn.Module):
    def __init__(self, x_start=1, x_end=200, x_step=1, x_scale=0.01):
        value = torch.log(
            torch.arange(start=x_start, end=x_end, step=x_step, dtype=torch.float32)
            * x_scale
        )
        value = torch.unsqueeze(value, 0)
        value.requires_grad = False
        self.value = value
        super().__init__()

    def forward(self, x):
        loc = torch.unsqueeze(x[:, 0], 1)
        reciprocal_sqrt2_scale = torch.unsqueeze(torch.exp(x[:, 1]), 1)
        # scale = torch.unsqueeze(torch.exp(x[:, 1]), 1)
        # reciprocal_sqrt2_scale = scale.reciprocal() / math.sqrt(2)
        return 0.5 * (1 + torch.erf((self.value - loc) * reciprocal_sqrt2_scale))
        # return torch.distributions.normal.Normal(loc=loc, scale=scale).cdf(self.value)


def infer(model, parameters=None):
    from kaggle.competitions import nflrush

    env = nflrush.make_env()
    for (test_df, sample_prediction_df) in env.iter_test():
        test_df = preprocess(test_df)
        sample_prediction_df.iloc[0, :] = _predict_cdf(test_df, model)
        env.predict(sample_prediction_df)
        if parameters:
            return sample_prediction_df

    env.write_submission_file()

    return None


def compose(transforms):
    def _compose(d):
        for t in transforms:
            d = t(d)
        return d

    return _compose


if __name__ == "__main__":
    import torchvision
    import ignite
    import logging.config
    import yaml

    logging_yaml = """
    version: 1
    disable_existing_loggers: False
    formatters:
        simple:
            format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    
    handlers:
        console:
            class: logging.StreamHandler
            level: INFO
            formatter: simple
            stream: ext://sys.stdout
    
    root:
        level: INFO
        handlers: [console]
    """
    conf_logging = yaml.safe_load(logging_yaml)
    logging.config.dictConfig(conf_logging)

    log.info("Read CSV file.")
    df = pd.read_csv("../input/nfl-big-data-bowl-2020/train.csv", low_memory=False)

    log.info("Preprocess.")
    df = preprocess(df)

    log.info("Set up dataset.")

    train_batch_size = 256
    train_params = dict(
        epochs=10,  # number of epochs to train
        time_limit=10800,
        early_stopping_params=dict(metric="loss", minimize=True, patience=1000),
        scheduler=ignite.contrib.handlers.param_scheduler.LinearCyclicalScheduler,
        scheduler_params=dict(
            param_name="lr",
            start_value=0.00000001 * train_batch_size,
            end_value=0.0000005 * train_batch_size,
            cycle_epochs=2,  # cycle_size: int(cycle_epochs * len(train_loader))
            cycle_mult=1.0,
            start_value_mult=1.0,
            end_value_mult=1.0,
            save_history=False,
        ),
        optimizer=torch.optim.Adam,
        # optimizer=torch.optim.SGD,
        optimizer_params=dict(
            # lr=train_batch_size / 1000,
            # momentum=1 - train_batch_size / 2000,
            weight_decay=0.01
            / train_batch_size
        ),
        loss_fn=NflCrpsLossFunc(min=-15, max=25),
        metrics=dict(loss=ignite.metrics.Loss(loss_fn=nfl_crps_loss)),
        train_data_loader_params=dict(batch_size=train_batch_size, num_workers=4),
        evaluate_train_data="COMPLETED",
        progress_update=False,
        seed=0,  #
    )

    if "PytorchUnsqueeze" not in dir():
        from kedex.contrib.ops.pytorch_ops import PytorchUnsqueeze
    if "PytorchSqueeze" not in dir():
        from kedex.contrib.ops.pytorch_ops import PytorchSqueeze
    if "PytorchFlatten" not in dir():
        from kedex.contrib.ops.pytorch_ops import PytorchFlatten

    # pytorch_model = torch.nn.Sequential(
    #     torchvision.models.resnet._resnet(
    #         arch="resnet9",
    #         block=torchvision.models.resnet.BasicBlock,
    #         layers=[1, 1, 1, 1],
    #         pretrained=False,
    #         progress=None,
    #         num_classes=199,
    #         # num_classes=2,
    #     ),
    #     torch.nn.Dropout(p=0.5),
    #     torch.nn.Linear(in_features=199, out_features=205),  # 199 + 2 + 2 + 2 = 205
    #     PytorchUnsqueeze(dim=1),
    #     torch.nn.AvgPool1d(kernel_size=3, stride=1, padding=0),
    #     torch.nn.AvgPool1d(kernel_size=3, stride=1, padding=0),
    #     torch.nn.AvgPool1d(kernel_size=3, stride=1, padding=0),
    #     PytorchSqueeze(dim=1),
    #     torch.nn.Sigmoid()
    # )

    pytorch_model = torch.nn.Sequential(
        torch.nn.Conv2d(in_channels=3, out_channels=8, kernel_size=(5, 15)),
        torch.nn.ReLU(),
        torch.nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(5, 15)),
        torch.nn.ReLU(),
        torch.nn.Conv2d(in_channels=16, out_channels=24, kernel_size=(5, 15)),
        torch.nn.ReLU(),
        torch.nn.Conv2d(in_channels=24, out_channels=32, kernel_size=(5, 15)),
        torch.nn.ReLU(),
        torch.nn.Conv2d(in_channels=32, out_channels=40, kernel_size=(5, 4)),
        torch.nn.ReLU(),
        PytorchFlatten(),
        torch.nn.Linear(in_features=400, out_features=205),
        PytorchUnsqueeze(dim=1),
        torch.nn.AvgPool1d(kernel_size=3, stride=1, padding=0),
        torch.nn.AvgPool1d(kernel_size=3, stride=1, padding=0),
        torch.nn.AvgPool1d(kernel_size=3, stride=1, padding=0),
        PytorchSqueeze(dim=1),
        torch.nn.Sigmoid(),
    )

    train_dataset = AugFieldImagesDataset(df, to_pytorch_tensor=True)

    log.info("Fit model.")
    if "pytorch_train" not in dir():
        from kedex.contrib.ops.pytorch_ops import pytorch_train
    model = pytorch_train(train_params=train_params, mlflow_logging=False)(
        pytorch_model, train_dataset
    )

    # model = fit_base_model(df)
    log.info("Infer.")
    infer(model)

    log.info("Completed.")
