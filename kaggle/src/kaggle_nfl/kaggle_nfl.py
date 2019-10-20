import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path
from collections import OrderedDict

from scipy.sparse import coo_matrix

import torch
from torchvision.transforms import ToTensor, Compose
import math

if __file__ == "kaggle_nfl":
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
        "GameId",
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
        "Distance",
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
        "HomeTeamAbbr",
        "VisitorTeamAbbr",
        "Week",
        # 'Stadium',
        # 'Location',
        # 'StadiumType',
        "Turf",
        "GameWeather",
        "Temperature",
        "Humidity",
        "WindSpeed",
        "WindDirection",
        # 'ToLeft',
        # 'IsBallCarrier',
        # 'TeamOnOffense',
        "IsOnOffense",
        "YardsFromOwnGoal",
        "X_std",
        "Y_std",
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


def play_df_to_image_and_yards(play_df):
    img_ch_2darr_list = []
    len_x = 30
    len_y = 60
    for i in range(3):
        cat_df = play_df.query("PlayerCategory == @i")
        count_df = (
            cat_df.groupby(["X_int", "Y_int"], as_index=False)["NflId"]
            .count()
            .astype(np.uint8)
        )
        ch_2darr = coo_matrix(
            (count_df["NflId"], (count_df["X_int"], count_df["Y_int"])),
            shape=(len_x, len_y),
        ).toarray()
        img_ch_2darr_list.append(ch_2darr)
    img_3darr = np.stack(img_ch_2darr_list, axis=2)

    if "Yards" in play_df.columns:
        yards = play_df["Yards"].iloc[0]
        return img_3darr, yards
    else:
        return img_3darr


class FieldImagesDataset:
    def __init__(self, df, channel_first=False, transform=None, target_transform=None):
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
        count_df.set_index(["PlayIndex", "PlayerCategory"], inplace=True)
        count_df.loc[:, "NflId"] = count_df["NflId"].astype(np.uint8)

        self.coo_dict = {
            pi: {
                ci: (
                    count_df["NflId"].values,
                    (count_df["X_int"].values, count_df["Y_int"].values),
                )
                for ci in range(3)
            }
            for pi in play_id_dict.keys()
        }

        len_x = 30
        len_y = 60
        self.shape = (len_x, len_y)
        self.channel_axis = 0 if channel_first else 2

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):

        play_coo_dict = self.coo_dict[index]
        img_ch_2darr_list = [
            coo_matrix(play_coo_dict[ci], shape=self.shape).toarray() for ci in range(3)
        ]
        img = np.stack(img_ch_2darr_list, axis=self.channel_axis)

        target = self.target_dict[index]

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return self.len


def generate_datasets(df, parameters):

    if "Validation" in df.columns:
        fit_df = df.query("Validation == 0").drop(columns=["Validation"])
        vali_df = df.query("Validation == 1").drop(columns=["Validation"])
    else:
        fit_df = df
        vali_df = df

    log.info("Setting up train_dataset from df shape: {}".format(fit_df.shape))
    train_dataset = FieldImagesDataset(fit_df, transform=ToTensor())
    log.info("Setting up val_dataset from df shape: {}".format(vali_df.shape))
    val_dataset = FieldImagesDataset(vali_df, transform=ToTensor())

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
    from skew_scaler import SkewScaler
    from mlflow import log_metrics, log_params

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

    img_3darr = play_df_to_image_and_yards(test_df)

    pytorch_model.eval()
    with torch.no_grad():
        imgs_3dtt = ToTensor()(img_3darr)
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


class PytorchLogNormalCDF(torch.nn.Module):
    def __init__(self, x_start=1, x_end=200, x_step=1, x_scale=0.01):
        value = torch.log(
            torch.arange(start=x_start, end=x_end, step=x_step, dtype=torch.float32)
            * x_scale
        )
        self.value = torch.unsqueeze(value, 0)
        super().__init__()

    def forward(self, x):
        loc = torch.unsqueeze(x[:, 0], 1)
        scale = torch.unsqueeze(torch.exp(x[:, 1]), 1)
        return 0.5 * (
            1 + torch.erf((self.value - loc) * scale.reciprocal() / math.sqrt(2))
        )
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


""" pytorch_ops.py """
import torch

from torch.utils.data import DataLoader
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import RunningAverage
from ignite.contrib.handlers.tqdm_logger import ProgressBar
import numpy as np
from pkg_resources import parse_version
import logging

log = logging.getLogger(__name__)


def pytorch_train(
    train_params,  # type: dict
    mlflow_logging=True,  # type: bool
):
    if mlflow_logging:
        try:
            import mlflow
        except ImportError:
            log.warning("Failed to import mlflow. MLflow logging is disabled.")
            mlflow_logging = False

    if mlflow_logging:
        import ignite

        if parse_version(ignite.__version__) >= parse_version("0.2.1"):
            from ignite.contrib.handlers.mlflow_logger import (
                MLflowLogger,
                OutputHandler,
                global_step_from_engine,
            )
        else:
            from .ignite.contrib.handlers.mlflow_logger import (
                MLflowLogger,
                OutputHandler,
                global_step_from_engine,
            )

    def _pytorch_train(model, train_dataset, val_dataset=None, parameters=None):

        train_data_loader_params = train_params.get("train_data_loader_params", dict())
        val_data_loader_params = train_params.get("val_data_loader_params", dict())
        epochs = train_params.get("epochs")
        progress_update = train_params.get("progress_update", dict())

        optim = train_params.get("optim")
        optim_params = train_params.get("optim_params", dict())
        loss_fn = train_params.get("loss_fn")
        metrics = train_params.get("metrics")

        evaluate_train_data = train_params.get("evaluate_train_data")
        evaluate_val_data = train_params.get("evaluate_val_data")

        seed = train_params.get("seed")
        cudnn_deterministic = train_params.get("cudnn_deterministic")
        cudnn_benchmark = train_params.get("cudnn_benchmark")

        if seed:
            torch.manual_seed(seed)
            np.random.seed(seed)
        if cudnn_deterministic:
            torch.backends.cudnn.deterministic = cudnn_deterministic
        if cudnn_benchmark:
            torch.backends.cudnn.benchmark = cudnn_benchmark

        optimizer = optim(model.parameters(), **optim_params)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        trainer = create_supervised_trainer(
            model, optimizer, loss_fn=loss_fn, device=device
        )

        train_data_loader_params.setdefault("shuffle", True)
        train_data_loader_params.setdefault("drop_last", True)
        train_loader = DataLoader(train_dataset, **train_data_loader_params)

        if evaluate_train_data:
            evaluator_train = create_supervised_evaluator(
                model, metrics=metrics, device=device
            )

        if evaluate_val_data:
            val_loader = DataLoader(val_dataset, **val_data_loader_params)
            evaluator_val = create_supervised_evaluator(
                model, metrics=metrics, device=device
            )

        pbar = None
        if isinstance(progress_update, dict):
            RunningAverage(output_transform=lambda x: x).attach(trainer, "loss")

            progress_update.setdefault("persist", True)
            progress_update.setdefault("desc", "")
            pbar = ProgressBar(**progress_update)
            pbar.attach(trainer, ["loss"])

        @trainer.on(Events.EPOCH_COMPLETED)
        def log_evaluation_results(engine):
            if evaluate_train_data:
                evaluator_train.run(train_loader)
                if pbar:
                    pbar.log_message(
                        _get_report_str(engine, evaluator_train, "Train Data")
                    )
            if evaluate_val_data:
                evaluator_val.run(val_loader)
                if pbar:
                    pbar.log_message(_get_report_str(engine, evaluator_val, "Val Data"))

        if mlflow_logging:
            mlflow_logger = MLflowLogger()

            logging_params = {
                "train_n_samples": len(train_dataset),
                "val_n_samples": len(val_dataset),
                "optim": optim.__name__,
                "loss_fn": loss_fn.__name__,
                "pytorch_version": torch.__version__,
                "ignite_version": ignite.__version__,
            }
            logging_params.update(_loggable_dict(train_data_loader_params, "train"))
            logging_params.update(_loggable_dict(val_data_loader_params, "val"))
            logging_params.update(_loggable_dict(optim_params))
            mlflow_logger.log_params(logging_params)

            if evaluate_train_data:
                mlflow_logger.attach(
                    evaluator_train,
                    log_handler=OutputHandler(
                        tag="train",
                        metric_names=list(metrics.keys()),
                        global_step_transform=global_step_from_engine(trainer),
                    ),
                    event_name=Events.EPOCH_COMPLETED,
                )
            if evaluate_val_data:
                mlflow_logger.attach(
                    evaluator_val,
                    log_handler=OutputHandler(
                        tag="val",
                        metric_names=list(metrics.keys()),
                        global_step_transform=global_step_from_engine(trainer),
                    ),
                    event_name=Events.EPOCH_COMPLETED,
                )

        trainer.run(train_loader, max_epochs=epochs)

        return model

    return _pytorch_train


def _get_report_str(engine, evaluator, tag=""):
    report_str = "[Epoch: {} | {} | Metrics: {}]".format(
        engine.state.epoch, tag, evaluator.state.metrics
    )
    return report_str


def _loggable_dict(d, prefix=None):
    return {
        ("{}_{}".format(prefix, k) if prefix else k): (
            "{}".format(v) if isinstance(v, (tuple, list, dict, set)) else v
        )
        for k, v in d.items()
    }


class PytorchSequential(torch.nn.Sequential):
    def __init__(self, modules):
        super().__init__(*modules)


class PytorchFlatten(torch.nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


""" """

if __name__ == "__main__":
    if __file__ == "kaggle_nfl":
        log.info("Completed 1st inference iteration. Skip the rest.")
        project_path = Path(__file__).resolve().parent.parent.parent

        src_path = project_path / "input" / "nfl-big-data-bowl-2020"
        if str(src_path) not in sys.path:
            sys.path.insert(0, str(src_path))

        if "PYTHONPATH" not in os.environ:
            os.environ["PYTHONPATH"] = src_path

    print("Read CSV file.")
    df = pd.read_csv("../input/nfl-big-data-bowl-2020/train.csv", low_memory=False)

    print("Preprocess.")
    df = preprocess(df)
    print("Fit model.")

    import ignite

    train_batch_size = 128
    loss_fn = nfl_crps_loss
    train_params = dict(
        optim=torch.optim.SGD,
        optim_params=dict(
            lr=train_batch_size / 1000,
            momentum=1 - train_batch_size / 2000,
            weight_decay=0.1 / train_batch_size,
        ),
        loss_fn=loss_fn,
        metrics=dict(loss=ignite.metrics.Loss(loss_fn=loss_fn)),
        train_data_loader_params=dict(batch_size=train_batch_size, num_workers=4),
        evaluate_train_data=False,
        evaluate_val_data=False,
        epochs=1,  # number of epochs to train
        seed=0,  #
    )

    pytorch_model = torch.nn.Sequential(
        torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3),
        torch.nn.ReLU(),
        torch.nn.AdaptiveMaxPool2d(output_size=(14, 29)),
        torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3),
        torch.nn.ReLU(),
        torch.nn.AdaptiveMaxPool2d(output_size=(6, 14)),
        torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
        torch.nn.ReLU(),
        torch.nn.AdaptiveMaxPool2d(output_size=1),
        PytorchFlatten(),
        torch.nn.Dropout(0.2),
        torch.nn.Linear(in_features=64, out_features=2),
        PytorchLogNormalCDF(x_start=1, x_end=200, x_scale=0.01),
    )

    train_dataset = FieldImagesDataset(df, transform=ToTensor())

    model = pytorch_train(train_params=train_params, mlflow_logging=False)(
        pytorch_model, train_dataset
    )

    # model = fit_base_model(df)
    print("Infer.")
    infer(model)
    print("Completed.")
