import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path
from collections import OrderedDict

from scipy.sparse import coo_matrix

import torch
from torchvision.transforms import ToTensor, Compose

if sys.version_info >= (3, 6, 8):
    from skew_scaler import SkewScaler
    from mlflow import log_metrics, log_params
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

        count_df = (
            df.groupby(
                ["PlayIndex", "PlayerCategory", "X_int", "Y_int"], as_index=False
            )["NflId"]
            .count()
            .astype(np.uint8)
        )
        count_df.set_index(["PlayIndex", "PlayerCategory"], inplace=True)

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

    log.info("fit_df.shape: {} | vali_df.shape: {}".format(fit_df.shape, vali_df.shape))
    train_dataset = FieldImagesDataset(fit_df, transform=ToTensor())
    val_dataset = FieldImagesDataset(vali_df, transform=ToTensor())

    return train_dataset, val_dataset


def generate_field_images(df, parameters):

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


def fit_base_model(df, parameters):

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


def infer(model, parameters):
    from kaggle.competitions import nflrush

    env = nflrush.make_env()
    for (test_df, sample_prediction_df) in env.iter_test():
        test_df = preprocess(test_df)
        sample_prediction_df.iloc[0, :] = _predict_cdf(test_df, model)
        env.predict(sample_prediction_df)
        if sys.version_info >= (3, 6, 8):
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
    if sys.version_info == (3, 6, 8):
        log.info("Completed 1st inference iteration. Skip the rest.")
        project_path = Path(__file__).resolve().parent.parent.parent

        src_path = project_path / "input" / "nfl-big-data-bowl-2020"
        if str(src_path) not in sys.path:
            sys.path.insert(0, str(src_path))

        if "PYTHONPATH" not in os.environ:
            os.environ["PYTHONPATH"] = src_path

    print("Read CSV file.")
    df = pd.read_csv("../input/nfl-big-data-bowl-2020/train.csv", low_memory=False)
    parameters = None
    print("Preprocess.")
    df = preprocess(df)
    print("Fit model.")
    model = fit_base_model(df, parameters)
    print("Infer.")
    infer(model, parameters)
    print("Completed.")
