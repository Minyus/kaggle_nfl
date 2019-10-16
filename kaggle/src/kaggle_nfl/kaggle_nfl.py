import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path
from collections import OrderedDict

from scipy.sparse import coo_matrix

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
    df["X_int"] = np.floor(df["X_std"] + 10).clip(lower=0, upper=119).astype(np.uint8)
    df["Y_int"] = np.floor(df["Y_std"]).clip(lower=0, upper=59).astype(np.uint8)

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


def play_generator(df, use_tqdm=True, **kwargs):

    # play_df = df.query("PlayId == @play_id")
    play_id_sr = df["PlayId"].drop_duplicates()
    play_id_list = play_id_sr.to_list()

    total = len(play_id_list)

    df.set_index("PlayId", inplace=True)

    iterator = range(total)

    def _play_generator():

        for i in iterator:
            play_id = play_id_list[i]
            play_df = df.xs(key=play_id, drop_level=False).reset_index()
            last = i == total - 1
            yield i, play_df, last

    if use_tqdm:
        from tqdm import tqdm

        return tqdm(_play_generator(), total=total, **kwargs)
    else:
        return _play_generator()


def generate_field_images(df, parameters):

    img_3darr_list = []
    play_df_list = []
    for i, play_df, last in play_generator(df):
        play_df_id = play_df["PlayId"].iloc[0]
        play_df_list.append(play_df_id)
        img = np.zeros((120, 60, 3), dtype=np.uint8)
        img_ch_2darr_list = []
        for i in range(3):
            cat_df = play_df.query("PlayerCategory == @i")
            count_df = (
                cat_df.groupby(["X_int", "Y_int"], as_index=False)["NflId"]
                .count()
                .astype(np.uint8)
            )
            ch_2darr = coo_matrix(
                (count_df["NflId"], (count_df["X_int"], count_df["Y_int"])),
                shape=(120, 60),
            ).toarray()
            img_ch_2darr_list.append(ch_2darr)
        img_3darr = np.stack(img_ch_2darr_list, axis=2)
        img_3darr_list.append(img_3darr)
    img_4darr = np.stack(img_3darr_list, axis=0)
    images = dict(images=img_4darr, names=play_df_list)
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
        play_dfs = play_generator(vali_df)

        for i, play_df, last in play_dfs:
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
                if hasattr(play_dfs, "set_postfix"):
                    play_dfs.set_postfix(ordered_dict=report_orddict, refresh=True)
                else:
                    print(report_orddict)
            assert not np.isnan(play_crps)

        log.info(metrics_orddict)
        log.info(crps_max_play_orddict)
        log_metrics(dict(metrics_orddict))
        log_params(dict(crps_max_play_orddict))

    return model


def _predict_cdf(test_df, model):

    yards_abs = test_df["YardsFromOwnGoal"].iloc[0]

    pred_arr = np.zeros(199)
    pred_arr[-100:] = np.ones(100)

    target_yards_sr = pd.Series(np.arange(-yards_abs, 100 - yards_abs, 1))
    outcome_sr = _relative_values(target_yards_sr, test_df["RelativeDefenceMeanYards"])
    assert not outcome_sr.isna().any()
    cdf_arr = model.cdf(outcome_sr)
    assert not np.isnan(cdf_arr).any()
    cdf_arr = np.maximum.accumulate(cdf_arr)
    # for i in range(len(cdf_arr) - 1):
    #     cdf_arr[i + 1] = max(cdf_arr[i + 1], cdf_arr[i])

    pred_arr[(99 - yards_abs) : (199 - yards_abs)] = cdf_arr
    return pred_arr


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
