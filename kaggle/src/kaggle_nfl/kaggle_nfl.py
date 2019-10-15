import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path
from collections import OrderedDict

if sys.version_info >= (3, 6, 8):
    from skew_scaler import SkewScaler
    from mlflow import log_metrics, log_params
    from tqdm import tqdm
    import logging

    log = logging.getLogger(__name__)


def preprocess(df, parameters=None):
    """ Reference:
    https://www.kaggle.com/statsbymichaellopez/nfl-tracking-initial-wrangling-voronoi-areas
    """
    df["ToLeft"] = df["PlayDirection"] == "left"
    df["IsBallCarrier "] = df["NflId"] == df["NflIdRusher"]

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

    return df


def _relative_values(abs_sr, comp_sr, offset=101, transform_func=None):
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


def fit_base_model(df, parameters):

    # play_df["Yards"].clip(lower=-10, upper=50, inplace=True)

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
        play_id_sr = vali_df["PlayId"].drop_duplicates()
        play_id_list = play_id_sr.to_list()

        vali_df.set_index("PlayId", inplace=True)

        play_crps_list = []

        use_tqdm = True
        if use_tqdm:
            play_id_tqdm = tqdm(play_id_list)

        for i, play_id in enumerate(play_id_tqdm):
            play_df = vali_df.xs(key=play_id, drop_level=False).reset_index()
            # play_df = vali_df.query("PlayId == @play_id")
            y_true = play_df["Yards"].iloc[0]
            cdf_arr = _predict_cdf(play_df, model)

            h_arr = np.ones(199)
            h_arr[: (99 + y_true)] = 0
            play_crps = ((cdf_arr - h_arr) ** 2).mean()
            play_crps_list.append(play_crps)
            if (not (i % 100)) or (i == len(play_id_list) - 1):
                play_crps_arr = np.array(play_crps_list)
                metrics_orddict = OrderedDict(
                    [
                        ("crps_mean", play_crps_arr.mean()),
                        ("crps_std", play_crps_arr.std()),
                        ("crps_max", play_crps_arr.max()),
                    ]
                )

                crps_max_play_id = play_id_list[play_crps_arr.argmax()]
                crps_max_play_df = vali_df.xs(
                    key=crps_max_play_id, drop_level=False
                ).reset_index()
                crps_max_play_orddict = crps_max_play_df.query(
                    "NflIdRusher == NflId"
                ).to_dict(orient="records", into=OrderedDict)[0]

                report_orddict = OrderedDict([])
                report_orddict.update(metrics_orddict)
                report_orddict.update(crps_max_play_orddict)
                if use_tqdm:
                    play_id_tqdm.set_postfix(ordered_dict=report_orddict, refresh=True)
                else:
                    print(report_orddict)
            assert not np.isnan(play_crps)

        log.info(metrics_orddict)
        log.info(crps_max_play_orddict)
        log_metrics(dict(metrics_orddict))
        log_params(dict(crps_max_play_orddict))

    return model


def _predict_cdf(test_df, model):
    test_df = preprocess(test_df)

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
    print("Fit model.")
    model = fit_base_model(df, parameters)
    print("Infer.")
    infer(model, parameters)
    print("Completed.")
