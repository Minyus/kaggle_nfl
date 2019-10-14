import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path

if sys.version_info >= (3, 6, 8):
    from skew_scaler import SkewScaler
    from mlflow import log_metrics, log_params
    from tqdm import tqdm
    import logging

    log = logging.getLogger(__name__)


def fit_base_model(df, parameters):

    # play_df["Yards"].clip(lower=-10, upper=50, inplace=True)

    model = SkewScaler()

    if "Validation" in df.columns:
        fit_df = df.query("Validation == 0").drop(columns=["Validation"])
        log.info("Fitting model using data shape: {}".format(fit_df.shape))
    else:
        fit_df = df

    model.fit(fit_df.drop_duplicates(subset="PlayId")["Yards"])

    if "Validation" in df.columns:
        vali_df = df.query("Validation == 1").drop(columns=["Validation"])
        play_id_sr = vali_df["PlayId"].drop_duplicates()
        play_id_list = play_id_sr.to_list()

        vali_df.set_index("PlayId", inplace=True)

        play_crps_list = []
        for play_id in tqdm(play_id_list):
            play_df = vali_df.xs(key=play_id, drop_level=False).reset_index()
            # play_df = vali_df.query("PlayId == @play_id")
            y_true = play_df["Yards"].iloc[0]
            cdf_arr = _predict_cdf(play_df, model)

            h_arr = np.ones(199)
            h_arr[: (99 + y_true)] = 0
            play_crps = ((cdf_arr - h_arr) ** 2).mean()
            play_crps_list.append(play_crps)

        play_crps_arr = np.array(play_crps_list)
        metrics = dict(
            crps_mean=play_crps_arr.mean(),
            crps_std=play_crps_arr.std(),
            crps_max=play_crps_arr.max(),
        )
        log.info(metrics)
        log_metrics(metrics)

        crps_max_play_id = play_id_list[play_crps_arr.argmax()]
        crps_max_play_df = vali_df.xs(
            key=crps_max_play_id, drop_level=False
        ).reset_index()
        crps_max_play = crps_max_play_df.query("NflIdRusher == NflId").to_dict(
            "records"
        )[0]
        log.info(crps_max_play)
        log_params(crps_max_play)

    return model


def _predict_cdf(test_df, model):
    yard_line = test_df["YardLine"].max()

    pred_arr = np.zeros(199)
    pred_arr[-100:] = np.ones(100)

    cdf_arr = model.cdf(np.arange(-yard_line, 100 - yard_line, 1))
    cdf_arr = np.maximum.accumulate(cdf_arr)
    # for i in range(len(cdf_arr) - 1):
    #     cdf_arr[i + 1] = max(cdf_arr[i + 1], cdf_arr[i])

    pred_arr[(99 - yard_line) : (199 - yard_line)] = cdf_arr
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
