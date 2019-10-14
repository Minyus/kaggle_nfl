import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path

if sys.version_info >= (3, 6, 8):
    from skew_scaler import SkewScaler


def fit_base_model(df, parameters):
    play_df = df.drop_duplicates(subset="PlayId")

    # play_df["Yards"].clip(lower=-10, upper=50, inplace=True)

    model = SkewScaler()

    model.fit(play_df["Yards"])

    return model


def _predict_cdf(test_df, model):
    play_arr = np.concatenate(
        (np.zeros(90), model.cdf(np.arange(-9, 50, 1)), np.ones(50)), axis=None
    )
    return play_arr


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
