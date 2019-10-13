import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path
from skew_scaler import SkewScaler
import logging

log = logging.getLogger(__name__)


def fit_base_model(df, parameters):
    play_df = df.drop_duplicates(subset="PlayId")
    model = SkewScaler()
    model.fit(play_df["Yards"])
    return model





def infer(model, parameters):
    from kaggle.competitions import nflrush
    log.info("Started inference.")

    env = nflrush.make_env()
    for (test_df, sample_prediction_df) in env.iter_test():
        sample_prediction_df.iloc[0, :] = model.cdf(np.arange(-99, 100, 1))
        env.predict(sample_prediction_df)
        if sys.version_info >= (3, 6, 8):
            log.info("Completed 1st inference iteration. Skip the rest.")
            return sample_prediction_df

    env.write_submission_file()
    log.info("Completed inference.")
    return sample_prediction_df


if __name__ == "__main__":
    if sys.version_info >= (3, 6, 8):
        log.info("Completed 1st inference iteration. Skip the rest.")
        project_path = Path(__file__).resolve().parent.parent.parent

        src_path = project_path / "input" / "nfl-big-data-bowl-2020"
        if str(src_path) not in sys.path:
            sys.path.insert(0, str(src_path))

        if "PYTHONPATH" not in os.environ:
            os.environ["PYTHONPATH"] = src_path

    print("Read CSV file.")
    df = pd.read_csv('../input/nfl-big-data-bowl-2020/train.csv', low_memory=False)
    parameters=None
    print("Fit model.")
    model = fit_base_model(df, parameters)
    print("Infer.")
    infer(model, parameters)
    print("Completed.")