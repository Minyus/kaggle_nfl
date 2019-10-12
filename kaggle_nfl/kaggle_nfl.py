import pandas as pd
import numpy as np
import sys
import logging

log = logging.getLogger(__name__)


def fit_base_model(df, parameters):
    train = df

    train = train.drop_duplicates(subset="PlayId")
    dist = train["Yards"].hist(density=True, cumulative=True, bins=200)

    train_own = train[train["FieldPosition"] == train["PossessionTeam"]]
    train_other = train[train["FieldPosition"] != train["PossessionTeam"]]

    own_cdf = np.histogram(train_own["Yards"], bins=range(-100, 100, 1))[
        0
    ].cumsum() / len(train_own)
    other_cdf = np.histogram(train_other["Yards"], bins=range(-100, 100, 1))[
        0
    ].cumsum() / len(train_other)

    model = dict(own_cdf=own_cdf, other_cdf=other_cdf)

    return model


from kaggle.competitions import nflrush


def infer(model, parameters):
    own_cdf = model.get("own_cdf")
    other_cdf = model.get("other_cdf")
    log.info("Started inference.")

    env = nflrush.make_env()
    for (test_df, sample_prediction_df) in env.iter_test():
        if test_df["FieldPosition"].iloc[0] != test_df["PossessionTeam"].iloc[0]:
            # when they are in the opponents half
            cdf = np.copy(other_cdf)
            cdf[-test_df["YardLine"].iloc[0] :] = 1
            sample_prediction_df.iloc[0, :] = cdf
        else:
            # when they are in their own half
            cdf = np.copy(own_cdf)
            cdf[-(100 - (test_df["YardLine"].iloc[0] + 50)) :] = 1
            sample_prediction_df.iloc[0, :] = cdf
        env.predict(sample_prediction_df)
        if sys.version_info >= (3, 6, 8):
            log.info("Completed 1st inference iteration. Skip the rest.")
            return sample_prediction_df

    env.write_submission_file()
    log.info("Completed inference.")
    return sample_prediction_df
