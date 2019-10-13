import pandas as pd


def df_query(**kwargs):
    def _df_query(df, parameters):
        kwargs.update(dict(inplace=True))
        df.query(**kwargs)
        return df

    return _df_query
