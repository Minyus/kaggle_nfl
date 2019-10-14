import pandas as pd
import numpy as np
import logging

log = logging.getLogger(__name__)

FLOAT_TYPES = ["float16", "float32", "float64"]
INT_TYPES = ["int8", "int16", "int32", "int64"]


def df_merge(**kwargs):
    def _df_merge(left_df, right_df, *argsignore, **kwargsignore):
        kwargs.setdefault("suffixes", (False, False))
        return pd.merge(left_df, right_df, **kwargs)

    return _df_merge


def df_concat(**kwargs):
    def _df_concat(df_0, df_1, *argsignore, **kwargsignore):
        new_col_values = kwargs.get("new_col_values")  # type: List[str]
        new_col_name = kwargs.get("new_col_name")  # type: str
        col_id = kwargs.get("col_id", "index")  # type: str
        sort = kwargs.get("sort", False)  # type: bool

        if col_id:
            df_0.set_index(keys=col_id, inplace=True)
            df_1.set_index(keys=col_id, inplace=True)
        else:
            col_id = df_0.index.name

        assert (isinstance(new_col_values, list) and len(new_col_values) == 2) or (
            new_col_values is None
        )
        names = [new_col_name, col_id] if new_col_name else col_id
        df_0 = pd.concat(
            [df_0, df_1],
            sort=sort,
            verify_integrity=bool(col_id),
            keys=new_col_values,
            names=names,
        )
        df_0.reset_index(inplace=True)
        return df_0

    return _df_concat


def df_sort_values(**kwargs):
    def _df_sort_values(df, *argsignore, **kwargsignore):
        """ https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.sort_values.html """
        kwargs.update(dict(inplace=True))
        df.sort_values(**kwargs)
        return df

    return _df_sort_values


def df_sample(**kwargs):
    def _df_sample(df, *argsignore, **kwargsignore):
        frac = kwargs.get("frac", 1.0)
        random_state = kwargs.get("random_state")
        col_sample = kwargs.get("col_sample")
        col_assign = kwargs.get("col_assign")

        log.info("DF shape before random sampling: {}".format(df.shape))
        if col_sample:
            population_arr = df[col_sample].unique()
            size = int(len(population_arr) * frac)
            np.random.seed(random_state)
            sample_arr = np.random.choice(population_arr, size=size, replace=False)
            if col_assign:
                assert isinstance(col_assign, str)
                sample_set = set(sample_arr.tolist())
                df[col_assign] = df[col_sample].map(
                    lambda v: np.int8(v in sample_set), na_action="ignore"
                )
            else:
                sample_series = pd.Series(sample_arr, name=col_sample)
                df = pd.merge(df, sample_series, how="right", on=col_sample)
        if not col_sample:
            if col_assign:
                raise NotImplementedError("Specify col_sample to use col_assign.")
            df = df.sample(frac=frac, random_state=random_state)
        log.info("DF shape after random sampling: {}".format(df.shape))
        return df

    return _df_sample


def df_filter(**kwargs):
    def _df_filter(df, *argsignore, **kwargsignore):
        items = kwargs.get("items")
        if items:
            if isinstance(items, str):
                items = [items]
            assert isinstance(items, list), "'items' should be a list or string"
            missing = [item for item in items if item not in df.columns]
            if missing:
                log.warning("filter could not find columns: {}".format(missing))
            items = [item for item in items if item in df.columns]
            kwargs.update(dict(items=items))
        return df.filter(**kwargs)

    return _df_filter


def df_get_cols(**kwargs):
    def _df_get_cols(df, *argsignore, **kwargsignore):
        return df.columns.to_list()

    return _df_get_cols


def df_filter_cols(**kwargs):
    def _df_filter_cols(df, *argsignore, **kwargsignore):
        return df_filter(**kwargs)(df, *argsignore, **kwargsignore).columns.to_list()

    return _df_filter_cols


def df_select_dtypes(**kwargs):
    def _df_select_dtypes(df, *argsignore, **kwargsignore):
        return df.select_dtypes(**kwargs)

    return _df_select_dtypes


def df_select_dtypes_cols(**kwargs):
    def _df_select_dtypes_cols(df, *argsignore, **kwargsignore):
        return df_select_dtypes(**kwargs)(
            df, *argsignore, **kwargsignore
        ).columns.to_list()

    return _df_select_dtypes_cols


def df_get_col_indexes(**kwargs):
    def _df_get_col_indexes(df, *argsignore, **kwargsignore):
        cols = kwargs.get("cols")
        assert cols
        for col in cols:
            if col not in df.columns:
                log.warning("Could not find column: {}".format(col))
        cols = [col for col in cols if col in df.columns.to_list()]
        indices = [df.columns.to_list().index(col) for col in cols]
        return indices

    return _df_get_col_indexes


def df_drop(**kwargs):
    def _df_drop(df, *argsignore, **kwargsignore):
        kwargs.update(dict(inplace=True))
        df.drop(**kwargs)
        return df

    return _df_drop


def df_drop_filter(**kwargs):
    def _df_drop_filter(df, *argsignore, **kwargsignore):
        cols_drop = df_filter(**kwargs)(
            df, *argsignore, **kwargsignore
        ).columns.to_list()
        df.drop(columns=cols_drop, inplace=True)
        return df

    return _df_drop_filter


def df_add_row_stat(**kwargs):
    def _df_add_row_stat(df, *argsignore, **kwargsignore):
        regex = kwargs.get("regex", r".*")
        prefix = kwargs.get("prefix", "stat_all")

        prefix = prefix or regex
        t_df = df.filter(regex=regex, axis=1)
        cols_float = t_df.select_dtypes(include=FLOAT_TYPES).columns.to_list()
        cols_int = t_df.select_dtypes(include=INT_TYPES).columns.to_list()

        if cols_float:
            df["{}_float_na_".format(prefix)] = (
                df[cols_float].isna().astype("int8").sum(axis=1)
            )

            df["{}_float_zero_".format(prefix)] = (
                (df[cols_float] == 0.0).astype("int8").sum(axis=1)
            )
            df["{}_float_pos_".format(prefix)] = (
                (df[cols_float] > 0.0).astype("int8").sum(axis=1)
            )
            df["{}_float_neg_".format(prefix)] = (
                (df[cols_float] < 0.0).astype("int8").sum(axis=1)
            )
            df["{}_float_pos_ones_".format(prefix)] = (
                (df[cols_float] == 1.0).astype("int8").sum(axis=1)
            )
            df["{}_float_neg_ones_".format(prefix)] = (
                (df[cols_float] == -1.0).astype("int8").sum(axis=1)
            )
            df["{}_float_gt_pos_ones_".format(prefix)] = (
                (df[cols_float] > 1.0).astype("int8").sum(axis=1)
            )
            df["{}_float_lt_neg_ones_".format(prefix)] = (
                (df[cols_float] < -1.0).astype("int8").sum(axis=1)
            )
            df["{}_float_max_".format(prefix)] = df[cols_float].max(axis=1)
            df["{}_float_min_".format(prefix)] = df[cols_float].min(axis=1)
            df["{}_float_mean_".format(prefix)] = df[cols_float].mean(axis=1)

        if cols_int:
            df["{}_int_zero_".format(prefix)] = (
                (df[cols_int] == 0).astype("int8").sum(axis=1)
            )
            df["{}_int_pos_".format(prefix)] = (
                (df[cols_int] > 0).astype("int8").sum(axis=1)
            )
            df["{}_int_neg_".format(prefix)] = (
                (df[cols_int] < 0).astype("int8").sum(axis=1)
            )
            df["{}_int_pos_ones_".format(prefix)] = (
                (df[cols_int] == 1).astype("int8").sum(axis=1)
            )
            df["{}_int_neg_ones_".format(prefix)] = (
                (df[cols_int] == -1).astype("int8").sum(axis=1)
            )
            df["{}_int_gt_pos_ones_".format(prefix)] = (
                (df[cols_int] > 1).astype("int8").sum(axis=1)
            )
            df["{}_int_lt_neg_ones_".format(prefix)] = (
                (df[cols_int] < -1).astype("int8").sum(axis=1)
            )
            df["{}_int_max_".format(prefix)] = df[cols_int].max(axis=1)
            df["{}_int_min_".format(prefix)] = df[cols_int].min(axis=1)
            df["{}_int_mean_".format(prefix)] = df[cols_int].mean(axis=1)

        return df

    return _df_add_row_stat


def df_query(**kwargs):
    def _df_query(df, *argsignore, **kwargsignore):
        kwargs.update(dict(inplace=True))
        df.query(**kwargs)
        return df

    return _df_query


def df_drop_duplicates(**kwargs):
    def _df_drop_duplicates(df, *argsignore, **kwargsignore):
        kwargs.update(dict(inplace=True))
        df.drop_duplicates(**kwargs)
        return df

    return _df_drop_duplicates


def df_apply(**kwargs):
    def _df_apply(df, *argsignore, **kwargsignore):
        return df.apply(**kwargs)

    return _df_apply


def df_applymap(**kwargs):
    def _df_applymap(df, *argsignore, **kwargsignore):
        return df.applymap(**kwargs)

    return _df_applymap


def df_pipe(**kwargs):
    def _df_pipe(df, *argsignore, **kwargsignore):
        return df.pipe(**kwargs)

    return _df_pipe


def df_agg(**kwargs):
    def _df_agg(df, *argsignore, **kwargsignore):
        return df.agg(**kwargs)

    return _df_agg


def df_aggregate(**kwargs):
    def _df_aggregate(df, *argsignore, **kwargsignore):
        return df.aggregate(**kwargs)

    return _df_aggregate


def df_transform(**kwargs):
    def _df_transform(df, *argsignore, **kwargsignore):
        return df.transform(**kwargs)

    return _df_transform


def df_groupby(**kwargs):
    def _df_groupby(df, *argsignore, **kwargsignore):
        return df.groupby(**kwargs)

    return _df_groupby


def df_rolling(**kwargs):
    def _df_rolling(df, *argsignore, **kwargsignore):
        return df.rolling(**kwargs)

    return _df_rolling


def df_expanding(**kwargs):
    def _df_expanding(df, *argsignore, **kwargsignore):
        return df.expanding(**kwargs)

    return _df_expanding


def df_ewm(**kwargs):
    def _df_ewm(df, *argsignore, **kwargsignore):
        return df.ewm(**kwargs)

    return _df_ewm


def sr_map(**kwargs):
    def _sr_map(sr, *argsignore, **kwargsignore):
        sr.map(**kwargs)

    return _sr_map
