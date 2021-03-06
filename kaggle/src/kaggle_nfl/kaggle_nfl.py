""" kaggle_nfl.py """

from importlib.util import find_spec

if find_spec("pipelinex"):
    from pipelinex import *

import pandas as pd
import random
import math
from itertools import chain
from scipy.stats import lognorm

import logging

log = logging.getLogger(__name__)


TEAM_CODE_DICT = dict(
    ARZ=0,
    ATL=1,
    BLT=2,
    BUF=3,
    CAR=4,
    CHI=5,
    CIN=6,
    CLV=7,
    DAL=8,
    DEN=9,
    DET=10,
    GB=11,
    HST=12,
    IND=13,
    JAX=14,
    KC=15,
    LA=16,
    LAC=17,
    MIA=18,
    MIN=19,
    NE=20,
    NO=21,
    NYG=22,
    NYJ=23,
    OAK=24,
    PHI=25,
    PIT=26,
    SEA=27,
    SF=28,
    TB=29,
    TEN=30,
    WAS=31,
)


OFFENSE_FORMATION_LIST = r"""_
ACE
EMPTY
I_FORM
JUMBO
PISTOL
SHOTGUN
SINGLEBACK
WILDCAT
""".splitlines()
OFFENSE_FORMATION_DICT = {e: i for i, e in enumerate(OFFENSE_FORMATION_LIST)}

POSITION_LIST = r"""_
CB
DE
DT
FB
HB
QB
RB
TE
WR
""".splitlines()
POSITION_DICT = {e: i for i, e in enumerate(POSITION_LIST)}

DROP_LIST = r"""Team
NflId
DisplayName
JerseyNumber
FieldPosition
OffenseFormation
OffensePersonnel
DefensePersonnel
TimeSnap
HomeTeamAbbr
VisitorTeamAbbr
PlayerHeight
PlayerWeight
PlayerBirthDate
PlayerCollegeName
Stadium
Location
StadiumType
Turf
GameWeather
WindSpeed
WindDirection
""".splitlines()


zeros11 = np.zeros((11, 11))
ones11 = np.ones((11, 11))
bipart_mask_2darr = np.block([[zeros11, ones11], [ones11, zeros11]])


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
    df["TeamOnOffense"] = (df["PossessionTeam"] == df["HomeTeamAbbr"]).map(home_dict)

    df["IsOnOffense"] = df["Team"] == df["TeamOnOffense"]
    df["YardsFromOwnGoal"] = -df["YardLine"] + 100
    df.loc[(df["FieldPosition"].astype(str) == df["PossessionTeam"]), "YardsFromOwnGoal"] = df["YardLine"]

    df["X_std"] = df["X"]
    df.loc[df["ToLeft"], "X_std"] = -df["X"] + 120
    df["X_std"] = df["X_std"] - 10

    df["Y_std"] = df["Y"]
    df.loc[df["ToLeft"], "Y_std"] = -df["Y"] + 53.6

    """ """
    df["PlayerCategory"] = df["IsOnOffense"].astype(np.uint8)
    df.loc[df["IsBallCarrier"], "PlayerCategory"] = 2

    X_float = df["X_std"] - df["YardsFromOwnGoal"] + 10
    Y_float = df["Y_std"]

    X_float[df["PlayerCategory"] == 0] = X_float + 0.5  # separate defense and offense

    df["X_int"] = X_float
    df["Y_int"] = Y_float

    """ """
    # df["Dir_rad"] = np.mod(90 - df["Dir"], 360) * math.pi / 180.0
    # df["Dir_std"] = df["Dir_rad"]
    # df.loc[df["ToLeft"], "Dir_std"] = np.mod(np.pi + df.loc[df["ToLeft"], "Dir_rad"], 2 * np.pi)
    df["Dir_std_2"] = df["Dir"] - 180 * df["ToLeft"].astype(np.float32)
    df["Dir_std_2"].fillna(90, inplace=True)
    df["Dir_std"] = df["Dir_std_2"] * math.pi / 180.0

    # df.rename(columns=dict(S="_S", A="_A"), inplace=True)
    df["_A"] = df["A"].astype(np.float32)
    df["_S"] = df["S"].astype(np.float32)
    df["_Dis10"] = 10 * df["Dis"].astype(np.float32)
    # is2017_sr = df["Season"] == 2017
    # df.loc[is2017_sr, "_S"] = df["_S"] * np.float32(4.56395617070357 / 3.93930840336135)
    # df.loc[is2017_sr, "_A"] = df["_A"] * np.float32(2.72513175405908 / 2.50504453781512)
    # df.loc[is2017_sr, "_Dis10"] = df["_Dis10"] * np.float32(4.458548487 / 4.505504202)

    # normal_dis10_flag_sr = df["_Dis10"] < 5.8
    # df.loc[normal_dis10_flag_sr, "_S"] = df["_Dis10"]

    df["_A"].clip(lower=0, upper=5.84, inplace=True)
    df["_S"].clip(lower=0, upper=7.59, inplace=True)
    df["_Dis10"].clip(lower=0, upper=7.59, inplace=True)
    # df["_S"] = 0.5 * df["_S"] + 0.5 * df["_Dis10"]

    motion_coef = 1.0
    motion_sr = motion_coef * df["_S"]

    df["_S_X"] = motion_sr * np.sin(df["Dir_std"])
    df["_S_Y"] = motion_sr * np.cos(df["Dir_std"])

    df["X_int_t1"] = X_float + df["_S_X"]
    df["Y_int_t1"] = Y_float + df["_S_Y"]

    """ """

    # df = DfRelative(
    #     flag="IsBallCarrier==False",
    #     columns={"X_int": "X_int_rr", "Y_int": "Y_int_rr", "X_int_t1": "X_int_t1_rr", "Y_int_t1": "Y_int_t1_rr"},
    #     groupby="PlayId",
    # )(df)
    #
    # df = DfEval(expr="X_int_rr = X_int_rr + 5")(df)
    # df = DfEval(expr="Y_int_rr = Y_int_rr + 26.8")(df)
    # df = DfEval(expr="X_int_t1_rr = X_int_t1_rr + 5")(df)
    # df = DfEval(expr="Y_int_t1_rr = Y_int_t1_rr + 26.8")(df)

    """ """

    df["SeasonCode"] = ((df["Season"].clip(lower=2017, upper=2018) - 2017)).astype(np.uint8)  # 2
    df["DownCode"] = (df["Down"].clip(lower=1, upper=5) - 1).astype(np.uint8)  # 5

    df["HomeOnOffense"] = df["PossessionTeam"] == df["HomeTeamAbbr"]
    df["HomeOnOffenseCode"] = df["HomeOnOffense"].astype(np.uint8)  # 2

    # df["OffenceTeamCode"] = df["PossessionTeam"].map(TEAM_CODE_DICT).fillna(0).astype(np.uint8)
    #
    # df["DefenceTeamAbbr"] = df["HomeTeamAbbr"]
    # df.loc[df["HomeOnOffense"], "DefenceTeamAbbr"] = df["VisitorTeamAbbr"]
    # df["DefenceTeamCode"] = df["DefenceTeamAbbr"].map(TEAM_CODE_DICT).fillna(0).astype(np.uint8)

    # df["ScoreDiff"] = df["VisitorScoreBeforePlay"] - df["HomeScoreBeforePlay"]
    # df.loc[df["HomeOnOffense"], "ScoreDiff"] = -df["ScoreDiff"]
    # df["ScoreDiffCode"] = (np.floor(df["ScoreDiff"].clip(lower=-35, upper=35) / 10) + 4).astype(np.uint8)  # 8

    df["YardsToGoal"] = 100 - df["YardsFromOwnGoal"].clip(lower=1, upper=99)
    df["YardsToGoalCode"] = np.floor(df["YardsToGoal"] / 2).astype(np.uint8)
    df["YardsToGoalP10Val"] = np.floor((df["YardsToGoal"] + 10).clip(upper=99)).astype(np.uint8)
    """ """

    df["OffenseFormationCode"] = df["OffenseFormation"].map(OFFENSE_FORMATION_DICT).fillna(0).astype(np.uint8)

    df["DefendersInTheBoxCode"] = df["DefendersInTheBox"].clip(lower=3, upper=11).fillna(0).astype(np.uint8)
    # df["PositionCode"] = df["Position"].map(POSITION_DICT).fillna(0).astype(np.uint8)

    """ """

    # try:
    #     df["SnapToHandoffTime"] = (
    #         pd.to_datetime(df["TimeHandoff"]) - pd.to_datetime(df["TimeSnap"])
    #     ).dt.total_seconds()
    # except:
    #     log.warning("Failed to compute ScaledRelativeHandoff.")
    #     df["SnapToHandoffTime"] = np.ones(len(df))
    # df["SnapToHandoffTimeCode"] = df["SnapToHandoffTime"].clip(lower=0, upper=4).fillna(1).astype(np.uint8)

    """ """

    df = DfFocusTransform(
        focus="PlayerCategory == 2",
        columns={
            "X_int": "X_Rusher",
            "Y_int": "Y_Rusher",
            # "_A": "A_Rusher",
            # "_S_X": "S_X_Rusher",
            # "_S_Y": "S_Y_Rusher",
        },
        func=np.max,
        groupby="PlayId",
        keep_others=True,
    )(df)

    df = DfEval("X_RR = X_int - X_Rusher")(df)
    df = DfEval("Y_RR = Y_int - Y_Rusher")(df)
    # df["D_RR"] = df[["X_RR", "Y_RR"]].apply(np.linalg.norm, axis=1)
    # df.sort_values(by=["PlayId", "PlayerCategory", "D_RR"], inplace=True)

    df = DfFocusTransform(
        focus="PlayerCategory == 0",
        columns={
            "X_int": "X_Defense_Max",
            "X_RR": "X_RR_Defense_Max",
            "Y_RR": "Y_RR_Defense_Max",
            # "D_RR": "D_RR_Defense_Max",
            # "_A": "A_Defense_Max",
            # "_S_X": "S_X_Defense_Max",
            # "_S_Y": "S_Y_Defense_Max",
        },
        func=np.max,
        groupby="PlayId",
        keep_others=True,
    )(df)
    df = DfFocusTransform(
        focus="PlayerCategory == 0",
        columns={
            "X_int": "X_Defense_Min",
            "X_RR": "X_RR_Defense_Min",
            "Y_RR": "Y_RR_Defense_Min",
            # "D_RR": "D_RR_Defense_Min",
            # "_A": "A_Defense_Min",
            # "_S_X": "S_X_Defense_Min",
            # "_S_Y": "S_Y_Defense_Min",
        },
        func=np.min,
        groupby="PlayId",
        keep_others=True,
    )(df)
    df = DfFocusTransform(
        focus="PlayerCategory == 0",
        columns={
            "X_int": "X_Defense_Mean",
            "X_RR": "X_RR_Defense_Mean",
            "Y_RR": "Y_RR_Defense_Mean",
            # "D_RR": "D_RR_Defense_Mean",
            # "_A": "A_Defense_Mean",
            # "_S_X": "S_X_Defense_Mean",
            # "_S_Y": "S_Y_Defense_Mean",
        },
        func=np.mean,
        groupby="PlayId",
        keep_others=True,
    )(df)
    df = DfFocusTransform(
        focus="PlayerCategory == 0",
        columns={
            "X_int": "X_Defense_Stdev",
            "X_RR": "X_RR_Defense_Stdev",
            "Y_RR": "Y_RR_Defense_Stdev",
            # "D_RR": "D_RR_Defense_Stdev",
            # "_A": "A_Defense_Stdev",
            # "_S_X": "S_X_Defense_Stdev",
            # "_S_Y": "S_Y_Defense_Stdev",
        },
        func=np.std,
        groupby="PlayId",
        keep_others=True,
    )(df)

    df = DfFocusTransform(
        focus="PlayerCategory == 1",
        columns={
            "X_int": "X_Offense_Max",
            "X_RR": "X_RR_Offense_Max",
            "Y_RR": "Y_RR_Offense_Max",
            # "D_RR": "D_RR_Offense_Max",
            # "_A": "A_Offense_Max",
            # "_S_X": "S_X_Offense_Max",
            # "_S_Y": "S_Y_Offense_Max",
        },
        func=np.max,
        groupby="PlayId",
        keep_others=True,
    )(df)
    df = DfFocusTransform(
        focus="PlayerCategory == 1",
        columns={
            "X_int": "X_Offense_Min",
            "X_RR": "X_RR_Offense_Min",
            "Y_RR": "Y_RR_Offense_Min",
            # "D_RR": "D_RR_Offense_Min",
            # "_A": "A_Offense_Min",
            # "_S_X": "S_X_Offense_Min",
            # "_S_Y": "S_Y_Offense_Min",
        },
        func=np.min,
        groupby="PlayId",
        keep_others=True,
    )(df)
    df = DfFocusTransform(
        focus="PlayerCategory == 1",
        columns={
            "X_int": "X_Offense_Mean",
            "X_RR": "X_RR_Offense_Mean",
            "Y_RR": "Y_RR_Offense_Mean",
            # "D_RR": "D_RR_Offense_Mean",
            # "_A": "A_Offense_Mean",
            # "_S_X": "S_X_Offense_Mean",
            # "_S_Y": "S_Y_Offense_Mean",
        },
        func=np.mean,
        groupby="PlayId",
        keep_others=True,
    )(df)
    df = DfFocusTransform(
        focus="PlayerCategory == 1",
        columns={
            "X_int": "X_Offense_Stdev",
            "X_RR": "X_RR_Offense_Stdev",
            "Y_RR": "Y_RR_Offense_Stdev",
            # "D_RR": "D_RR_Offense_Stdev",
            # "_A": "A_Offense_Stdev",
            # "_S_X": "S_X_Offense_Stdev",
            # "_S_Y": "S_Y_Offense_Stdev",
        },
        func=np.std,
        groupby="PlayId",
        keep_others=True,
    )(df)

    """ """

    # df = DfSpatialFeatures(
    #     output="n_connected",
    #     coo_cols=["X_int", "Y_int"],
    #     groupby="PlayId",
    #     affinity_scale="PlayerCategory == 0",
    #     col_name_fmt="Defense_NConn",
    #     binary_affinity=True,
    #     unit_distance=5.0,
    #     keep_others=True,
    #     sort=True,
    # )(df)
    #
    # df = DfSpatialFeatures(
    #     output="n_connected",
    #     coo_cols=["X_int", "Y_int"],
    #     groupby="PlayId",
    #     affinity_scale="PlayerCategory != 0",
    #     col_name_fmt="Offense_NConn",
    #     binary_affinity=True,
    #     unit_distance=5.0,
    #     keep_others=True,
    #     sort=True,
    # )(df)

    # df = DfSpatialFeatures(
    #     output="n_connected",
    #     coo_cols=["X_int", "Y_int"],
    #     groupby="PlayId",
    #     affinity_scale=bipart_mask_2darr,
    #     col_name_fmt="Bipart_NConn",
    #     binary_affinity=True,
    #     unit_distance=5.0,
    #     keep_others=True,
    #     sort=True,
    # )(df)

    """ """

    df.query(expr="Season >= 2018", inplace=True)

    df.drop(columns=DROP_LIST, inplace=True)

    return df


def _relative_values(abs_sr, comp_sr, offset=101, transform_func=None):
    transform_func = np.log10
    if len(comp_sr) != len(abs_sr):
        comp_sr = comp_sr.iloc[0]
    denominator_sr = comp_sr + offset
    assert (denominator_sr > 0.0) if isinstance(denominator_sr, float) else (denominator_sr > 0.0).all()

    numerator_sr = abs_sr + offset
    values_sr = numerator_sr / denominator_sr
    assert not values_sr.isna().any()
    if transform_func and callable(transform_func):
        values_sr = transform_func(values_sr)
    return values_sr


class BaseProbas:
    def __init__(self, groupby=None, yards_query="-10 <= Yards < 40"):
        self.groupby = groupby
        self.yards_query = yards_query
        self.agg_df = None

    def fit(self, df):
        df = df.copy()
        if self.groupby is None:
            self.keys = ["CONSTANT"]
            df["CONSTANT"] = 0.0
        elif isinstance(self.groupby, str):
            self.keys = [self.groupby]
        elif isinstance(self.groupby, list):
            self.keys = self.groupby
        else:
            raise ValueError

        agg_df = DfAgg(groupby=self.keys + ["Yards"], columns="PlayId", count_yards=("PlayId", "count"))(df)
        agg_df.reset_index(drop=False, inplace=True)
        agg_df = DfDuplicate(columns={"count_yards": "count_total"})(agg_df)
        agg_df = DfTransform(groupby=self.keys, columns="count_total", func="sum", keep_others=True)(agg_df)
        agg_df.reset_index(drop=False, inplace=True)
        agg_df = DfQuery(expr=self.yards_query)(agg_df)

        agg_df = DfEval("H = 0 \n W = Yards + 10 \n value = (count_yards / count_total)")(agg_df)
        agg_df = DfFilter(items=self.keys + ["H", "W", "value"])(agg_df)
        agg_df = DfSortValues(by=self.keys + ["H", "W"])(agg_df)
        self.agg_df = agg_df

    def transform(self, df):
        assert isinstance(df, pd.DataFrame)
        assert self.agg_df is not None, "BasePrabas needs to be fitted before calling transform."
        if self.groupby is None:
            df = df.copy()
            df["CONSTANT"] = 0.0
        return pd.merge(left=df, right=self.agg_df, how="left", on=self.keys)

    def fit_transform(self, df):
        self.fit(df)
        return self.transform(df)


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


def ordinal_dict(ls):
    return {ls[i]: i for i in range(len(ls))}


CONTINUOUS_COLS = [
    """
YardsToGoalP10Val
X_Defense_Max
X_RR_Defense_Max
Y_RR_Defense_Max
X_Offense_Max
X_RR_Offense_Max
Y_RR_Offense_Max
X_Defense_Min
X_RR_Defense_Min
Y_RR_Defense_Min
X_Offense_Min
X_RR_Offense_Min
Y_RR_Offense_Min
X_Defense_Mean
X_RR_Defense_Mean
Y_RR_Defense_Mean
X_Offense_Mean
X_RR_Offense_Mean
Y_RR_Offense_Mean
X_RR_Defense_Stdev
Y_RR_Defense_Stdev
X_RR_Offense_Stdev
Y_RR_Offense_Stdev
X_Rusher
Y_Rusher
""".strip().splitlines(),
    #     """
    # A_Defense_Max
    # S_X_Defense_Max
    # S_Y_Defense_Max
    # A_Offense_Max
    # S_X_Offense_Max
    # S_Y_Offense_Max
    # A_Rusher
    # S_X_Rusher
    # S_Y_Rusher
    # """.strip().splitlines(),
    #     """
    # A_Defense_Min
    # S_X_Defense_Min
    # S_Y_Defense_Min
    # A_Offense_Min
    # S_X_Offense_Min
    # S_Y_Offense_Min
    # A_Rusher
    # S_X_Rusher
    # S_Y_Rusher
    # """.strip().splitlines(),
    #     """
    # A_Defense_Mean
    # S_X_Defense_Mean
    # S_Y_Defense_Mean
    # A_Offense_Mean
    # S_X_Offense_Mean
    # S_Y_Offense_Mean
    # A_Rusher
    # S_X_Rusher
    # S_Y_Rusher
    # """.strip().splitlines(),
    #     """
    # A_Defense_Stdev
    # S_X_Defense_Stdev
    # S_Y_Defense_Stdev
    # A_Offense_Stdev
    # S_X_Offense_Stdev
    # S_Y_Offense_Stdev
    # A_Rusher
    # S_X_Rusher
    # S_Y_Rusher
    # """.strip().splitlines(),
]

CATEGORICAL_COLS = [
    # "YardsToGoalCode",
    # "SeasonCode",
    "DownCode",
    # "ScoreDiffCode",
    "HomeOnOffenseCode",
    # "OffenceTeamCode",
    # "DefenceTeamCode",
    "OffenseFormationCode",
    "DefendersInTheBoxCode",
    # "PositionCode",
    # "SnapToHandoffTimeCode"
    # "Defense_NConn",
    # "Offense_NConn",
    # "Bipart_NConn",
]


class FieldImagesDataset:
    def __init__(
        self,
        df,
        base_probas,
        coo_cols_list=[
            ["X_int", "Y_int"],  # 1st snapshot
            ["X_int_t1", "Y_int_t1"],  # 2nd snapshot
            # ["X_int_rr", "Y_int_rr"],  # 3rd snapshot
            # ["X_int_t1_rr", "Y_int_t1_rr"],  # 4th snapshot
        ],
        coo_size=[30, 54],
        value_cols=[
            # "_count",
            # "_S",
            "_A",
            "_S_X",
            "_S_Y",
            # "_S_left",
            # "_S_right",
            # "_Dis10_X",
            # "_Dis10_Y",
        ],
        to_pytorch_tensor=False,
        store_as_sparse_tensor=False,
        augmentation={},
        transform=None,
        target_transform=None,
    ):

        if "Yards" not in df.columns:
            df["Yards"] = np.nan

        play_target_df = df[["PlayId", "Yards"]].drop_duplicates().reset_index(drop=True)
        self.len = len(play_target_df)
        self.target_dict = play_target_df["Yards"].to_dict()

        play_id_dict = play_target_df["PlayId"].to_dict()
        self.play_id_dict = play_id_dict
        play_index_dict = {v: k for k, v in play_id_dict.items()}
        df["PlayIndex"] = df["PlayId"].map(play_index_dict)

        df["_count"] = np.float32(1 / 255.0)

        dim_col = "PlayerCategory"
        dim_size = 3

        if 1:  # to_pytorch_tensor:
            coo_cols_ = ["H", "W"]
            dim_cols_ = ["Channel"] + coo_cols_
            agg_df_list = []
            for coo_cols in coo_cols_list:
                agg_df = df.groupby(["PlayIndex", dim_col] + coo_cols, as_index=False)[value_cols].sum()
                agg_df.rename(columns={coo_cols[0]: coo_cols_[0], coo_cols[1]: coo_cols_[1]}, inplace=True)
                agg_df_list.append(agg_df)

            t_size = len(coo_cols_list)
            agg_df = DfConcat(new_col_name="T")(*agg_df_list)

            melted_df = agg_df.melt(id_vars=["PlayIndex", "T", dim_col] + coo_cols_)
            value_cols_dict = ordinal_dict(value_cols)
            # melted_df["Channel"] = (
            #     melted_df["T"] * dim_size * len(value_cols)
            #     + melted_df[dim_col] * len(value_cols)
            #     + melted_df["variable"].map(value_cols_dict)
            # )
            # melted_df["Channel"] = (
            #     melted_df["T"]
            #     + t_size * melted_df[dim_col]
            #     + t_size * dim_size * melted_df["variable"].map(value_cols_dict)
            # )
            melted_df["Channel"] = (
                melted_df[dim_col]
                + dim_size * melted_df["T"]
                + dim_size * t_size * melted_df["variable"].map(value_cols_dict)
            )

            melted_df.loc[:, "value"] = melted_df["value"].astype(np.float32)
            melted_df.set_index("PlayIndex", inplace=True)

            dim_sizes_ = [dim_size * len(value_cols) * len(coo_cols_list)] + coo_size

            spatial_independent_cols = list(chain.from_iterable(CONTINUOUS_COLS)) + CATEGORICAL_COLS
            melted_si_df = None
            if spatial_independent_cols:

                rusher_df = df.query("PlayerCategory == 2")  # Rusher
                agg_si_df = (
                    rusher_df[["PlayIndex"] + spatial_independent_cols].copy().drop_duplicates().reset_index(drop=True)
                )
                melted_si_df = agg_si_df.melt(id_vars=["PlayIndex"])
                melted_si_df["Channel"] = dim_sizes_[0]
                melted_si_df["H"] = 0
                melted_si_df["W"] = copy.deepcopy(melted_si_df["value"].values)

                """ Categorical """
                categorical_cols_dict = ordinal_dict(CATEGORICAL_COLS)
                melted_si_df.loc[melted_si_df["variable"].isin(CATEGORICAL_COLS), "H"] = (
                    melted_si_df["variable"].map(categorical_cols_dict)
                    + len(CONTINUOUS_COLS)
                    + int(base_probas is not None)
                )
                melted_si_df.loc[melted_si_df["variable"].isin(CATEGORICAL_COLS), "value"] = 1.0

                """ Continuous """
                for i, cont_cols_1d in enumerate(CONTINUOUS_COLS):
                    melted_si_df.loc[melted_si_df["variable"].isin(cont_cols_1d), "H"] = i + int(
                        base_probas is not None
                    )
                    continuous_cols_dict = ordinal_dict(cont_cols_1d)
                    melted_si_df.loc[melted_si_df["variable"].isin(cont_cols_1d), "W"] = melted_si_df["variable"].map(
                        continuous_cols_dict
                    )

                """ Base probas """
                if base_probas is not None:
                    base_probas_df = base_probas.transform(rusher_df[["PlayIndex"]])
                    base_probas_df["Channel"] = dim_sizes_[0]
                    melted_si_df = DfConcat()(melted_si_df, base_probas_df)

                melted_si_df.sort_values(by=["PlayIndex", "Channel", "H", "W"], inplace=True)

                melted_si_df.loc[:, "value"] = melted_si_df["value"].astype(np.float32)

                melted_si_df.set_index("PlayIndex", inplace=True)

                dim_sizes_[0] += 1

            f = torch.sparse_coo_tensor if store_as_sparse_tensor else dict
            coo_dict = dict()
            for pi in play_id_dict.keys():
                play_df = melted_df.xs(pi)
                values = play_df["value"].values
                indices = play_df[dim_cols_].values
                if melted_si_df is not None:
                    play_si_df = melted_si_df.xs(pi)
                    values_si = play_si_df["value"].values
                    indices_si = play_si_df[dim_cols_].values
                    coo_3d = f(
                        values=values, indices=indices, values_si=values_si, indices_si=indices_si, size=dim_sizes_
                    )
                else:
                    coo_3d = f(values=values, indices=indices, size=dim_sizes_)
                coo_dict[pi] = coo_3d

        self.coo_dict = coo_dict

        self.to_pytorch_tensor = to_pytorch_tensor
        self.store_as_sparse_tensor = store_as_sparse_tensor

        assert isinstance(augmentation, dict)
        self.augmentation = augmentation

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):

        play_coo_3d = self.coo_dict[index]

        if self.to_pytorch_tensor:

            if not self.store_as_sparse_tensor:
                size = play_coo_3d.get("size")
                indices_arr = play_coo_3d["indices"]

                if self.augmentation:
                    horizontal_flip_proba = self.augmentation.get("horizontal_flip_proba")
                    horizontal_shift_std = self.augmentation.get("horizontal_shift_std")
                    vertical_shift_std = self.augmentation.get("vertical_shift_std")
                    if horizontal_flip_proba:
                        indices_arr = _add_horizontal_flip(indices_arr, horizontal_flip_proba, size)
                    if horizontal_shift_std:
                        indices_arr = _add_normal_horizontal_shift(indices_arr, horizontal_shift_std)
                    if vertical_shift_std:
                        indices_arr = _add_normal_vertical_shift(indices_arr, vertical_shift_std)

                indices_si_arr = play_coo_3d.get("indices_si", None)
                if indices_si_arr is not None:
                    indices_arr = np.concatenate([indices_arr, indices_si_arr], axis=0)
                indices_arr[:, 1] = indices_arr[:, 1].clip(0, size[1] - 1)
                indices_arr[:, 2] = indices_arr[:, 2].clip(0, size[2] - 1)

                indices_arr = np.floor(indices_arr).astype(np.int64)
                indices_2dtt = torch.from_numpy(indices_arr.transpose())

                values_arr = play_coo_3d.get("values")
                if indices_si_arr is not None:
                    values_si_arr = play_coo_3d.get("values_si")
                    # values_si_arr = np.ones(shape=indices_si_arr.shape[0], dtype=np.float32)
                    values_arr = np.concatenate([values_arr, values_si_arr], axis=0)
                values_1dtt = torch.from_numpy(values_arr)
                play_coo_3dtt = torch.sparse_coo_tensor(indices=indices_2dtt, values=values_1dtt, size=size)
            img = play_coo_3dtt.to_dense()

        else:
            play_coo_dict = play_coo_3d
            img_ch_2darr_list = [play_coo_dict[ci].toarray() for ci in range(3)]
            img = np.stack(img_ch_2darr_list, axis=2)

        target = self.target_dict[index]

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return self.len


def _add_normal_shift(a, stdev=1.0, size=(1,)):
    return a + stdev * np.random.standard_normal(size=size)


def _add_normal_horizontal_shift(indices_arr, horizontal_shift_std):
    indices_arr[:, 2:] = _add_normal_shift(indices_arr[:, 2:], horizontal_shift_std)
    return indices_arr


def _add_normal_vertical_shift(indices_arr, vertical_shift_std):
    indices_arr[:, 1:2] = _add_normal_shift(indices_arr[:, 1:2], vertical_shift_std)
    return indices_arr


def _random_cond(p=None):
    return p and random.random() < p


def _random_flip(indices_arr, p, size, dim):
    assert isinstance(indices_arr, np.ndarray)
    if _random_cond(p):
        indices_arr[:, dim] = (size[dim] - 1) - indices_arr[:, dim]
    return indices_arr


def _add_horizontal_flip(indices_arr, horizontal_flip_proba, size):
    if _random_cond(horizontal_flip_proba):
        indices_arr = _random_flip(indices_arr, p=horizontal_flip_proba, size=size, dim=2)
    return indices_arr


def _add_vertical_flip(indices_arr, horizontal_flip_proba, size):
    if _random_cond(horizontal_flip_proba):
        indices_arr = _random_flip(indices_arr, p=horizontal_flip_proba, size=size, dim=1)
    return indices_arr


def generate_datasets(df, parameters=None):

    if "Validation" in df.columns:
        fit_df = df.query("Validation == 0").drop(columns=["Validation"])
        vali_df = df.query("Validation == 1").drop(columns=["Validation"])
    else:
        fit_df = df
        vali_df = df

    augmentation = parameters.get("augmentation", dict())
    base_probas = BaseProbas()
    base_probas.fit(fit_df)

    log.info("Setting up train_dataset from df shape: {}".format(fit_df.shape))
    train_dataset = FieldImagesDataset(fit_df, base_probas, to_pytorch_tensor=True, augmentation=augmentation)

    log.info("Setting up val_dataset from df shape: {}".format(vali_df.shape))
    val_dataset = FieldImagesDataset(vali_df, base_probas, to_pytorch_tensor=True)

    return train_dataset, val_dataset, base_probas


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


def _predict_cdf(test_df, pytorch_model, base_probas, parameters=None):

    tta = parameters.get("tta")
    augmentation = parameters.get("augmentation")

    yards_abs = test_df["YardsFromOwnGoal"].iloc[0]
    yards_abs = int(yards_abs)

    pytorch_model.eval()
    with torch.no_grad():
        if tta:
            imgs_3dtt_list = [
                FieldImagesDataset(test_df, base_probas, to_pytorch_tensor=True, augmentation=augmentation)[0][0]
                for _ in range(tta)
            ]
            imgs_4dtt = torch.stack(imgs_3dtt_list, dim=0)
            out_2dtt = pytorch_model(imgs_4dtt)
            pred_arr = torch.mean(out_2dtt, dim=0, keepdim=False).numpy()
        else:
            imgs_3dtt, _ = FieldImagesDataset(test_df, base_probas, to_pytorch_tensor=True)[0]
            imgs_4dtt = torch.unsqueeze(imgs_3dtt, 0)  # instead of DataLoader
            out_2dtt = pytorch_model(imgs_4dtt)
            pred_arr = torch.squeeze(out_2dtt).numpy()

    pred_arr = np.maximum.accumulate(pred_arr)
    pred_arr[: (99 - yards_abs)] = 0.0
    pred_arr[(199 - yards_abs) :] = 1.0
    return pred_arr


def crps_loss(input, target, l1=False, target_to_index=None, reduction="mean"):
    index_1dtt = target_to_index(target) if target_to_index else target
    h_1dtt = torch.arange(input.shape[1])

    h_2dtt = (h_1dtt.reshape(1, -1) >= index_1dtt.reshape(-1, 1)).type(torch.FloatTensor)

    if l1:
        ret = torch.abs(input - h_2dtt)
    else:
        ret = (input - h_2dtt) ** 2

    if reduction != "none":
        ret = torch.mean(ret) if reduction == "mean" else torch.sum(ret)
    return ret


def yards_to_index(y_1dtt):
    return y_1dtt + 99


def nfl_crps_loss(input, target):
    return crps_loss(input, target, target_to_index=yards_to_index)


def nfl_l1crps_loss(input, target):
    return crps_loss(input, target, l1=True, target_to_index=yards_to_index)


class NflCrpsLossFunc:
    def __init__(self, min=None, max=None, desc_penalty=None, l1=False):
        self.min = min
        self.max = max
        self.clip = (min is not None) or (max is not None)
        self.desc_penalty = desc_penalty
        self.l1 = l1

    def __call__(self, input, target):
        if self.clip:
            target = torch.clamp(target, min=self.min, max=self.max)
        if self.l1:
            loss = nfl_l1crps_loss(input, target)
        else:
            loss = nfl_crps_loss(input, target)
        if self.desc_penalty:
            penalty_tt = torch.relu(tensor_shift(input, offset=1) - input)
            penalty = torch.mean(penalty_tt)
            loss += penalty * self.desc_penalty
        return loss


class NflL1CrpsLossFunc(NflCrpsLossFunc):
    def __init__(self, **kwargs):
        super().__init__(l1=True, **kwargs)


def tensor_shift(tt, offset=1):
    out_tt = torch.zeros_like(tt)
    out_tt[:, offset:] = tt[:, :-offset]
    return out_tt


def infer(model, base_probas=None, transformer=None, parameters={}):
    from kaggle.competitions import nflrush

    env = nflrush.make_env()
    for (test_df, sample_prediction_df) in env.iter_test():
        test_df = preprocess(test_df)
        if transformer is not None:
            test_df = transformer.transform(test_df)
        sample_prediction_df.iloc[0, :] = _predict_cdf(test_df, model, base_probas, parameters)
        env.predict(sample_prediction_df)

    env.write_submission_file()

    return sample_prediction_df


def get_test_df(parameters={}):
    from kaggle.competitions import nflrush

    env = nflrush.make_env()

    test_df_list = []
    for (test_df, sample_prediction_df) in env.iter_test():
        test_df_list.append(test_df)
        env.predict(sample_prediction_df)

    test_df = pd.concat(test_df_list)

    return test_df


def final_validation(dataset, pytorch_model, parameters={}):
    tta = parameters.get("tta")
    if tta:
        dataset.augmentation = parameters.get("augmentation", {})
    else:
        tta = 1

    train_params = parameters.get("train_params", {})
    val_dataset_size_limit = train_params.get("val_dataset_size_limit")
    if val_dataset_size_limit and val_dataset_size_limit < len(dataset):
        n_samples = val_dataset_size_limit
    else:
        n_samples = len(dataset)

    from tqdm import trange

    pytorch_model.eval()
    with torch.no_grad():
        pred_1dtt_list = []
        target_0dtt_list = []
        for i in trange(n_samples):
            imgs_3dtt_list = []
            for _ in range(tta):
                imgs_3dtt, target = dataset[i]
                imgs_3dtt_list.append(imgs_3dtt)

            imgs_4dtt = torch.stack(imgs_3dtt_list, dim=0)
            out_2dtt = pytorch_model(imgs_4dtt)
            pred_1dtt = torch.mean(out_2dtt, dim=0, keepdim=False)
            pred_1dtt_list.append(pred_1dtt)

            target_0dtt = torch.tensor(target)
            target_0dtt_list.append(target_0dtt)

        pred_2dtt = torch.stack(pred_1dtt_list, dim=0)
        target_1dtt = torch.stack(target_0dtt_list, dim=0)

        loss_2dtt = crps_loss(pred_2dtt, target_1dtt, target_to_index=yards_to_index, reduction="none")
        loss_1dtt = torch.mean(loss_2dtt, dim=1, keepdim=False)
        loss_mean = float(torch.mean(loss_1dtt).numpy())
        loss_std = float(torch.std(loss_1dtt).numpy())
        final_dict = dict(final_crps_mean=loss_mean, final_crps_std=loss_std)

    log.info("{}".format(final_dict))
    try:
        from mlflow import log_metrics

        log_metrics(final_dict)
    except:
        log.warning("Failed to log final loss mean and std.")

    loss_df = pd.DataFrame(dict(loss=loss_1dtt.numpy()))

    return loss_df


logging_yaml = """
version: 1
disable_existing_loggers: False
formatters:
    simple:
        format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


handlers:
    console:
        class: logging.StreamHandler
        level: INFO
        formatter: simple
        stream: ext://sys.stdout

root:
    level: INFO
    handlers: [console]
"""


if __name__ == "__main__":
    import ignite
    import logging.config
    import yaml

    conf_logging = yaml.safe_load(logging_yaml)
    logging.config.dictConfig(conf_logging)

    if "params_yaml" not in dir():
        load_path = Path("../conf/base/parameters.yml")
        with load_path.open("r") as local_file:
            parameters = yaml.safe_load(local_file)
    else:
        parameters = yaml.safe_load(params_yaml)
    parameters["MODULE_ALIASES"] = {"pipelinex": "__main__", "kaggle_nfl.kaggle_nfl": "__main__"}
    train_params = HatchDict(parameters).get("train_params")
    train_params["progress_update"] = False
    train_params.pop("val_data_loader_params")
    train_params.pop("evaluate_val_data")
    q_transformer = HatchDict(parameters).get("q_transformer")
    pytorch_model = HatchDict(parameters).get("pytorch_model")
    augmentation = parameters.get("augmentation")

    log.info("Read CSV file.")
    df = pd.read_csv("../input/nfl-big-data-bowl-2020/train.csv", low_memory=False)

    log.info("Preprocess.")
    df = preprocess(df)

    if q_transformer:
        log.info("Fit transformer and transform.")
        df, transformer = q_transformer(df)
    else:
        transformer = None

    log.info("Set up dataset.")
    base_probas = BaseProbas()
    base_probas.fit(df)
    train_dataset = FieldImagesDataset(df, base_probas, to_pytorch_tensor=True, augmentation=augmentation)

    log.info("Fit model.")
    model = NetworkTrain(train_params=train_params, mlflow_logging=False)(pytorch_model, train_dataset)

    log.info("Infer.")
    infer(model, base_probas, transformer, parameters)

    log.info("Completed.")
