""" kaggle_nfl.py """

import pandas as pd
import numpy as np
import random
from collections import OrderedDict
import math

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


def df_concat(**kwargs):
    def _df_concat(df_0, df_1, *argsignore, **kwargsignore):
        new_col_values = kwargs.get("new_col_values")  # type: List[str]
        new_col_name = kwargs.get("new_col_name")  # type: str
        col_id = kwargs.get("col_id")  # type: str
        sort = kwargs.get("sort", False)  # type: bool

        if col_id:
            df_0.set_index(keys=col_id, inplace=True)
            df_1.set_index(keys=col_id, inplace=True)
        else:
            col_id = df_0.index.name

        assert (isinstance(new_col_values, list) and len(new_col_values) == 2) or (new_col_values is None)
        names = [new_col_name, col_id] if new_col_name else col_id
        df_0 = pd.concat([df_0, df_1], sort=sort, verify_integrity=bool(col_id), keys=new_col_values, names=names)
        if new_col_name:
            df_0.reset_index(inplace=True, level=new_col_name)
        return df_0

    return _df_concat


def _groupby(df, groupby, columns):
    if not isinstance(groupby, dict):
        groupby = dict(by=groupby)
    if groupby:
        df = df.groupby(**groupby)
    if columns:
        df = df[columns]
    return df


def df_transform(groupby=None, columns=None, **kwargs):
    def _df_transform(df, *argsignore, **kwargsignore):
        df = _groupby(df, groupby, columns)
        return df.transform(**kwargs)

    return _df_transform


def df_duplicate(**kwargs):
    columns = kwargs.get("columns")
    assert columns and isinstance(columns, dict)
    col_list = list(columns.keys())

    def _df_duplicate(df, *argsignore, **kwargsignore):
        for col in col_list:
            assert col in df.columns, "{} not in the data frame.".format(col)
        new_df = df[col_list].rename(**kwargs)
        df = pd.concat([df, new_df], axis=1, sort=False)
        return df

    return _df_duplicate


def df_cond_replace(cond, columns, value=np.nan, **kwargs):
    def _df_cond_replace(df, *argsignore, **kwargsignore):
        df.loc[df.eval(cond), columns] = value
        return df

    return _df_cond_replace


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
    df["_S"] = df["S"].astype(np.float32)
    df["_A"] = df["A"].astype(np.float32)
    df["_Dis10"] = 10 * df["Dis"].astype(np.float32)
    is2017_sr = df["Season"] == 2017
    # df.loc[is2017_sr, "_S"] = df["_S"] * np.float32(4.56395617070357 / 3.93930840336135)
    # df.loc[is2017_sr, "_A"] = df["_A"] * np.float32(2.72513175405908 / 2.50504453781512)
    # df.loc[is2017_sr, "_Dis10"] = df["_Dis10"] * np.float32(4.458548487 / 4.505504202)

    motion_coef = 1.0
    motion_sr = motion_coef * df["_S"]

    df["_S_X"] = motion_sr * np.sin(df["Dir_std"])
    df["_S_Y"] = motion_sr * np.cos(df["Dir_std"])

    df["X_int_t1"] = X_float + df["_S_X"]
    df["Y_int_t1"] = Y_float + df["_S_Y"]

    df["_S_left"] = df["_S_Y"]
    df = df_cond_replace(cond="_S_left < 0", columns="_S_left", value=0)(df)
    df["_S_right"] = -df["_S_Y"]
    df = df_cond_replace(cond="_S_right < 0", columns="_S_right", value=0)(df)

    """ """

    dis10_motion_sr = motion_coef * df["_Dis10"]
    df["_Dis10_X"] = dis10_motion_sr * np.sin(df["Dir_std"])
    df["_Dis10_Y"] = dis10_motion_sr * np.cos(df["Dir_std"])

    df["X_int_Dis10_t1"] = X_float + df["_Dis10_X"]
    df["Y_int_Dis10_t1"] = Y_float + df["_Dis10_Y"]

    """ """

    df["SeasonCode"] = ((df["Season"].clip(lower=2017, upper=2018) - 2017)).astype(np.uint8)  # 2
    df["DownCode"] = (df["Down"].clip(lower=1, upper=4) - 1).astype(np.uint8)  # 4

    df["HomeOnOffense"] = df["PossessionTeam"] == df["HomeTeamAbbr"]
    df["HomeOnOffenseCode"] = df["HomeOnOffense"].astype(np.uint8)  # 2

    df["OffenceTeamCode"] = df["PossessionTeam"].map(TEAM_CODE_DICT).fillna(0).astype(np.uint8)

    df["DefenceTeamAbbr"] = df["HomeTeamAbbr"]
    df.loc[df["HomeOnOffense"], "DefenceTeamAbbr"] = df["VisitorTeamAbbr"]
    df["DefenceTeamCode"] = df["DefenceTeamAbbr"].map(TEAM_CODE_DICT).fillna(0).astype(np.uint8)

    df["ScoreDiff"] = df["VisitorScoreBeforePlay"] - df["HomeScoreBeforePlay"]
    df.loc[df["HomeOnOffense"], "ScoreDiff"] = -df["ScoreDiff"]
    df["ScoreDiffCode"] = (np.floor(df["ScoreDiff"].clip(lower=-35, upper=35) / 10) + 4).astype(np.uint8)  # 8

    df["YardsToGoalCode"] = np.floor((100 - df["YardsFromOwnGoal"].clip(lower=1, upper=99)) / 2).astype(np.uint8)

    """ """

    df["OffenseFormationCode"] = df["OffenseFormation"].map(OFFENSE_FORMATION_DICT).fillna(0).astype(np.uint8)

    df["DefendersInTheBoxCode"] = df["DefendersInTheBox"].clip(lower=3, upper=11).fillna(0).astype(np.uint8)
    df["PositionCode"] = df["Position"].map(POSITION_DICT).fillna(0).astype(np.uint8)

    """ """

    try:
        df["SnapToHandoffTime"] = (
            pd.to_datetime(df["TimeHandoff"]) - pd.to_datetime(df["TimeSnap"])
        ).dt.total_seconds()
    except:
        log.warning("Failed to compute ScaledRelativeHandoff.")
        df["SnapToHandoffTime"] = np.ones(len(df))
    df["SnapToHandoffTimeCode"] = df["SnapToHandoffTime"].clip(lower=0, upper=4).fillna(1).astype(np.uint8)

    """ """

    # df = df_duplicate(columns={"Dir_std": "Rusher_Dir_std"})(df)
    # df = df_cond_replace(cond="IsBallCarrier == False", columns="Rusher_Dir_std", value=np.nan)(df)
    # df["Rusher_Dir_std"] = df_transform(groupby="PlayId", columns="Rusher_Dir_std", func="max")(df)
    #
    # df["_RusherDirSimilarity"] = np.cos(df["Dir_std"] - df["Rusher_Dir_std"])
    """ """

    # df = df.filter(items=cols)
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


class FieldImagesDataset:
    def __init__(
        self,
        df,
        coo_cols_list=[
            ["X_int", "Y_int"],  # 1st snapshot
            # ["X_int_t1", "Y_int_t1"],  # 2nd snapshot
            ["X_int_Dis10_t1", "Y_int_Dis10_t1"],  # 2nd snapshot
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
        spatial_independent_cols=[
            "YardsToGoalCode",
            "SeasonCode",
            "DownCode",
            "ScoreDiffCode",
            "HomeOnOffenseCode",
            "OffenceTeamCode",
            "DefenceTeamCode",
            "OffenseFormationCode",
            "DefendersInTheBoxCode",
            "PositionCode",
            # "SnapToHandoffTimeCode"
        ],
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
            if t_size == 1:
                agg_df = agg_df_list[0]
                agg_df["T"] = 0
            elif t_size == 2:
                agg_df = df_concat(new_col_name="T", new_col_values=[0, 1])(agg_df_list[0], agg_df_list[1])
            else:
                raise ValueError("coo_cols_list length not supported.")

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

            melted_si_df = None
            if spatial_independent_cols:
                spatial_independent_dict = ordinal_dict(spatial_independent_cols)
                agg_si_df = df.query("PlayerCategory == 2")  # Rusher
                agg_si_df = agg_si_df[["PlayIndex"] + spatial_independent_cols].drop_duplicates().reset_index(drop=True)
                melted_si_df = agg_si_df.melt(id_vars=["PlayIndex"])
                melted_si_df["Channel"] = dim_sizes_[0]
                melted_si_df.rename(columns={"variable": "H", "value": "W"}, inplace=True)
                melted_si_df.loc[:, "H"] = melted_si_df["H"].map(spatial_independent_dict)
                melted_si_df["value"] = 1.0
                # melted_df = pd.concat([melted_df, melted_si_df], sort=False)
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
                    indices_si = play_si_df[dim_cols_].values
                    coo_3d = f(values=values, indices=indices, indices_si=indices_si, size=dim_sizes_)
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
                    values_si_arr = np.ones(shape=indices_si_arr.shape[0], dtype=np.float32)
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

    log.info("Setting up train_dataset from df shape: {}".format(fit_df.shape))
    train_dataset = FieldImagesDataset(fit_df, to_pytorch_tensor=True, augmentation=augmentation)

    log.info("Setting up val_dataset from df shape: {}".format(vali_df.shape))
    val_dataset = FieldImagesDataset(vali_df, to_pytorch_tensor=True)

    return train_dataset, val_dataset


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


def _predict_cdf(test_df, pytorch_model, parameters=None):

    tta = parameters.get("tta")
    augmentation = parameters.get("augmentation")

    yards_abs = test_df["YardsFromOwnGoal"].iloc[0]

    pytorch_model.eval()
    with torch.no_grad():
        if tta:
            imgs_3dtt_list = [
                FieldImagesDataset(test_df, to_pytorch_tensor=True, augmentation=augmentation)[0][0] for _ in range(tta)
            ]
            imgs_4dtt = torch.stack(imgs_3dtt_list, dim=0)
            out_2dtt = pytorch_model(imgs_4dtt)
            pred_arr = torch.mean(out_2dtt, dim=0, keepdim=False).numpy()
        else:
            imgs_3dtt, _ = FieldImagesDataset(test_df, to_pytorch_tensor=True)[0]
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

    h_2dtt = (h_1dtt.reshape(1, -1) >= index_1dtt.reshape(-1, 1)).type(torch.FloatTensor)

    ret = (input - h_2dtt) ** 2
    if reduction != "none":
        ret = torch.mean(ret) if reduction == "mean" else torch.sum(ret)
    return ret


def yards_to_index(y_1dtt):
    return y_1dtt + 99


def nfl_crps_loss(input, target):
    return crps_loss(input, target, target_to_index=yards_to_index)


class NflCrpsLossFunc:
    def __init__(self, min=None, max=None, desc_penalty=None):
        self.min = min
        self.max = max
        self.clip = (min is not None) or (max is not None)
        self.desc_penalty = desc_penalty

    def __call__(self, input, target):
        if self.clip:
            target = torch.clamp(target, min=self.min, max=self.max)
        loss = nfl_crps_loss(input, target)
        if self.desc_penalty:
            penalty_tt = torch.relu(tensor_shift(input, offset=1) - input)
            penalty = torch.mean(penalty_tt)
            loss += penalty * self.desc_penalty
        return loss


def tensor_shift(tt, offset=1):
    out_tt = torch.zeros_like(tt)
    out_tt[:, offset:] = tt[:, :-offset]
    return out_tt


def infer(model, parameters={}):
    from kaggle.competitions import nflrush

    env = nflrush.make_env()
    for (test_df, sample_prediction_df) in env.iter_test():
        test_df = preprocess(test_df)
        sample_prediction_df.iloc[0, :] = _predict_cdf(test_df, model, parameters)
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
        final_dict = dict(final_loss_mean=loss_mean, final_loss_std=loss_std)

    log.info("{}".format(final_dict))
    try:
        from mlflow import log_metrics

        log_metrics(final_dict)
    except:
        log.warning("Failed to log final loss mean and std.")

    loss_df = pd.DataFrame(dict(loss=loss_1dtt.numpy()))

    return loss_df


"""
https://github.com/kornia/kornia/blob/master/kornia/filters/kernels.py
"""

from typing import Tuple, List

import torch
import torch.nn as nn


def normalize_kernel2d(input: torch.Tensor) -> torch.Tensor:
    r"""Normalizes both derivative and smoothing kernel.
    """
    if len(input.size()) < 2:
        raise TypeError("input should be at least 2D tensor. Got {}".format(input.size()))
    norm: torch.Tensor = input.abs().sum(dim=-1).sum(dim=-1)
    return input / (norm.unsqueeze(-1).unsqueeze(-1))


def gaussian(window_size, sigma):
    x = torch.arange(window_size).float() - window_size // 2
    if window_size % 2 == 0:
        x = x + 0.5
    gauss = torch.exp((-x.pow(2.0) / float(2 * sigma ** 2)))
    return gauss / gauss.sum()


def laplacian_1d(window_size) -> torch.Tensor:
    r"""One could also use the Laplacian of Gaussian formula
        to design the filter.
    """

    filter_1d = torch.ones(window_size)
    filter_1d[window_size // 2] = 1 - window_size
    laplacian_1d: torch.Tensor = filter_1d
    return laplacian_1d


def get_box_kernel2d(kernel_size: Tuple[int, int]) -> torch.Tensor:
    r"""Utility function that returns a box filter."""
    kx: float = float(kernel_size[0])
    ky: float = float(kernel_size[1])
    scale: torch.Tensor = torch.tensor(1.0) / torch.tensor([kx * ky])
    tmp_kernel: torch.Tensor = torch.ones(1, kernel_size[0], kernel_size[1])
    return scale.to(tmp_kernel.dtype) * tmp_kernel


def get_binary_kernel2d(window_size: Tuple[int, int]) -> torch.Tensor:
    r"""Creates a binary kernel to extract the patches. If the window size
    is HxW will create a (H*W)xHxW kernel.
    """
    window_range: int = window_size[0] * window_size[1]
    kernel: torch.Tensor = torch.zeros(window_range, window_range)
    for i in range(window_range):
        kernel[i, i] += 1.0
    return kernel.view(window_range, 1, window_size[0], window_size[1])


def get_sobel_kernel_3x3() -> torch.Tensor:
    """Utility function that returns a sobel kernel of 3x3"""
    return torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])


def get_sobel_kernel_5x5_2nd_order() -> torch.Tensor:
    """Utility function that returns a 2nd order sobel kernel of 5x5"""
    return torch.tensor(
        [
            [-1.0, 0.0, 2.0, 0.0, -1.0],
            [-4.0, 0.0, 8.0, 0.0, -4.0],
            [-6.0, 0.0, 12.0, 0.0, -6.0],
            [-4.0, 0.0, 8.0, 0.0, -4.0],
            [-1.0, 0.0, 2.0, 0.0, -1.0],
        ]
    )


def _get_sobel_kernel_5x5_2nd_order_xy() -> torch.Tensor:
    """Utility function that returns a 2nd order sobel kernel of 5x5"""
    return torch.tensor(
        [
            [-1.0, -2.0, 0.0, 2.0, 1.0],
            [-2.0, -4.0, 0.0, 4.0, 2.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [2.0, 4.0, 0.0, -4.0, -2.0],
            [1.0, 2.0, 0.0, -2.0, -1.0],
        ]
    )


def get_diff_kernel_3x3() -> torch.Tensor:
    """Utility function that returns a sobel kernel of 3x3"""
    return torch.tensor([[-0.0, 0.0, 0.0], [-1.0, 0.0, 1.0], [-0.0, 0.0, 0.0]])


def get_sobel_kernel2d() -> torch.Tensor:
    kernel_x: torch.Tensor = get_sobel_kernel_3x3()
    kernel_y: torch.Tensor = kernel_x.transpose(0, 1)
    return torch.stack([kernel_x, kernel_y])


def get_diff_kernel2d() -> torch.Tensor:
    kernel_x: torch.Tensor = get_diff_kernel_3x3()
    kernel_y: torch.Tensor = kernel_x.transpose(0, 1)
    return torch.stack([kernel_x, kernel_y])


def get_sobel_kernel2d_2nd_order() -> torch.Tensor:
    gxx: torch.Tensor = get_sobel_kernel_5x5_2nd_order()
    gyy: torch.Tensor = gxx.transpose(0, 1)
    gxy: torch.Tensor = _get_sobel_kernel_5x5_2nd_order_xy()
    return torch.stack([gxx, gxy, gyy])


def get_diff_kernel2d_2nd_order() -> torch.Tensor:
    gxx: torch.Tensor = torch.tensor([[0.0, 0.0, 0.0], [1.0, -2.0, 1.0], [0.0, 0.0, 0.0]])
    gyy: torch.Tensor = gxx.transpose(0, 1)
    gxy: torch.Tensor = torch.tensor([[-1.0, 0.0, 1.0], [0.0, 0.0, 0.0], [1.0, 0.0, -1.0]])
    return torch.stack([gxx, gxy, gyy])


def get_spatial_gradient_kernel2d(mode: str, order: int) -> torch.Tensor:
    r"""Function that returns kernel for 1st or 2nd order image gradients,
    using one of the following operators: sobel, diff"""
    if mode not in ["sobel", "diff"]:
        raise TypeError(
            "mode should be either sobel\
                         or diff. Got {}".format(
                mode
            )
        )
    if order not in [1, 2]:
        raise TypeError(
            "order should be either 1 or 2\
                         Got {}".format(
                order
            )
        )
    if mode == "sobel" and order == 1:
        kernel: torch.Tensor = get_sobel_kernel2d()
    elif mode == "sobel" and order == 2:
        kernel = get_sobel_kernel2d_2nd_order()
    elif mode == "diff" and order == 1:
        kernel = get_diff_kernel2d()
    elif mode == "diff" and order == 2:
        kernel = get_diff_kernel2d_2nd_order()
    else:
        raise NotImplementedError("")
    return kernel


def get_gaussian_kernel1d(kernel_size: int, sigma: float, force_even: bool = False) -> torch.Tensor:
    r"""Function that returns Gaussian filter coefficients.
    Args:
        kernel_size (int): filter size. It should be odd and positive.
        sigma (float): gaussian standard deviation.
        force_even (bool): overrides requirement for odd kernel size.
    Returns:
        Tensor: 1D tensor with gaussian filter coefficients.
    Shape:
        - Output: :math:`(\text{kernel_size})`
    Examples::
        >>> kornia.image.get_gaussian_kernel(3, 2.5)
        tensor([0.3243, 0.3513, 0.3243])
        >>> kornia.image.get_gaussian_kernel(5, 1.5)
        tensor([0.1201, 0.2339, 0.2921, 0.2339, 0.1201])
    """
    if not isinstance(kernel_size, int) or ((kernel_size % 2 == 0) and not force_even) or (kernel_size <= 0):
        raise TypeError("kernel_size must be an odd positive integer. " "Got {}".format(kernel_size))
    window_1d: torch.Tensor = gaussian(kernel_size, sigma)
    return window_1d


def get_gaussian_kernel2d(
    kernel_size: Tuple[int, int], sigma: Tuple[float, float], force_even: bool = False
) -> torch.Tensor:
    r"""Function that returns Gaussian filter matrix coefficients.
    Args:
        kernel_size (Tuple[int, int]): filter sizes in the x and y direction.
         Sizes should be odd and positive.
        sigma (Tuple[int, int]): gaussian standard deviation in the x and y
         direction.
        force_even (bool): overrides requirement for odd kernel size.
    Returns:
        Tensor: 2D tensor with gaussian filter matrix coefficients.
    Shape:
        - Output: :math:`(\text{kernel_size}_x, \text{kernel_size}_y)`
    Examples::
        >>> kornia.image.get_gaussian_kernel2d((3, 3), (1.5, 1.5))
        tensor([[0.0947, 0.1183, 0.0947],
                [0.1183, 0.1478, 0.1183],
                [0.0947, 0.1183, 0.0947]])
        >>> kornia.image.get_gaussian_kernel2d((3, 5), (1.5, 1.5))
        tensor([[0.0370, 0.0720, 0.0899, 0.0720, 0.0370],
                [0.0462, 0.0899, 0.1123, 0.0899, 0.0462],
                [0.0370, 0.0720, 0.0899, 0.0720, 0.0370]])
    """
    if not isinstance(kernel_size, tuple) or len(kernel_size) != 2:
        raise TypeError("kernel_size must be a tuple of length two. Got {}".format(kernel_size))
    if not isinstance(sigma, tuple) or len(sigma) != 2:
        raise TypeError("sigma must be a tuple of length two. Got {}".format(sigma))
    ksize_x, ksize_y = kernel_size
    sigma_x, sigma_y = sigma
    kernel_x: torch.Tensor = get_gaussian_kernel1d(ksize_x, sigma_x, force_even)
    kernel_y: torch.Tensor = get_gaussian_kernel1d(ksize_y, sigma_y, force_even)
    kernel_2d: torch.Tensor = torch.matmul(kernel_x.unsqueeze(-1), kernel_y.unsqueeze(-1).t())
    return kernel_2d


def get_laplacian_kernel1d(kernel_size: int) -> torch.Tensor:
    r"""Function that returns the coefficients of a 1D Laplacian filter.
    Args:
        kernel_size (int): filter size. It should be odd and positive.
    Returns:
        Tensor (float): 1D tensor with laplacian filter coefficients.
    Shape:
        - Output: math:`(\text{kernel_size})`
    Examples::
        >>> kornia.image.get_laplacian_kernel(3)
        tensor([ 1., -2.,  1.])
        >>> kornia.image.get_laplacian_kernel(5)
        tensor([ 1.,  1., -4.,  1.,  1.])
    """
    if not isinstance(kernel_size, int) or kernel_size % 2 == 0 or kernel_size <= 0:
        raise TypeError("ksize must be an odd positive integer. Got {}".format(kernel_size))
    window_1d: torch.Tensor = laplacian_1d(kernel_size)
    return window_1d


def get_laplacian_kernel2d(kernel_size: int) -> torch.Tensor:
    r"""Function that returns Gaussian filter matrix coefficients.
    Args:
        kernel_size (int): filter size should be odd.
    Returns:
        Tensor: 2D tensor with laplacian filter matrix coefficients.
    Shape:
        - Output: :math:`(\text{kernel_size}_x, \text{kernel_size}_y)`
    Examples::
        >>> kornia.image.get_laplacian_kernel2d(3)
        tensor([[ 1.,  1.,  1.],
                [ 1., -8.,  1.],
                [ 1.,  1.,  1.]])
        >>> kornia.image.get_laplacian_kernel2d(5)
        tensor([[  1.,   1.,   1.,   1.,   1.],
                [  1.,   1.,   1.,   1.,   1.],
                [  1.,   1., -24.,   1.,   1.],
                [  1.,   1.,   1.,   1.,   1.],
                [  1.,   1.,   1.,   1.,   1.]])
    """
    if not isinstance(kernel_size, int) or kernel_size % 2 == 0 or kernel_size <= 0:
        raise TypeError("ksize must be an odd positive integer. Got {}".format(kernel_size))

    kernel = torch.ones((kernel_size, kernel_size))
    mid = kernel_size // 2
    kernel[mid, mid] = 1 - kernel_size ** 2
    kernel_2d: torch.Tensor = kernel
    return kernel_2d


"""
https://github.com/kornia/kornia/blob/master/kornia/filters/filter.py
"""

from typing import Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

# from kornia.filters.kernels import normalize_kernel2d


def compute_padding(kernel_size: Tuple[int, int]) -> List[int]:
    """Computes padding tuple."""
    # 4 ints:  (padding_left, padding_right,padding_top,padding_bottom)
    # https://pytorch.org/docs/stable/nn.html#torch.nn.functional.pad
    assert len(kernel_size) == 2, kernel_size
    computed = [(k - 1) // 2 for k in kernel_size]
    return [computed[1], computed[1], computed[0], computed[0]]


def filter2D(
    input: torch.Tensor, kernel: torch.Tensor, border_type: str = "reflect", normalized: bool = False
) -> torch.Tensor:
    r"""Function that convolves a tensor with a kernel.
    The function applies a given kernel to a tensor. The kernel is applied
    independently at each depth channel of the tensor. Before applying the
    kernel, the function applies padding according to the specified mode so
    that the output remains in the same shape.
    Args:
        input (torch.Tensor): the input tensor with shape of
          :math:`(B, C, H, W)`.
        kernel (torch.Tensor): the kernel to be convolved with the input
          tensor. The kernel shape must be :math:`(B, kH, kW)`.
        border_type (str): the padding mode to be applied before convolving.
          The expected modes are: ``'constant'``, ``'reflect'``,
          ``'replicate'`` or ``'circular'``. Default: ``'reflect'``.
        normalized (bool): If True, kernel will be L1 normalized.
    Return:
        torch.Tensor: the convolved tensor of same size and numbers of channels
        as the input.
    """
    if not isinstance(input, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(type(input)))

    if not isinstance(kernel, torch.Tensor):
        raise TypeError("Input kernel type is not a torch.Tensor. Got {}".format(type(kernel)))

    if not isinstance(border_type, str):
        raise TypeError("Input border_type is not string. Got {}".format(type(kernel)))

    if not len(input.shape) == 4:
        raise ValueError("Invalid input shape, we expect BxCxHxW. Got: {}".format(input.shape))

    if not len(kernel.shape) == 3:
        raise ValueError("Invalid kernel shape, we expect BxHxW. Got: {}".format(kernel.shape))

    borders_list: List[str] = ["constant", "reflect", "replicate", "circular"]
    if border_type not in borders_list:
        raise ValueError(
            "Invalid border_type, we expect the following: {0}." "Got: {1}".format(borders_list, border_type)
        )

    # prepare kernel
    b, c, h, w = input.shape
    tmp_kernel: torch.Tensor = kernel.to(input.device).to(input.dtype)
    tmp_kernel = tmp_kernel.repeat(c, 1, 1, 1)
    if normalized:
        tmp_kernel = normalize_kernel2d(tmp_kernel)
    # pad the input tensor
    height, width = tmp_kernel.shape[-2:]
    padding_shape: List[int] = compute_padding((height, width))
    input_pad: torch.Tensor = F.pad(input, padding_shape, mode=border_type)

    # convolve the tensor with the kernel
    return F.conv2d(input_pad, tmp_kernel, padding=0, stride=1, groups=c)


""" 
https://github.com/kornia/kornia/blob/master/kornia/filters/gaussian.py 
"""

from typing import Tuple

import torch
import torch.nn as nn

# import kornia
# from kornia.filters.kernels import get_gaussian_kernel2d


class GaussianBlur2d(nn.Module):
    r"""Creates an operator that blurs a tensor using a Gaussian filter.
    The operator smooths the given tensor with a gaussian kernel by convolving
    it to each channel. It suports batched operation.
    Arguments:
        kernel_size (Tuple[int, int]): the size of the kernel.
        sigma (Tuple[float, float]): the standard deviation of the kernel.
        borde_type (str): the padding mode to be applied before convolving.
          The expected modes are: ``'constant'``, ``'reflect'``,
          ``'replicate'`` or ``'circular'``. Default: ``'reflect'``.
    Returns:
        Tensor: the blurred tensor.
    Shape:
        - Input: :math:`(B, C, H, W)`
        - Output: :math:`(B, C, H, W)`
    Examples::
        >>> input = torch.rand(2, 4, 5, 5)
        >>> gauss = kornia.filters.GaussianBlur((3, 3), (1.5, 1.5))
        >>> output = gauss(input)  # 2x4x5x5
    """

    def __init__(self, kernel_size: Tuple[int, int], sigma: Tuple[float, float], border_type: str = "reflect") -> None:
        super(GaussianBlur2d, self).__init__()
        self.kernel_size: Tuple[int, int] = kernel_size
        self.sigma: Tuple[float, float] = sigma
        self.kernel: torch.Tensor = torch.unsqueeze(get_gaussian_kernel2d(kernel_size, sigma), dim=0)

        assert border_type in ["constant", "reflect", "replicate", "circular"]
        self.border_type = border_type

    def __repr__(self) -> str:
        return (
            self.__class__.__name__
            + "(kernel_size="
            + str(self.kernel_size)
            + ", "
            + "sigma="
            + str(self.sigma)
            + ", "
            + "border_type="
            + self.border_type
            + ")"
        )

    def forward(self, x: torch.Tensor):  # type: ignore
        return filter2D(x, self.kernel, self.border_type)


######################
# functional interface
######################


def gaussian_blur2d(
    input: torch.Tensor, kernel_size: Tuple[int, int], sigma: Tuple[float, float], border_type: str = "reflect"
) -> torch.Tensor:
    r"""Function that blurs a tensor using a Gaussian filter.
    See :class:`~kornia.filters.GaussianBlur` for details.
    """
    return GaussianBlur2d(kernel_size, sigma, border_type)(input)


""" """


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

    if ("ModuleConcat" not in dir()) or ("HatchDict" not in dir()):
        from kedex import *

    conf_logging = yaml.safe_load(logging_yaml)
    logging.config.dictConfig(conf_logging)

    parameters = yaml.safe_load(params_yaml)
    parameters["MODULE_ALIASES"] = {"kedex": "__main__", "kaggle_nfl.kaggle_nfl": "__main__"}
    train_params = HatchDict(parameters).get("train_params")
    train_params["progress_update"] = False
    train_params.pop("val_data_loader_params")
    train_params.pop("evaluate_val_data")
    pytorch_model = HatchDict(parameters).get("pytorch_model")
    augmentation = parameters.get("augmentation")

    log.info("Read CSV file.")
    df = pd.read_csv("../input/nfl-big-data-bowl-2020/train.csv", low_memory=False)

    log.info("Preprocess.")
    df = preprocess(df)

    log.info("Set up dataset.")
    train_dataset = FieldImagesDataset(df, to_pytorch_tensor=True, augmentation=augmentation)

    log.info("Fit model.")
    model = neural_network_train(train_params=train_params, mlflow_logging=False)(pytorch_model, train_dataset)

    log.info("Infer.")
    infer(model, parameters)

    log.info("Completed.")
