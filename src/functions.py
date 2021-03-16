import sys
import pandas as pd
import numpy as np

import gc
import pathlib
import namegenerator
from sklearn.model_selection import StratifiedKFold, KFold
import category_encoders as ce
from sklearn.metrics import roc_auc_score, mean_squared_error
from tqdm.autonotebook import tqdm
from sklearn.cluster import KMeans
import joblib
import sklearn.preprocessing as prep
import sklearn.pipeline

sys.path.insert(0, r"C:\Users\Pavel\Documents\projects\dzl")
from dzl import SklearnClassifierFoldTrainer, TabNetFoldTrainer, XGBoostClassifierFoldTrainer, \
    LightGBMClassifierFoldTrainer, CatboostClassifierFoldTrainer, LGBMOptunaOptimizer, CatBoostOptunaOptimizer, \
    XGBoostOptunaOptimizer, LogRegOptunaOptimizer, HistGradientBoostingOptunaOptimizer, TabNetOptunaOptimizer, \
    LightGBMRegressorFoldTrainer

from dzl.src.base import DataBlock, create_data_manager, CVTrainer, CVVoteTrainer

encoders_dict = {enc_cls.__name__: enc_cls for enc_cls in [ce.BackwardDifferenceEncoder,
                                                           ce.BaseNEncoder,
                                                           ce.BinaryEncoder,
                                                           ce.CatBoostEncoder,
                                                           ce.CountEncoder,
                                                           ce.GLMMEncoder,
                                                           ce.HashingEncoder,
                                                           ce.HelmertEncoder,
                                                           ce.JamesSteinEncoder,
                                                           ce.LeaveOneOutEncoder,
                                                           ce.MEstimateEncoder,
                                                           ce.OneHotEncoder,
                                                           ce.OrdinalEncoder,
                                                           ce.SumEncoder,
                                                           ce.PolynomialEncoder,
                                                           ce.TargetEncoder,
                                                           ce.WOEEncoder]
                 }


class WidsDataBlock(DataBlock):
    def __init__(self, data: pd.DataFrame, index=None):
        if index is None:
            index = ['encounter_id']
        super().__init__(index=index, data=data)


class WidsCVTrainer(CVTrainer):

    def score(self, typ: str):
        ypred, ytrue = self.predict(typ).align(self.ds.labeled.y, join='right')
        return roc_auc_score(ytrue, ypred)


apache_maps = dict(gcs_eyes_apache=dict(enumerate(['Not tested',
                                                   'Does not open eyes',
                                                   'Opens eyes in response to pain',
                                                   'Opens eyes in response to sound',
                                                   'Opens eyes spontaneously'])),
                   gcs_verbal_apache=dict(enumerate(['Not tested',
                                                     'Makes no sounds',
                                                     'Makes sounds',
                                                     'Words',
                                                     'Confused, disoriented',
                                                     'Oriented, converses normally'])),
                   gcs_motor_apache=dict(enumerate(['Not tested',
                                                    'Makes no movements',
                                                    'Extension to painful stimuli',
                                                    'Abnormal flexion to painful stimuli',
                                                    'Flexion or Withdrawal to painful stimuli',
                                                    'Localizes to painful stimuli',
                                                    'Obeys commands']))
                   )

categorical_feats = ['ethnicity',
                     'gender',
                     'hospital_admit_source',
                     'icu_admit_source',
                     'icu_stay_type',
                     'icu_type',
                     'elective_surgery',
                     'icu_id',
                     # 'readmission_status',
                     'apache_post_operative',
                     'arf_apache',
                     'intubated_apache',
                     'ventilated_apache',
                     'aids',
                     'cirrhosis',
                     'hepatic_failure',
                     'immunosuppression',
                     'leukemia',
                     'lymphoma',
                     'solid_tumor_with_metastasis',
                     'gcs_unable_apache',
                     'gcs_eyes_apache',
                     'gcs_verbal_apache',
                     'gcs_motor_apache',
                     'apache_2_diagnosis',
                     'apache_3j_diagnosis'
                     ]


def impute_column(data, column, target_col='diabetes_mellitus'):
    class WidsColumnImputerCVTrainer(CVTrainer):

        def score(self, typ: str):
            ypred, ytrue = self.predict(typ).align(self.ds.labeled.y, join='right')
            return mean_squared_error(ytrue, ypred)

    dt = data.drop(columns=[target_col]).copy()
    dt[f'is_train_{column}'] = (~dt[column].isnull()).astype(int).values

    X = dt.drop(columns=[column])
    y = dt[column].values
    cat_columns = dt.select_dtypes(['object', 'category']).columns.tolist()
    encoder = ce.CatBoostEncoder(cols=cat_columns)
    encoder.fit(X, y)
    dt = encoder.transform(X)
    dt[column] = y

    data_block = WidsDataBlock(dt)

    ds = create_data_manager(data_block=data_block,
                             cv_column='cv_fold',
                             train_split_column=f'is_train_{column}',
                             label_columns=[column],
                             cv_object=KFold(n_splits=5, shuffle=True, random_state=42)
                             )

    lgbm_params = dict(init_params=dict(metric='rmse',
                                        max_depth=-1,
                                        num_leaves=31,
                                        min_data_per_group=10,
                                        learning_rate=0.1),
                       fit_params=dict(verbose=-1,
                                       early_stopping_rounds=100
                                       )
                       )

    lgbm_trainer = WidsColumnImputerCVTrainer(fold_trainer_cls=LightGBMRegressorFoldTrainer,
                                              ds=ds,
                                              model_name=f'{column}_impute',
                                              params=lgbm_params,
                                              save_path=pathlib.Path('../.')
                                              )
    lgbm_trainer.fit()

    dt = (data.join(lgbm_trainer
                    .predict('tst')
                    .iloc[:, 0]
                    .to_frame(f'tmp_{column}')
                    .groupby(level=0).mean()
                    )
          .assign(**{column: lambda dx: dx[column].fillna(0) + dx[f'tmp_{column}'].fillna(0)})
          .drop(columns=[f'tmp_{column}'])
          )
    return dt


def categorical_encode(data, target_col, encoder_name, encoder_params=None, columns=None):
    if encoder_params is None:
        encoder_params = {}

    if columns is None:
        columns = data.select_dtypes('category').columns.tolist()

    dt = data.copy()
    X = dt.drop(columns=[target_col])
    y = dt[target_col].values
    encoder = encoders_dict[encoder_name](cols=columns, **encoder_params)
    encoder.fit(X, y)
    dt = encoder.transform(X)
    dt[target_col] = y
    return dt


def stage1_feature_engineering(in_data):
    data = in_data.copy()
    data['apache_3j_diagnosis_grp'] = (data.apache_3j_diagnosis // 1).fillna(999999).astype(int).astype(str)
    data['apache_2_diagnosis'] = (data.apache_2_diagnosis // 1).fillna(999999).astype(int).astype(str)
    categorical_feats.append('apache_3j_diagnosis_grp')

    for cat_feat in categorical_feats:
        data[cat_feat] = data[cat_feat].astype('str')

    for col_max in data.columns:
        if not col_max.endswith('_max'):
            continue
        col_min = col_max.replace('_max', '_min')
        col_avg = col_max.replace('_max', '_avg')
        col_rng = col_max.replace('_max', '_rng')

        data[col_avg] = data[col_min].add(data[col_max]).div(2)
        data[col_rng] = data[col_max].subtract(data[col_min])

    for col in data.columns:
        if 'd1' not in col:
            continue
        h_col = col.replace('d1_', 'h1_')
        if h_col not in data.columns:
            continue
        new_col = col.replace('d1_', '') + '_diff'
        data[new_col] = data[col] - data[h_col]

    for col_max in data.columns:
        if not col_max.endswith('_max'):
            continue
        col_min = col_max.replace('_max', '_min')
        col_avg = col_max.replace('_max', '_avg')
        col_rng = col_max.replace('_max', '_rng')

        data[col_min] = data[col_avg] - data[col_rng].div(2)
        data[col_max] = data[col_avg] + data[col_rng].div(2)

    for col in data.columns:
        if 'd1' not in col:
            continue
        h_col = col.replace('d1_', 'h1_')
        if h_col not in data.columns:
            continue
        new_col = col.replace('d1_', '') + '_diff'
        data[new_col] = data[col] - data[h_col]

    return data


def stage3_feature_engineering(in_data):
    data = in_data.copy()
    apache_cols = ['gcs_eyes_apache', 'gcs_motor_apache', 'gcs_verbal_apache']
    data[apache_cols] = data[apache_cols].astype(float)

    data = (data
            .assign(comorbidity_score=lambda dx: dx['aids'].values * 23 +
                                                 dx['cirrhosis'] * 4 +
                                                 dx['hepatic_failure'] * 16 +
                                                 dx['immunosuppression'] * 10 +
                                                 dx['leukemia'] * 10 +
                                                 dx['lymphoma'] * 13 +
                                                 dx['solid_tumor_with_metastasis'] * 11)
            .assign(gcs_sum=lambda dx: dx['gcs_eyes_apache'] +
                                       dx['gcs_motor_apache'] +
                                       dx['gcs_verbal_apache']
                    )
            )

    data['apache_2_diagnosis_type'] = data.apache_2_diagnosis.astype(int).round(-1).fillna(-100).astype('int32')
    data['apache_3j_diagnosis_type'] = data.apache_3j_diagnosis.astype(float).round(-2).fillna(-100).astype('int32')

    data['bmi_type'] = data.bmi.fillna(0).apply(lambda x: 5 * (round(int(x) / 5)))
    data['height_type'] = data.height.fillna(0).apply(lambda x: 5 * (round(int(x) / 5)))
    data['weight_type'] = data.weight.fillna(0).apply(lambda x: 5 * (round(int(x) / 5)))
    data['age_type'] = data.age.fillna(0).apply(lambda x: 10 * (round(int(x) / 10)))
    data['gcs_sum_type'] = data.gcs_sum.fillna(0).apply(lambda x: 2.5 * (round(int(x) / 2.5))).divide(2.5)

    data['apache_3j_diagnosis_x'] = data['apache_3j_diagnosis'].astype('str').str.split('.', n=1, expand=True)[0]
    data['apache_2_diagnosis_x'] = data['apache_2_diagnosis'].astype('str').str.split('.', n=1, expand=True)[0]
    data['apache_3j_diagnosis_split1'] = np.where(data['apache_3j_diagnosis'].isna(), np.nan,
                                                  data['apache_3j_diagnosis'].astype('str').str.split('.', n=1,
                                                                                                      expand=True)[1])

    data['apache_2_diagnosis_split1'] = data['apache_2_diagnosis'].astype(int).apply(lambda x: x % 10)

    IDENTIFYING_COLS = ['age_type', 'height_type', 'ethnicity', 'gender', 'bmi_type']
    data['profile'] = data[IDENTIFYING_COLS].apply(lambda x: hash(tuple(x)), axis=1)

    d_cols = [c for c in data.columns if (c.startswith("d1"))]
    h_cols = [c for c in data.columns if (c.startswith("h1"))]
    data["dailyLabs_row_nan_count"] = data[d_cols].isna().sum(axis=1)
    data["hourlyLabs_row_nan_count"] = data[h_cols].isna().sum(axis=1)
    data["diff_labTestsRun_daily_hourly"] = data["dailyLabs_row_nan_count"] - data["hourlyLabs_row_nan_count"]

    lab_col = [c for c in data.columns if ((c.startswith("h1")) | (c.startswith("d1")))]
    lab_col_names = list(set(list(map(lambda i: i[3: -4], lab_col))))

    first_h = []
    for v in lab_col_names:
        first_h.append(v + "_started_after_firstHour")
        colsx = [x for x in data.columns if v in x]
        data[v + "_nans"] = data.loc[:, colsx].isna().sum(axis=1)
        data[v + "_d1_value_range"] = data[f"d1_{v}_max"].subtract(data[f"d1_{v}_min"])
        data[v + "_h1_value_range"] = data[f"h1_{v}_max"].subtract(data[f"h1_{v}_min"])
        data[v + "_d1_h1_max_eq"] = (data[f"d1_{v}_max"] == data[f"h1_{v}_max"]).astype(np.int8)
        data[v + "_d1_h1_min_eq"] = (data[f"d1_{v}_min"] == data[f"h1_{v}_min"]).astype(np.int8)
        data[v + "_d1_zero_range"] = (data[v + "_d1_value_range"] == 0).astype(np.int8)
        data[v + "_h1_zero_range"] = (data[v + "_h1_value_range"] == 0).astype(np.int8)
        data[v + "_tot_change_value_range_normed"] = abs(
            (data[v + "_d1_value_range"].div(data[v + "_h1_value_range"])))  # .div(df[f"d1_{v}_max"]))
        data[v + "_started_after_firstHour"] = ((data[f"h1_{v}_max"].isna()) & (data[f"h1_{v}_min"].isna())) & (
            ~data[f"d1_{v}_max"].isna())
        data[v + "_day_more_extreme"] = (
                (data[f"d1_{v}_max"] > data[f"h1_{v}_max"]) | (data[f"d1_{v}_min"] < data[f"h1_{v}_min"]))
        data[v + "_day_more_extreme"].fillna(False)

    data["total_Tests_started_After_firstHour"] = data[first_h].sum(axis=1)

    data["total_Tests_started_After_firstHour"].describe()

    data['diasbp_indicator'] = (
            (data['d1_diasbp_invasive_max'] == data['d1_diasbp_max']) & (
            data['d1_diasbp_noninvasive_max'] == data['d1_diasbp_invasive_max']) |
            (data['d1_diasbp_invasive_min'] == data['d1_diasbp_min']) & (
                    data['d1_diasbp_noninvasive_min'] == data['d1_diasbp_invasive_min']) |
            (data['h1_diasbp_invasive_max'] == data['h1_diasbp_max']) & (
                    data['h1_diasbp_noninvasive_max'] == data['h1_diasbp_invasive_max']) |
            (data['h1_diasbp_invasive_min'] == data['h1_diasbp_min']) & (
                    data['h1_diasbp_noninvasive_min'] == data['h1_diasbp_invasive_min'])
    ).astype(np.int8)

    data['mbp_indicator'] = (
            (data['d1_mbp_invasive_max'] == data['d1_mbp_max']) & (
            data['d1_mbp_noninvasive_max'] == data['d1_mbp_invasive_max']) |
            (data['d1_mbp_invasive_min'] == data['d1_mbp_min']) & (
                    data['d1_mbp_noninvasive_min'] == data['d1_mbp_invasive_min']) |
            (data['h1_mbp_invasive_max'] == data['h1_mbp_max']) & (
                    data['h1_mbp_noninvasive_max'] == data['h1_mbp_invasive_max']) |
            (data['h1_mbp_invasive_min'] == data['h1_mbp_min']) & (
                    data['h1_mbp_noninvasive_min'] == data['h1_mbp_invasive_min'])
    ).astype(np.int8)

    data['sysbp_indicator'] = (
            (data['d1_sysbp_invasive_max'] == data['d1_sysbp_max']) & (
            data['d1_sysbp_noninvasive_max'] == data['d1_sysbp_invasive_max']) |
            (data['d1_sysbp_invasive_min'] == data['d1_sysbp_min']) & (
                    data['d1_sysbp_noninvasive_min'] == data['d1_sysbp_invasive_min']) |
            (data['h1_sysbp_invasive_max'] == data['h1_sysbp_max']) & (
                    data['h1_sysbp_noninvasive_max'] == data['h1_sysbp_invasive_max']) |
            (data['h1_sysbp_invasive_min'] == data['h1_sysbp_min']) & (
                    data['h1_sysbp_noninvasive_min'] == data['h1_sysbp_invasive_min'])
    ).astype(np.int8)

    data['d1_mbp_invnoninv_max_diff'] = data['d1_mbp_invasive_max'] - data['d1_mbp_noninvasive_max']
    data['h1_mbp_invnoninv_max_diff'] = data['h1_mbp_invasive_max'] - data['h1_mbp_noninvasive_max']
    data['d1_mbp_invnoninv_min_diff'] = data['d1_mbp_invasive_min'] - data['d1_mbp_noninvasive_min']
    data['h1_mbp_invnoninv_min_diff'] = data['h1_mbp_invasive_min'] - data['h1_mbp_noninvasive_min']
    data['d1_diasbp_invnoninv_max_diff'] = data['d1_diasbp_invasive_max'] - data['d1_diasbp_noninvasive_max']
    data['h1_diasbp_invnoninv_max_diff'] = data['h1_diasbp_invasive_max'] - data['h1_diasbp_noninvasive_max']
    data['d1_diasbp_invnoninv_min_diff'] = data['d1_diasbp_invasive_min'] - data['d1_diasbp_noninvasive_min']
    data['h1_diasbp_invnoninv_min_diff'] = data['h1_diasbp_invasive_min'] - data['h1_diasbp_noninvasive_min']
    data['d1_sysbp_invnoninv_max_diff'] = data['d1_sysbp_invasive_max'] - data['d1_sysbp_noninvasive_max']
    data['h1_sysbp_invnoninv_max_diff'] = data['h1_sysbp_invasive_max'] - data['h1_sysbp_noninvasive_max']
    data['d1_sysbp_invnoninv_min_diff'] = data['d1_sysbp_invasive_min'] - data['d1_sysbp_noninvasive_min']
    data['h1_sysbp_invnoninv_min_diff'] = data['h1_sysbp_invasive_min'] - data['h1_sysbp_noninvasive_min']

    data = data.rename(columns={'pao2_apache': 'pao2fio2ratio_apache', 'ph_apache': 'arterial_ph_apache'})

    for v in ['albumin', 'bilirubin', 'bun', 'glucose', 'hematocrit', 'pao2fio2ratio', 'arterial_ph', 'resprate',
              'sodium',
              'temp', 'wbc', 'creatinine']:
        data[f'{v}_indicator'] = (
                ((data[f'{v}_apache'] == data[f'd1_{v}_max']) & (data[f'd1_{v}_max'] == data[f'h1_{v}_max'])) |
                ((data[f'{v}_apache'] == data[f'd1_{v}_max']) & (data[f'd1_{v}_max'] == data[f'd1_{v}_min'])) |
                ((data[f'{v}_apache'] == data[f'd1_{v}_max']) & (data[f'd1_{v}_max'] == data[f'h1_{v}_min'])) |
                ((data[f'{v}_apache'] == data[f'h1_{v}_max']) & (data[f'h1_{v}_max'] == data[f'd1_{v}_max'])) |
                ((data[f'{v}_apache'] == data[f'h1_{v}_max']) & (data[f'h1_{v}_max'] == data[f'h1_{v}_min'])) |
                ((data[f'{v}_apache'] == data[f'h1_{v}_max']) & (data[f'h1_{v}_max'] == data[f'd1_{v}_min'])) |
                ((data[f'{v}_apache'] == data[f'd1_{v}_min']) & (data[f'd1_{v}_min'] == data[f'd1_{v}_max'])) |
                ((data[f'{v}_apache'] == data[f'd1_{v}_min']) & (data[f'd1_{v}_min'] == data[f'h1_{v}_min'])) |
                ((data[f'{v}_apache'] == data[f'd1_{v}_min']) & (data[f'd1_{v}_min'] == data[f'h1_{v}_max'])) |
                ((data[f'{v}_apache'] == data[f'h1_{v}_min']) & (data[f'h1_{v}_min'] == data[f'h1_{v}_max'])) |
                ((data[f'{v}_apache'] == data[f'h1_{v}_min']) & (data[f'h1_{v}_min'] == data[f'd1_{v}_min'])) |
                ((data[f'{v}_apache'] == data[f'h1_{v}_min']) & (data[f'h1_{v}_min'] == data[f'd1_{v}_max']))
        ).astype(np.int8)

    more_extreme_cols = [c for c in data.columns if (c.endswith("_day_more_extreme"))]
    data["total_day_more_extreme"] = data[more_extreme_cols].sum(axis=1)

    data["d1_resprate_div_mbp_min"] = data["d1_resprate_min"].div(data["d1_mbp_min"])
    data["d1_resprate_div_sysbp_min"] = data["d1_resprate_min"].div(data["d1_sysbp_min"])
    data["d1_lactate_min_div_diasbp_min"] = data["d1_lactate_min"].div(data["d1_diasbp_min"])
    data["d1_heartrate_min_div_d1_sysbp_min"] = data["d1_heartrate_min"].div(data["d1_sysbp_min"])
    data["d1_hco3_div"] = data["d1_hco3_max"].div(data["d1_hco3_min"])
    data["d1_resprate_times_resprate"] = data["d1_resprate_min"].multiply(data["d1_resprate_max"])
    data["left_average_spo2"] = (2 * data["d1_spo2_max"] + data["d1_spo2_min"]) / 3
    data["total_chronic"] = data[["aids", "cirrhosis", 'hepatic_failure']].sum(axis=1)
    data["total_cancer_immuno"] = data[
        ['immunosuppression', 'leukemia', 'lymphoma', 'solid_tumor_with_metastasis']].sum(
        axis=1)
    data["has_complicator"] = data[["aids", "cirrhosis", 'hepatic_failure',
                                    'immunosuppression', 'leukemia', 'lymphoma', 'solid_tumor_with_metastasis']].max(
        axis=1)
    data[["has_complicator", "total_chronic", "total_cancer_immuno", "has_complicator"]].describe()

    data['apache_3j'] = np.where(data['apache_3j_diagnosis_type'] < 0, np.nan,
                                 np.where(data['apache_3j_diagnosis_type'] < 200, 'Cardiovascular',
                                          np.where(data['apache_3j_diagnosis_type'] < 400, 'Respiratory',
                                                   np.where(data['apache_3j_diagnosis_type'] < 500, 'Neurological',
                                                            np.where(data['apache_3j_diagnosis_type'] < 600, 'Sepsis',
                                                                     np.where(data['apache_3j_diagnosis_type'] < 800,
                                                                              'Trauma',
                                                                              np.where(
                                                                                  data[
                                                                                      'apache_3j_diagnosis_type'] < 900,
                                                                                  'Haematological',
                                                                                  np.where(data[
                                                                                               'apache_3j_diagnosis_type'] < 1000,
                                                                                           'Renal/Genitourinary',
                                                                                           np.where(data[
                                                                                                        'apache_3j_diagnosis_type'] < 1200,
                                                                                                    'Musculoskeletal/Skin disease',
                                                                                                    'Operative Sub-Diagnosis Codes'))))))))
                                 )
    return data


def stage4_feature_engineering(in_data, target):
    data = in_data.copy()
    cols2fix = [col for col in data.columns if (col.split('_')[-1] in ['rng', 'max', 'avg', 'min', 'apache'])]
    cols2fix += ['intubated_apache', 'ventilated_apache', 'gcs_unable_apache', 'arf_apache']
    cols2fix = list(set(cols2fix))
    data[cols2fix] = data[cols2fix].astype(np.float32)
    for fix_col, new_name in zip(['icu_id',
                                  'apache_3j_diagnosis',
                                  'apache_3j_diagnosis_grp',
                                  'apache_2_diagnosis',
                                  'profile',
                                  'apache_3j'],
                                 ['icuid',
                                  '3j', '3jgrp',
                                  'diag2', 'profile',
                                  '3j_grp']):
        mean_df = data.groupby([fix_col])[cols2fix].mean()
        mean_df.columns = [col + '_' + new_name + '_mean' for col in mean_df.columns]
        data = data.merge(mean_df, left_on=fix_col, right_index=True, how='left').drop(columns=[fix_col])

    data['a_b_ratio'] = data['age'] / data['bmi']

    for col_avg in data.columns:
        if not col_avg.endswith('_avg'):
            continue
        data[col_avg + '_aratio'] = data[col_avg] / data['age']
        data[col_avg + '_bratio'] = data[col_avg] / data['bmi']

    data['agi'] = data['weight'] / data['age']
    data = data.select_dtypes(np.float64).astype(np.float32).join(data.select_dtypes(exclude=np.float64))

    data = categorical_encode(data, target, 'TargetEncoder',
                              columns=data.select_dtypes(['object', 'category']).columns.tolist())
    return data


def stage2_feature_engineering(in_data):
    data = in_data.copy()
    data = impute_column(data, 'age')
    data = impute_column(data, 'weight')
    data = impute_column(data, 'height')
    data = data.assign(bmi=lambda dx: dx.weight / dx.height.div(100).pow(2))
    return data


def feature_selection(in_data, selected_columns_path):
    data = in_data.copy()
    selected_columns = joblib.load(selected_columns_path)
    data = data.loc[:, selected_columns]
    empty_columns = list(set(data.isnull().mean().loc[lambda s: s.eq(1)].index.tolist()))
    assert len(empty_columns) == 0
    return data


def clustering_imputation(in_data, target, imputers_path):
    data = in_data.copy()
    y = data[target]
    is_train = data['is_train']

    cluster_column = 'cluster_id'
    dt = data[['bmi', 'age', 'is_train', target]]
    data_block = WidsDataBlock(dt)
    ds = create_data_manager(data_block=data_block,
                             cv_column='cv_fold',
                             train_split_column=f'is_train',
                             label_columns=[target]
                             )

    ds.set_new_cv(StratifiedKFold(n_splits=5, shuffle=True, random_state=42))

    params = dict(pipeline={'sklearn.preprocessing.StandardScaler': {},
                            'sklearn.cluster.KMeans': {'n_clusters': 64}
                            }
                  )

    print(params)
    trainer = CVVoteTrainer(fold_trainer_cls=SklearnClassifierFoldTrainer,
                            ds=ds,
                            model_name='clustering',
                            params=params,
                            save_path=pathlib.Path('../.'),
                            task='predict'
                            )
    trainer.fit()

    data = data.join(trainer.extract_features().data.rename(columns={'clustering__diabetes_mellitus': cluster_column}))
    data = data.replace(np.Inf, np.nan)

    if imputers_path.exists():
        print('loading imputers from a file')
        imputers = joblib.load(imputers_path)
    else:
        imputers = {cluster_id: data.loc[lambda dx: dx.cluster_id.eq(cluster_id)].mean().to_dict() for cluster_id in
                    tqdm(data.cluster_id.unique())}
        joblib.dump(imputers, imputers_path)
    all_parts = []
    for cluster_id in tqdm(data.cluster_id.unique()):
        dt = data.loc[lambda dx: dx.cluster_id.eq(cluster_id)]
        all_parts.append(dt.fillna(imputers[cluster_id]))
    data = pd.concat(all_parts).drop(columns=[cluster_column])
    data = pd.DataFrame(prep.scale(data), index=data.index, columns=data.columns)
    data[target] = y
    data['is_train'] = is_train
    return data
