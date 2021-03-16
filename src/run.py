import joblib
import pandas as pd
import pandas as pd
import numpy as np
import sys
from functions import stage1_feature_engineering, impute_column, WidsDataBlock, stage2_feature_engineering, \
    feature_selection
import sklearn.pipeline
import pathlib
from tqdm.auto import tqdm
import namegenerator
from sklearn import metrics
import joblib

sys.path.insert(0, r"C:\Users\Pavel\Documents\projects\dzl")
from dzl.src.base import create_data_manager, CVTrainer, CVVoteTrainer
from dzl import SklearnClassifierFoldTrainer, TabNetFoldTrainer, XGBoostClassifierFoldTrainer, \
    LightGBMClassifierFoldTrainer, CatboostClassifierFoldTrainer, LGBMOptunaOptimizer, CatBoostOptunaOptimizer, \
    XGBoostOptunaOptimizer, LogRegOptunaOptimizer, HistGradientBoostingOptunaOptimizer, TabNetOptunaOptimizer

from src.functions import stage1_feature_engineering, impute_column, stage3_feature_engineering, categorical_encode, \
    stage4_feature_engineering, clustering_imputation, WidsDataBlock

train_path = pathlib.Path('../../input/widsdatathon2021/TrainingWiDS2021.csv')
test_path = pathlib.Path('../../input/widsdatathon2021/UnlabeledWiDS2021.csv')

root_path = pathlib.Path(r'C:\Users\Pavel\Documents\projects\kaggle-widsdatathon2021')

outputs_path = root_path / 'outputs'
config_path = root_path / 'config'

imputers_path = outputs_path / 'imputers.model'
features_path = outputs_path / 'data.pkl'
best_models_path = config_path / 'all_models.params'
selected_columns_path = config_path / 'selected_columns.lst'
# Reading the Data
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

model_params = joblib.load(best_models_path)

target = 'diabetes_mellitus'

if not features_path.exists():
    cols2drop = ['Unnamed: 0', 'hospital_id', 'readmission_status', 'pre_icu_los_days']

    train['is_train'] = 1
    test['is_train'] = 0
    data = pd.concat([train, test]).drop(cols2drop, axis=1).set_index('encounter_id')

    data = stage1_feature_engineering(in_data=data)
    data = stage2_feature_engineering(in_data=data)
    data = stage3_feature_engineering(in_data=data)
    data = stage4_feature_engineering(in_data=data, target=target)

    data = feature_selection(in_data=data, selected_columns_path=selected_columns_path)
    data = clustering_imputation(in_data=data, target=target, imputers_path=imputers_path)
    data.to_pickle(features_path)

else:
    print('Loading precomputed features')
    data = pd.read_pickle(features_path)

data_block = WidsDataBlock(data)

ds = create_data_manager(data_block=data_block,
                         cv_column='cv_fold',
                         train_split_column=f'is_train',
                         label_columns=[target]
                         )

for single_model_params in model_params:
    try:
        params = single_model_params['params']
        model_cls = single_model_params['model_fold_class']

        model_name = single_model_params['model_name']

        trainer = CVTrainer(fold_trainer_cls=model_cls,
                            ds=ds,
                            model_name=model_name,
                            params=params,
                            save_path=outputs_path,
                            metric='roc_auc_score'
                            )

        trainer.fit()
        print(trainer.score('trn'), trainer.score('val'))
        trainer.save()
    except Exception as e:
        print(e)
        pass
