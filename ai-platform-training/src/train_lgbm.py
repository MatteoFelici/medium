import numpy as np
import pandas as pd
import lightgbm

parameters = {
    'application': 'binary',
    'objective': 'binary',
    'metric': 'auc',
    'is_unbalance': 'true',
    'boosting': 'gbdt',
    'num_leaves': 31,
    'feature_fraction': 0.5,
    'bagging_fraction': 0.5,
    'bagging_freq': 20,
    'learning_rate': 0.05,
    'verbose': 0
}

run = wandb.init(project="aiplatform-lgbm-project", config=parameters)

train = pd.read_csv('gs://test-zganatra/kraggle_lgbm_data/train.csv')

y = train.target.values
train.drop(['id', 'target'], inplace=True, axis=1)
x = train.values

x, x_test, y, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

categorical_features = [c for c, col in enumerate(train.columns) if 'cat' in col]
train_data = lightgbm.Dataset(x, label=y, categorical_feature=categorical_features)
test_data = lightgbm.Dataset(x_test, label=y_test)

categorical_features = [c for c, col in enumerate(train.columns) if 'cat' in col]
train_data = lightgbm.Dataset(x, label=y, categorical_feature=categorical_features)
test_data = lightgbm.Dataset(x_test, label=y_test)

model = lightgbm.train(parameters,
                       train_data,
                       valid_sets=test_data,
                       num_boost_round=5000,
                       early_stopping_rounds=100,
                       callbacks=[wandb_callback()])

submission = pd.read_csv('gs://test-zganatra/kraggle_lgbm_data/test.csv')
ids = submission['id'].values

submission.drop('id', inplace=True, axis=1)
x = submission.values
y = model.predict(x)
