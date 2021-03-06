import argparse
import os
import joblib
import subprocess
import pandas as pd
import hypertune

from sklearn.model_selection import train_test_split, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, \
    f1_score

STORAGE_BUCKET = 'bank-marketing-model'
DATA_PATH = 'bank-additional-full.csv'
LOCAL_PATH = '/tmp'
PROJECT_ID = 'medium-articles'

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Job related arguments
    parser.add_argument('--job-dir', type=str, required=True,
                        help='Google Storage path where to store tuning '
                             'artifacts (string, required)')
    parser.add_argument('--n-jobs', type=int, default=1,
                        help='Number of parallel jobs to run (int, default 1)')

    # Train related arguments
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Percentage of data to use as test set (float, '
                             'default 0.2)')

    # Hyperparameters to tune
    parser.add_argument('--max-depth', type=int, default=10,
                        help='Maximum depth of each tree in Random Forest model'
                             ' (integer, default 10)')
    parser.add_argument('--min-samples-split', type=float, default=0.05,
                        help='Minimum number of samples (as fraction of total) '
                             'to split a node of a tree (float, default 0.05)')
    parser.add_argument('--max-features', type=float, default=None,
                        help='Number of features to use (as fraction of total) '
                             'for each tree in Random Forest model (float, '
                             'default square root of number of columns)')
    parser.add_argument('--max-samples', type=float, default=0.5,
                        help='Number of samples to use (as fraction of total) '
                             'for each tree in Random Forest model (float, '
                             'default 0.5)')

    # Parse arguments
    args = parser.parse_args()

    # Download dataset
    subprocess.call([
        'gsutil', 'cp',
        # Storage path
        os.path.join('gs://', STORAGE_BUCKET, DATA_PATH),
        # Local path
        os.path.join(LOCAL_PATH, 'dataset.csv')
    ])

    # Read data with pandas - separator is ';'
    df = pd.read_csv(os.path.join(LOCAL_PATH, 'dataset.csv'), sep=';')

    # Split data between train and test
    # For cross validation, we need only train set
    train, _ = train_test_split(df, test_size=args.test_size)

    y_train = (train['y'] == 'yes').astype(int)
    train = train.drop('y', 1)

    # Create a scikit-learn pipeline with preprocessing steps + model
    # First, define numeric and categorical features to use
    num_features = [0, 10, 11, 12, 13, 15, 16, 17, 18, 19]
    cat_features = [1, 2, 3, 4, 5, 6, 7, 8, 9, 14]

    pipeline = Pipeline([
        # The ColumnTransformer divide the preprocessing process between
        # categorical and numerical data
        ('data_prep',
         ColumnTransformer([
             ('num_prep', StandardScaler(), num_features),
             ('cat_prep', OneHotEncoder(handle_unknown='ignore'), cat_features)
         ])),
        # ML model
        ('model',
         RandomForestClassifier(
             random_state=1123,
             n_jobs=args.n_jobs,
             n_estimators=500,
             max_depth=args.max_depth,
             max_features=args.max_features,
             min_samples_split=args.min_samples_split,
             class_weight='balanced',
             max_samples=args.max_samples
         ))
    ])

    # We use a 5-fold cross validation
    scores = cross_validate(pipeline, train, y_train,
                            scoring=['accuracy', 'precision', 'recall', 'f1'],
                            cv=5,
                            return_train_score=True)

    # Save metrics
    results = pd.DataFrame(scores)
    results.loc['avg'] = results.mean()
    results.to_csv(os.path.join(LOCAL_PATH, 'results.csv'))

    # Upload results Dataframe to Storage
    subprocess.call([
        'gsutil', 'cp',
        # Local path of results
        os.path.join(LOCAL_PATH, 'results.csv'),
        os.path.join(args.job_dir, 'results.csv')
    ])

    # Here we pass the metric to the hypertune framework

    # Instantiate a hypertune object
    hpt = hypertune.HyperTune()
    # Compute the average metric
    avg_f1 = scores['test_f1'].mean()
    # Pass the value to hyperopt
    hpt.report_hyperparameter_tuning_metric(
        hyperparameter_metric_tag='F1',
        metric_value=avg_f1,
        global_step=1
    )
