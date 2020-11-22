import argparse
import os
import joblib
import subprocess
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, \
    f1_score
import pandas as pd


STORAGE_BUCKET = 'bank-marketing-model'
DATA_PATH = 'bank-additional-full.csv'
LOCAL_PATH = '/tmp'
PROJECT_ID = 'medium-articles'


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--storage-path', type=str, required=True,
                        help='Google Storage path where to store training '
                             'artifacts (string, required)')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Percentage of data to use as test set (float, '
                             'default 0.2)')
    parser.add_argument('--n-estimators', type=int, default=100,
                        help='Number of trees in Random Forest model '
                             '(integer, default 100)')
    parser.add_argument('--max-depth', type=int, default=10,
                        help='Maximum depth of each tree in Random Forest model'
                             ' (integer, default 10)')
    parser.add_argument('--min-samples_split', type=float, default=0.05,
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
    parser.add_argument('--n-jobs', type=int, default=1,
                        help='Number of parallel jobs to run (int, default 1)')

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
    train, test = train_test_split(df, test_size=args.test_size,
                                   random_state=42)

    y_train = (train['y'] == 'yes').astype(int)
    train = train.drop('y', 1)
    y_test = (test['y'] == 'yes').astype(int)
    test = test.drop('y', 1)

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
             random_state=42,
             n_jobs=args.n_jobs,
             n_estimators=args.n_estimators,
             max_depth=args.max_depth,
             max_features=args.max_features if args.max_features is not
                                                  None else 'sqrt',
             min_samples_split=args.min_samples_split,
             class_weight='balanced',
             max_samples=args.max_samples
         ))
    ])

    # Train the model
    pipeline.fit(train, y_train)

    # Save model
    joblib.dump(pipeline, os.path.join(LOCAL_PATH, 'model.joblib'))

    # Get predictions
    pred_train = pipeline.predict(train)
    pred_test = pipeline.predict(test)

    # Calculate a bunch of performance metrics
    results = pd.DataFrame(
        {'accuracy': [accuracy_score(y_train, pred_train),
                      accuracy_score(y_test, pred_test)],
         'precision': [precision_score(y_train, pred_train),
                       precision_score(y_test, pred_test)],
         'recall': [recall_score(y_train, pred_train),
                    recall_score(y_test, pred_test)],
         'f1': [f1_score(y_train, pred_train),
                f1_score(y_test, pred_test)]},
        index=['train', 'test']
    )
    results.to_csv(os.path.join(LOCAL_PATH, 'results.csv'))

    # Upload model and results Dataframe to Storage
    subprocess.call([
        'gsutil', 'cp',
        # Local path of the model
        os.path.join(LOCAL_PATH, 'model.joblib'),
        os.path.join(args.storage_path, 'model.joblib')
    ])
    subprocess.call([
        'gsutil', 'cp',
        # Local path of results
        os.path.join(LOCAL_PATH, 'results.csv'),
        os.path.join(args.storage_path, 'results.csv')
    ])
