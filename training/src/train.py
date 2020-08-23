import argparse
import os
import subprocess
from sklearn.model_selection import train_test_split
import pandas as pd

STORAGE_BUCKET = 'telco-churn-model' 
DATA_PATH = 'WA_Fn-UseC_-Telco-Customer-Churn.csv'
LOCAL_PATH = '/tmp/dataset.csv'
PROJECT_ID = 'medium-articles'

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--test-size', required=True, type=float)
    
    # Parse arguments
    args = parser.parse_args()

    # Download dataset
    # client = storage.Client(project=PROJECT_ID)
    # bucket = client.get_bucket(STORAGE_BUCKET)
    # blob = bucket.get_blob(DATA_PATH)
    # blob.download_to_file(LOCAL_PATH)
    subprocess.call([
        'gsutil', 'cp',
        os.path.join('gs://', STORAGE_BUCKET, DATA_PATH),
        LOCAL_PATH
    ])
    
    # Read data with pandas
    df = pd.read_csv(LOCAL_PATH)
    
    # Split data between train and test
    train, test = train_test_split(df, test_size=args.test_size)

    print(train.shape)
    print(test.shape)
