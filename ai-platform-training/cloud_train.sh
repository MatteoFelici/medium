JOB_NAME=bank_marketing_$(date +%Y%m%d_%H%M%S)
REGION=europe-west6

gcloud ai-platform jobs submit training $JOB_NAME \
    --module-name=src.train \
    --package-path=./src \
    --staging-bucket=gs://bank-marketing-model \
    --region=$REGION \
    --scale-tier=CUSTOM \
    --master-machine-type=n1-standard-8 \
    --python-version=3.7 \
    --runtime-version=2.2 \
    -- \
    --storage-path=gs://bank-marketing-model/$JOB_NAME \
    --n-estimators=500 \
    --n-jobs=8