JOB_NAME=bank_marketing_tune_$(date +%Y%m%d_%H%M%S)
REGION=europe-west6

gcloud ai-platform jobs submit training "$JOB_NAME" \
    --module-name=src.tune \
    --package-path=./src \
    --job-dir=gs://bank-marketing-model/"$JOB_NAME" \
    --region=$REGION \
    --scale-tier=CUSTOM \
    --master-machine-type=n1-standard-8 \
    --python-version=3.7 \
    --runtime-version=2.2 \
    --config=./config.yaml \
    -- \
    --n-jobs=8