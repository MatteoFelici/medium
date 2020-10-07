JOB_NAME=bank_marketing_tune_$(date +%Y%m%d_%H%M%S)
REGION=europe-west6

gcloud ai-platform jobs submit training "$JOB_NAME" \
    --module-name=src.tune \
    --package-path=./src \
    --job-dir=gs://bank-marketing-model/"$JOB_NAME" \
    --region=$REGION \
    --python-version=3.7 \
    --runtime-version=2.1 \
    --config=./config.yaml \
    -- \
    --n-jobs=8