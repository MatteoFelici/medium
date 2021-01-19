MODEL_NAME=bank_marketing
VERSION_NAME=v1_0
REGION=europe-west1
INSTANCES_FILE=./query_instances.json


gcloud ai-platform predict \
    --model "$MODEL_NAME" \
    --version "$VERSION_NAME" \
    --region "$REGION" \
    --json-instances "$INSTANCES_FILE"