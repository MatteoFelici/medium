MODEL_NAME=bank_marketing
REGION=europe-west1

gcloud ai-platform models create "$MODEL_NAME" \
    --region "$REGION"