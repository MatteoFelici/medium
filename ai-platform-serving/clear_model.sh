MODEL_NAME=bank_marketing
VERSION_NAME=v1_0
MODEL_DIR=gs://bank-marketing-model/serving_model
REGION=europe-west1


gcloud ai-platform versions delete "$VERSION_NAME" \
  --model "$MODEL_NAME" \
  --region "$REGION"

gcloud ai-platform models delete "$MODEL_NAME" \
  --region "$REGION"