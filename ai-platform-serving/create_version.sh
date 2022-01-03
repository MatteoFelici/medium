MODEL_NAME=bank_marketing
VERSION_NAME=v1_0
MODEL_DIR=gs://bank-marketing-model/serving_model
REGION=europe-west1


gcloud ai-platform versions create "$VERSION_NAME" \
  --model "$MODEL_NAME" \
  --origin "$MODEL_DIR" \
  --region "$REGION" \
  --framework scikit-learn \
  --python-version 3.7 \
  --runtime-version 2.2 \
  --machine-type n1-standard-4