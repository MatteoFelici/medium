MODEL_DIR=gs://bank-marketing-model/serving_model

gcloud ai-platform local predict \
    --model-dir "$MODEL_DIR" \
    --json-instances ./query_instances.json \
    --framework scikit-learn