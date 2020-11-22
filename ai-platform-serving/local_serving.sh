gcloud ai-platform local predict \
    --model-dir gs://bank-marketing-model/bank_marketing_20201122_124009 \
    --json-instances ./query_instances.json \
    --framework scikit-learn