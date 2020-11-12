gcloud ai-platform local predict --model-dir gs://bank-marketing-model/bank_marketing_20200827_121556 \
    --json-instances ./query_instances.json \
    --framework scikit-learn