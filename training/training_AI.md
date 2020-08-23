# Google AI Platform adventures - Training
Welcome to the first articles in this series about doing Machine Learning stuff on the Google Cloud Platform!

In particular, we will take a look at the [AI Platform](https://cloud.google.com/ai-platform/docs "docs"). It is a subset of tools strictly related with Machine Learning, among which:
- AI Platform Training, for training/tuning models on the cloud
- AI Platform Prediction, to host trained models on the cloud
- AI Pipelines, to create a step-by-step process using Kubernetes and Docker Images

and many others.

*DISCLAIMER: I am not affiliated with Google in any way, I simply decided to write these articles to share the knowledge I acquired using these tools in my daily job.*

For this first article, I'll focus on _AI Platform Training_, a product to run training jobs on the Cloud with custom code and customizable machines.

In this tutorial we will see how to train an end-to-end Machine Learning model by:
- writing the actual Python application with the definition of the training
- run a local test
- run the training job on the Cloud Platform

For the tutorial, you will need:
- an active Google Cloud Platform account (you can setup a new account visiting the [homepage](https://cloud.google.com/)) and a GCP project 
- Python3 and `gcloud` (Google Cloud SDK, [here](https://cloud.google.com/sdk/docs) the instructions) installed on your workstation
- the [dataset used for tutorial](https://www.kaggle.com/blastchar/telco-customer-churn "Kaggle's Telco Customer Churn")

## Step 1: store the data on Google Storage
After you downloaded the dataset on your local machine, go to the Google Storage of your GCP project. Create a new bucket (I'll call it `telco-churn-model`) and load the dataset in it. The situation should look like this:

![Bucket](./images/bucket.png)

If you want the more geeky way, you can use the `gsutil` application from the command line ([how to install gsutil](https://cloud.google.com/storage/docs/gsutil_install)). Then, from the command line, go into the local directory containing the dataset and

```
gsutil mb gs://telco-churn-model
gsutil cp WA_Fn-UseC_-Telco-Customer-Churn.csv gs://telco-churn-model
```

## Step 2: write the Python training application
If you're reading this, chances are that you already know how to write an end-to-end Python program to train a Machine Learning model. Anyway, since we're planning to train the model on the Cloud, there are a few steps that are someway different than the usual __Kaggle-ish__ code.

First thing to bare in mind, **you can use only scikit-learn, XGBoost or Tensorflow** to train your model in the "classic" Training job (there is a way to use a custom Python environment, but we'll see it in another article).

So, the basics of the Python application are:
- download the dataset from Storage
- do some data preparation (split train-test, missing imputation, ...)
- train the model
- store the trained model on Storage
