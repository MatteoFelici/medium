# Google AI Platform adventures - Training
Welcome to the first articles in this series about doing Machine Learning stuff on the Google Cloud AI Platform!

I've found this platform very useful, with a lot of potential and plenty of good tools to build ML models. But I've found that the very first steps in using some of these tools are quite challenging. For this reason, I'd like to share my experience with all the new users of this platform.

For this first entrance, I'll focus on _AI Platform Training_, a product to run training jobs on the Cloud with custom code and customizable machines.

In this tutorial we will see how to train an end-to-end Machine Learning model by:
- writing the actual Python application with the definition of the training
- run a local test
- run the training job on the Cloud Platform

For the tutorial, you will need:
- an active Google Cloud Platform account (you can setup a new account visiting the [homepage](https://cloud.google.com/))
- Python3 and `gcloud` (Google Cloud SDK, [here](https://cloud.google.com/sdk/docs) the instructions) installed on your workstation
- the [dataset used for tutorial](https://www.kaggle.com/blastchar/telco-customer-churn "Kaggle's Telco Customer Churn")

## Step 1: write the Python training application
If you're reading this, chances are that you already know how to write an end-to-end Python program to train a Machine Learning model. Anyway, since we're planning to train the model on the Cloud, there are a few steps that are someway different than the usual __Kaggle-ish__ code.