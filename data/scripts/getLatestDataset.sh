#!/bin/bash

# Set your Google Cloud authentication credentials
export GOOGLE_APPLICATION_CREDENTIALS='data/scripts/mlops37_service_account.json'

# Authenticate with Google Cloud
gcloud auth activate-service-account --key-file="$GOOGLE_APPLICATION_CREDENTIALS"

# Run the DVC pull command
dvc pull
