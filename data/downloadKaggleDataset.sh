#!/bin/bash

curl -L -o "$(dirname "$0")/raw/daigt-proper-train-dataset.zip"\
    https://www.kaggle.com/api/v1/datasets/download/thedrcat/daigt-proper-train-dataset

unzip "$(dirname "$0")/raw/daigt-proper-train-dataset.zip" -d "$(dirname "$0")/raw/"
rm "$(dirname "$0")/raw/daigt-proper-train-dataset.zip"