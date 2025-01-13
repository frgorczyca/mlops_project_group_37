# text_detect

Text_detect is an exam project for the DTU course 02476 MLOps.
The goal is to develop a machine learning pipeline and use a machine learning model to detect whether a text is generated by an AI or written by real people.
The data we have used is the kaggle dataset "DAIGT Proper Train Dataset" (https://www.kaggle.com/datasets/thedrcat/daigt-proper-train-dataset/data?select=train_drcat_04.csv) and the base-model used for the classification is the LLM transformer RoBERTa with pretrained weights (https://huggingface.co/FacebookAI/roberta-large-mnli?text=The+dog+was+lost.+Nobody+lost+any+animal). 

# How to use

To-be-updated

## Project structure

The directory structure of the project looks like this:
```txt
├── .github/                  # Github actions and dependabot
│   ├── dependabot.yaml
│   └── workflows/
│       └── tests.yaml
├── configs/                  # Configuration files
├── data/                     # Data directory
│   ├── processed
│   └── raw
├── dockerfiles/              # Dockerfiles
│   ├── api.Dockerfile
│   └── train.Dockerfile
├── docs/                     # Documentation
│   ├── mkdocs.yml
│   └── source/
│       └── index.md
├── models/                   # Trained models
├── notebooks/                # Jupyter notebooks
├── reports/                  # Reports
│   └── figures/
├── src/                      # Source code
│   ├── project_name/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── data.py
│   │   ├── evaluate.py
│   │   ├── models.py
│   │   ├── train.py
│   │   └── visualize.py
└── tests/                    # Tests
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_data.py
│   └── test_model.py
├── .gitignore
├── .pre-commit-config.yaml
├── downloadKaggleDataset.sh  # Bash script to download datasets
├── LICENSE
├── pyproject.toml            # Python project file
├── README.md                 # Project README
├── requirements.txt          # Project requirements
├── requirements_dev.txt      # Development requirements
└── tasks.py                  # Project tasks
```

## Contributers

Artur Adam Habuda s233190
Eline Siegumfeldt s183540
Franciszek Marek Gorczyca s233664
Max-Peter Schrøder s214238
