# mlops_sample_project 🚀

This is a sample project for machine learning operations. Template is created by Nicki Skafte Detlefsen.

## Project structure 📁

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
|   ├── test_training.py
│   └── test_model.py
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── pyproject.toml            # Python project file
├── README.md                 # Project README
├── requirements.txt          # Project requirements
├── requirements_dev.txt      # Development requirements
└── tasks.py                  # Project tasks
```

Created using [mlops_template](https://github.com/SkafteNicki/mlops_template), a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting started with Machine Learning Operations (MLOps).

## Docker 🐳

### How to Build Train Docker Image 🛠️
```bash
docker build -f dockerfiles/train.dockerfile . -t train:latest
```

### How to Run Train Docker Image ▶️
```bash
docker run --name experiment1 train:latest
```

### How to Fetch Models File from Docker Container 📂
```bash
docker cp experiment1:models/model.pth models/model_experiment1.pth
```

### How to Evaluate with Docker 📊
```bash
docker build -f dockerfiles/evaluate.dockerfile . -t evaluate:latest
docker run --name evaluate evaluate:latest
```

## How to Run Code with Hydra 🐍

Ensure that `exp2.yaml` is defined.

```bash
python src/mlops_sample_project/train.py train_experiments=exp2
```

Output is saved to the outputs folder.

## Installation 💻

To install the project in editable mode, run the following command:

```bash
pip install -e .
```

To install the project in production mode, run the following command:

```bash
pip install .
```

in the root of the project.
