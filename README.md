# Multimodal-Paper-Mill-Detection

This repository includes the dataset and code for "Improving machine learning-based paper mill detection in cancer research using full text, image and citation data."

**dataset_dois.csv** includes the DOIs of all papers extracted for this project along with other metadata

**model_splits.csv** assigns each paper to internal or external validation and records what types of models they were used to train

**build_dataset.py** uses the DOIs and public APIs to gather abstract, full text, citation and image data for each paper in the dataset

**train_models.py** trains all five models

**evaluate_models.py** reports evaluation metrics for each of the five models.

Python 3.9+ and 1-2 GB of storage are needed to run this pipeline. A GPU isn't mandatory but model training may take multiple hours without one. 

Run the scripts in the following order:
1. build_dataset.py
2. train_models.py
3. evaluate_models.py
