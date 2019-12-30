# Criminal Investment in Residential Property


## Directories:

- **data** : h5 files that contain (nearly) raw data from Bexar County and Texas Secretary of State; will hold preprocessed h5 file as well
- **build** : Scripts that merge and pre-process raw data from Bexar County and Texas Secretary of State

## Setup

1. Clone repo
2. Create new conda environment

```
conda create --name <new_env> python=3.7
conda activate <new_env>
```
3. `cd` into the repo directory and run:

```
pip install -r requirements.txt
```
4. Merge the two raw datasets into a preprocessed file:

```
python build/bexar_data_download_merge.py
```

## Run Inference

## Build Model

## Analysis
