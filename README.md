# Criminal Investment in Residential Property


## Directories:

- **data** : h5 files that contain (nearly) raw data from Bexar County and Texas Secretary of State; will hold preprocessed h5 file as well
- **build** : Scripts that merge and pre-process raw data from Bexar County and Texas Secretary of State

## Setup

1. Prior to cloning the repo, ensure Git Large File Storage is installed (`brew install git-lfs`)
2. Clone repo
3. Create new conda environment

```
conda create --name <new_env> python=3.7
conda activate <new_env>
conda install nb_conda
```
4. With `pwd` being the repo directory, run:

```
pip install -r requirements.txt
```
5. Merge the two raw datasets into a preprocessed file:

```
python build/merge.py
```

6. Build the features used for modeling:
```
python build/process.py
```

## Run Inference

## Build Model

## Analysis
