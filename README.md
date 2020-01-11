# Criminal Investment in Residential Property


## Directories:

- **data** : CSV of labeled properties, raw h5 files, preprocessed data, and processed data
- **build** : Scripts that merge and pre-process raw data from Bexar County and Texas Comptroller

## Setup

1. Clone repository.
2. Go to this [Google Drive link](https://drive.google.com/drive/folders/16hbhfiExi2Nf6zO56Dzl_28kw2cKKsB0?usp=sharing) to download the raw datasets for both Bexar Property and the Texas Comptroller.
3. Select "Download All" in the upper righthand corner. This will download a zip file.
4. Unzip the zip file and move the resulting "raw_h5_files" folder to this repo under the data folder. The two files will, together, be ~750 MB in size.



Create new conda environment.

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
