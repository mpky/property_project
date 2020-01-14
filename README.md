# Detecting Criminal Investment in Residential Property

## Contents:

- **data**:
  - raw_h5_files/bexar_property_all.h5 - 2019 Bexar County Appraisal District data with appraisal values from 2015, 2016, 2017, and 2018 added.
  - raw_h5_files/texas_corp_merged.h5 - State of Texas Comptroller of Public Accounts company data merged with the Comptroller's dataset that includes company directors and officers.
  - labels/criminal_properties_labels.csv - Dataset of properties located in Bexar County that have been used to launder the proceeds of some form of crime. One such example is 1115 Links Cv, San Antonio, TX 78260, which is owned by Red Kaizen Investments LLC. Red Kaizen Investments LLC is one of dozens of companies named as defendants in the court case against Rafael Olvera Amezcua, a Mexican financier accused of running a sham savings and loans business that defrauded depositors of more than $160 million.

- **build**:
  - build/merge.py - Script that joins the relevant columns from the Comptroller dataset with the cleaned Bexar property data. The output is a preprocessed h5 file.
  - build/process.py - Script that generates features for modeling and joins the labeled properties data. The output is a processed h5 file.

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
