#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import yaml
import os
import warnings
warnings.simplefilter('ignore')

def config_loader():
    """Set config path and load variables."""
    config_path = os.path.join('./configs/config.yaml')
    with open(config_path,'r') as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.BaseLoader)
    return cfg

def load_df(filepath):
    """Read in processed data."""
    print("Reading in processed data.")
    df = pd.read_hdf(filepath)
    return df

def get_true_labels():
    """Return dataframe with only properties with true labels."""
    cfg = config_loader()
    filepath = cfg['processed']
    df = load_df(filepath=filepath)


    # Fill na values for price_psf with 0 (means it's an empty lot)
    df['price_psf'] = df['price_psf'].fillna(0)

    # Filter out wonky situations (erroneous property encoding, sq footage of 1, etc.)
    percentile = float(cfg['pctile_filter'])
    trim_prop_df = df[(df.price_psf<df.price_psf.quantile(percentile))]

    # Get only properties owned by the top n most frequent owners
    largest_owners = int(cfg['largest_owners'])
    top_owners_list = trim_prop_df.py_owner_name.value_counts()[:largest_owners].index
    true_labels_df = trim_prop_df[(trim_prop_df.py_owner_name.str.contains('|'.join(top_owners_list))==True) |
                (trim_prop_df.crim_prop==1)
    ]

    # Grab all columns that are under the 70,000 nan limit
    nan_limit = int(cfg['nan_limit'])
    check_nan = trim_prop_df.isnull().sum()
    variables_list = check_nan[check_nan<nan_limit].index
    variables_list = variables_list[variables_list.isin(trim_prop_df.columns[trim_prop_df.dtypes!='object'])]
    variables_list = variables_list.drop([
        'py_owner_id','py_addr_zip_cass','prop_val_yr','appraised_val',
        'Prior_Mkt_Val','bexar_2015_market_value','bexar_2016_market_value',
        'bexar_2017_market_value','bexar_2018_market_value','owner_zip_code',
        'property_zip','neighborhood_code'
    ])

    sub_df = true_labels_df[variables_list]
    df_true_labels = sub_df.dropna()
    print("Number of properties after reducing:", len(df_true_labels))
    return df_true_labels

def export_to_h5():
    """Export new df to h5."""
    df_true_labels = get_true_labels()
    print("Exporting dataframe to h5 file.")
    df_true_labels.to_hdf(
        './data/processed/bexar_true_labels.h5',
        key='bexar_true_labels.h5',
        mode='w',
        format='t'
    )
    print("Dataframe has been saved as bexar_true_labels.h5")

if __name__ == '__main__':
    export_to_h5()
