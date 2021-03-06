#!/usr/bin/env python
# coding: utf-8

"""
Run this with PWD being the top level of the repo and python build/merge.py.

This script will join the relevant columns from the TX Comptroller data file
(texas_corp_merged.h5) with the cleaned Bexar property data
(bexar_property_all.h5).
"""

import pandas as pd
import numpy as np
import warnings
warnings.simplefilter('ignore')
import os
import yaml

def config_loader():
    """Set config path and load variables."""
    config_path = os.path.join('./configs/config.yaml')
    with open(config_path,'r') as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.BaseLoader)
    return cfg

def merge_data():
    """
    Merge bexar_property_all.h5 with texas_corp_merged.h5 and return
    bexar_preprocessed.h5
    """

    cfg = config_loader()
    property_filepath = cfg['property_all']
    tx_corp_filepath = cfg['texas_corp_merged']

    # Load property dataframe
    print("Loading property data from",property_filepath)
    bexar_property = pd.read_hdf(property_filepath)
    # Load comptroller data that has officers info
    print("Loading corporate data from",tx_corp_filepath)
    tx_corporate_df = pd.read_hdf(tx_corp_filepath)


    # Strip commas and periods
    # Clean the name fields for both sets (remove commas, periods)
    print("Cleaning fields.")
    tx_corporate_df['cleaned_name'] = tx_corporate_df['Taxpayer Name'].str.replace('.','',regex=False)
    tx_corporate_df['cleaned_name'] = tx_corporate_df['cleaned_name'].str.replace(',','',regex=False)

    # Strip commas and periods from property data
    bexar_property['cleaned_name'] = bexar_property.py_owner_name.str.replace('.','',regex=False)
    bexar_property['cleaned_name'] = bexar_property['cleaned_name'].str.replace(',','',regex=False)

    # Strip trailing spaces and capitalize
    bexar_property.cleaned_name = bexar_property.cleaned_name.str.upper().str.strip()
    tx_corporate_df.cleaned_name = tx_corporate_df.cleaned_name.str.upper().str.strip()

    # Merge based on name, but taxpayer name is not always unique
    # To get unique names in tx_corporate will use .drop_duplicates and keep the first entry

    # To do: see if this way of dropping duplicates is adversely affecting the data
    print("Merging in corporate data and dropping duplicate entries.")
    tx_corporate_df.drop_duplicates(subset='cleaned_name',keep='first',inplace=True)
    corp_prop_merged = pd.merge(bexar_property, tx_corporate_df[[
        'Taxpayer Number',
        'Taxpayer Name',
        'Taxpayer Address',
        'Taxpayer City',
        'Taxpayer State',
        'Taxpayer Organizational Type',
        'Taxpayer Zip Code',
        'SOS Charter Date',
        'SOS Status Date',
        'SOS Status Code',
        'Right to Transact Business Code',
        'Officer/Director Latest Year Filed',
        'Officer/Director Name',
        'Officer/Director Title',
        'Officer/Director State',
        'cleaned_name'
        ]], how='left', on='cleaned_name')


    if not os.path.exists('./data/preprocessed'):
        os.makedirs('./data/preprocessed')
    print("Exporting dataframe to h5 file.")
    corp_prop_merged.to_hdf('./data/preprocessed/bexar_preprocessed.h5',
                            key='bexar_preprocessed.h5',
                            mode='w'
                           )
    print('Dataframe has been exported as bexar_preprocessed.h5 in "preprocessed" data folder.')

if __name__ == '__main__':
    merge_data()
