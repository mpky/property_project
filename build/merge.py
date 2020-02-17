"""
Run this with PWD being the top level of the repo and python build/merge.py
"""

import pandas as pd
import numpy as np
import warnings
warnings.simplefilter('ignore')
import os

def merge_data():
    """
    Merge bexar_property_all.h5 with texas_corp_merged.h5 and return
    bexar_preprocessed.h5
    """
    # Load property dataframe
    bexar_property = pd.read_hdf('./data/raw_h5_files/bexar_property_all.h5')
    # Load comptroller data that has officers info
    tx_corporate_df = pd.read_hdf('./data/raw_h5_files/texas_corp_merged.h5')


    # Strip commas and periods
    # Clean the name fields for both sets (remove commas, periods)
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

    # To do: see if this way of dropping duplicates is adversley affecting the data
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

    corp_prop_merged.to_hdf('./data/preprocessed/bexar_preprocessed.h5',
                            key='bexar_preprocessed.h5',
                            mode='w'
                           )
    print('Dataframe has been exported as bexar_preprocessed.h5 to "preprocessed" data folder.')

if __name__ == '__main__':
    merge_data()
