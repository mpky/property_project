"""
Run this with PWD being the top level of the repo and
python3 build/bexar_data_download_merge.py

This script will:
1. Join the relevant columns from the TX Comptroller data file
(texas_corp_merged.h5) with the cleaned Bexar property data (bexar_property_all.h5)
2. Create the three features the Dutch study found most significant in their analysis
"""

import pandas as pd
import numpy as np
import warnings
warnings.simplefilter('ignore')

def merge_data():
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

    return corp_prop_merged


def clean_add_features(corp_prop_merged):
    # Convert SOS Charter date to datetime type
    corp_prop_merged['SOS Charter Date'] = pd.to_datetime(corp_prop_merged['SOS Charter Date'],format='%Y%m%d')
    # Convert deed date
    corp_prop_merged['deed_dt'] = pd.to_datetime(corp_prop_merged['deed_dt'],format='%m%d%Y')

    # Add main three features from Dutch paper
    # Owner is a just-established company
    # Subtract the two dates and see if the difference is within 365 days
    corp_prop_merged['deed_charter_diff'] = corp_prop_merged['deed_dt'] - corp_prop_merged['SOS Charter Date']

    # If this is <= 365, consider "just-established"
    corp_prop_merged['just_established_owner'] = np.where(corp_prop_merged['deed_charter_diff'] <= '365 days',1,0)
    # 16,349 properties in the dataset are considered to be owned by 'just-established' companies

    # Foreign Owner
    corp_prop_merged['foreign_based_owner'] = np.where(corp_prop_merged['py_addr_country'] != 'US',1,0)

    # Unusual Price Fluctuations
    # Determining what qualifies as "unusual" to the EDA portion, but I will calculate YoY differences for each year of data I have
    # YoY difference from 2018 to 2019
    corp_prop_merged['yoy_diff_2019'] = corp_prop_merged.market_value - corp_prop_merged.bexar_2018_market_value
    # YoY difference from 2017 to 2018
    corp_prop_merged['yoy_diff_2018'] = corp_prop_merged.bexar_2018_market_value - corp_prop_merged.bexar_2017_market_value
    # YoY difference from 2016 to 2017
    corp_prop_merged['yoy_diff_2017'] = corp_prop_merged.bexar_2017_market_value - corp_prop_merged.bexar_2016_market_value
    # YoY difference from 2015 to 2016
    corp_prop_merged['yoy_diff_2016'] = corp_prop_merged.bexar_2016_market_value - corp_prop_merged.bexar_2015_market_value

    # Export as h5 file
    print('Exporting dataframe to h5 file.')
    corp_prop_merged.to_hdf('./data/processed/bexar_merged_df.h5',
                            key='bexar_merged_df.h5',
                            mode='w'
                           )
    print('Dataframe has been exported as bexar_merged_df.h5 to "processed" data folder.')

if __name__ == '__main__':
    clean_add_features(merge_data())
