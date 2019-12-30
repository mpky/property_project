"""
Run this with PWD being the top level of the repo and
python build/process.py

This script will create all relevant features necessary for modeling and output
into processed h5 file.
"""


import pandas as pd
import numpy as np
import warnings
warnings.simplefilter('ignore')
import os


def clean_add_features():
    # Read in preprocessed simplefilter
    corp_prop_merged = pd.read_hdf('./data/preprocessed/bexar_preprocessed.h5')

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

    # Mkdir
    if not os.path.exists('./data/processed'):
        os.makedirs('./data/processed')

    corp_prop_merged.to_hdf('./data/processed/bexar_processed.h5',
                            key='bexar_processed.h5',
                            mode='w'
                           )
    print('Dataframe has been exported as bexar_processed.h5 to "processed" data folder.')

if __name__ == '__main__':
    clean_add_features()
