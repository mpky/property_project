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
import yaml
import os

def price_change(dataframe,column_1,column_2,new_column):
    """Return YoY price change %."""
    dataframe[new_column] = (dataframe[column_1] - dataframe[column_2])/dataframe[column_2]
    dataframe.loc[np.isinf(dataframe[new_column]), new_column] = np.nan
    return dataframe

def config_loader():
    """Set config path and load variables."""
    config_path = os.path.join('./configs/config.yaml')
    with open(config_path,'r') as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.BaseLoader)
    return cfg

def dataframe_loader(filepath):
    """Return dataframe from bexar_preprocessed.h5."""
    print("Reading in preprocessed data from h5 file.")
    corp_prop_merged = pd.read_hdf(filepath)
    print('Dataframe loaded.')
    return corp_prop_merged

def engineer_features():
    """Create a number of features to be used in modeling."""
    cfg = config_loader()
    preprocessed_filepath = cfg['preprocessed']
    dataframe = dataframe_loader(preprocessed_filepath)


    print("Engineering features.")
    # Create features and convert datatypes
    # Filter down to residential objects only
    dataframe = dataframe[dataframe['prop_type_cd'] == 'R'].reset_index(
        drop=True)

    # Convert SOS Charter date to datetime type
    dataframe['SOS Charter Date'] = pd.to_datetime(
        dataframe['SOS Charter Date'], format='%Y%m%d')
    # Convert deed date
    dataframe['deed_dt'] = pd.to_datetime(
        dataframe['deed_dt'], format='%m%d%Y')

    # Add main three features from Dutch paper
    # Owner is a just-established company
    # Subtract the two dates and see if the difference is within the specificied timeframe
    # (Default is 365 days)
    dataframe['deed_charter_diff'] = dataframe['deed_dt'] - \
        dataframe['SOS Charter Date']

    # 365 days is the defualt definition for "just-established"
    dataframe['just_established_owner'] = np.where(
        dataframe['deed_charter_diff'] <= cfg['recently_founded'], 1, 0)

    # Foreign Owner
    dataframe['foreign_based_owner'] = np.where(
        dataframe['py_addr_country'] != 'US', 1, 0)

    # Make a new column for owners based outside of the state of Texas
    dataframe['out_of_state_owner'] = np.where(
        dataframe['py_addr_state'] != cfg['out_of_state'], 1, 0)

    # Unusual Price Fluctuations
    # YoY % difference from 2018 to 2019
    price_change(dataframe,'market_value','bexar_2018_market_value','yoy_diff_2019')
    price_change(dataframe,'bexar_2018_market_value','bexar_2017_market_value','yoy_diff_2018')
    price_change(dataframe,'bexar_2017_market_value','bexar_2016_market_value','yoy_diff_2017')
    price_change(dataframe,'bexar_2016_market_value','bexar_2015_market_value','yoy_diff_2016')

    # Convert several columns to categories and rename
    dataframe['property_zip'] = dataframe.situs_zip.astype('category')
    dataframe['neighborhood_code'] = dataframe.hood_cd.astype('category')
    dataframe['owner_zip_code'] = dataframe.py_addr_zip.astype('category')

    return dataframe, cfg

def binarize_features():
    """Convert categorical features to binary."""
    dataframe, cfg = engineer_features()

    # Feature for if the owner has decided to make confidential their information
    # Convert from T/F to 1/0
    dataframe['py_confidential_flag'] = np.where(
        dataframe['py_confidential_flag'] == 'T', 1, 0)

    # Do the same for flag suppression on address for owner
    dataframe['py_address_suppress_flag'] = np.where(
        dataframe['py_address_suppress_flag'] == 'T', 1, 0)

    # Binarize if the owner's address is deliverable by mail
    dataframe['py_addr_ml_deliverable'] = np.where(
        dataframe['py_addr_ml_deliverable'] == 'Y', 1, 0)

    # Binarize if the property has an entity agent and drop entity agent columns
    dataframe['entity_agent_binary'] = np.where(
        dataframe['entity_agent_id'] == 0, 0, 1)

        # Make two columns for legal entities and those that are "likely" legal
    # entities based on name
    dataframe['owner_legal_person'] = np.where(
        (dataframe.dba.notna() | dataframe['Taxpayer Name'].notna()), 1, 0)

    # Name-based legal entities (minus trusts)
    dataframe['owner_likely_company'] = np.where(
        dataframe.cleaned_name.str.contains('|'.join(cfg['terms_for_company'])) == True, 1, 0)

    # Trusts are not covered by the GTO but are a common vehicle for money laundering
    # Make a separate column for them
    dataframe['owner_is_trust'] = np.where(
        dataframe.cleaned_name.str.contains(r'TRUST') == True, 1, 0)

    # Binarize if the appraiser is confidential
    dataframe['appr_confidential_flag'] = np.where(
        dataframe['appr_confidential_flag'] == 'T', 1, 0)

    # Get dummies to make binary columns for each company status code
    bexar_sos_status = pd.get_dummies(
        dataframe['SOS Status Code'], prefix='sos_status_code_')
    dataframe = pd.concat([dataframe, bexar_sos_status], axis=1)

    # Feature for if the owner owns multiple properties
    taxpayer_count = dataframe['Taxpayer Number'].value_counts()
    multiple_prop_list = taxpayer_count[taxpayer_count > 1].index.values

    # Multiple appearances of the same py_owner_id will also indicate the owner owns multiple properties
    py_owner_id_count = dataframe.py_owner_id.value_counts()
    multi_prop_list2 = py_owner_id_count[py_owner_id_count>1].index.values
    # Binarize based on the two columns
    dataframe['owner_owns_multiple'] = np.where((dataframe['py_owner_id'].isin(multi_prop_list2))
                                                       | (dataframe['Taxpayer Number'].isin(multiple_prop_list)), 1, 0)

    # Binarize if the ownership is split among several parties
    dataframe['partial_owner'] = np.where(
        dataframe['partial_owner'] == 'T', 1, 0)

    # GTO stipulations - Price above 300k and company owner
    dataframe['two_gto_reqs'] = np.where(((dataframe.owner_legal_person == 1) |
                                                 (dataframe.owner_likely_company)) &
                                                (dataframe.market_value >= 300000), 1, 0)


    return dataframe

def drop_columns():
    """Drop extraneous columns."""
    dataframe = binarize_features()
    # Dropping all other entity agent columns
    dataframe.drop(columns=[
        'entity_agent_id', 'entity_agent_name',
        'entity_agent_addr_line1', 'entity_agent_addr_line2',
        'entity_agent_addr_line3', 'entity_agent_city',
        'entity_agent_state', 'entity_agent_country'],
        inplace=True)

    # Dropping all chief appraiser data
    dataframe.drop(columns=[
        'ca_agent_id', 'ca_agent_name', 'ca_agent_addr_line1',
        'ca_agent_addr_line2', 'ca_agent_addr_line3', 'ca_agent_city',
        'ca_agent_state', 'ca_agent_country', 'ca_agent_zip'],
        inplace=True)

     # Drop unnecessary columns
    dataframe.drop(
        columns=['sup_num', 'sup_action', 'sup_cd', 'sup_desc', 'udi_group','appr_address_suppress_flag'],
        inplace=True)

        # Drop the old columns
    dataframe.drop(
        columns=['situs_zip','hood_cd','py_addr_zip'], inplace=True)

    return dataframe

def trim_values():
    """
    Trim down market values that are > 99.9% percentile and <= $100.
    On the top end, it's mostly military bases and on the low it's publicly-
    owned land and potentially mis-coded properties.
    """
    dataframe = drop_columns()

    top_pctile = dataframe.market_value.quantile(0.999)
    dataframe = dataframe[(dataframe.market_value > 100) & (
        dataframe.market_value < top_pctile)].reset_index(drop=True)

    # Price per square foot
    dataframe['price_psf'] = dataframe.market_value / \
        dataframe.Sq_ft
    # Fill the divide by zero infs with nan
    dataframe.loc[np.isinf(
        dataframe['price_psf']), 'price_psf'] = np.nan

    return dataframe

def merge_labels():
    """Merge labels for criminally-linked properties from criminal_properties_labels.csv."""
    dataframe = trim_values()

    print('Merging labels for criminally-linked properties.')
    labels = pd.read_csv('./data/labels/criminal_properties_labels.csv')
    labels['prop_id'] = labels['Property ID']
    labels['crim_prop'] = 1
    labels['crim_address'] = labels.Address
    dataframe = pd.merge(dataframe,labels[['prop_id','crim_address','crim_prop']],
                           how='left',on='prop_id')
    dataframe.crim_prop.fillna(value=0,inplace=True)

    return dataframe

def export_dataframe():
    """Export now-processed dataframe as bexar_processed.h5."""
    dataframe = merge_labels()

    print('Exporting dataframe to h5 file.')

    # Mkdir
    if not os.path.exists('./data/processed'):
        os.makedirs('./data/processed')

    dataframe.to_hdf('./data/processed/bexar_processed.h5',
                            key='bexar_processed.h5',
                            mode='w',
                            format='t'
                            )

    print('Dataframe has been exported as bexar_processed.h5 to "processed" data folder.')

def clean_add_features():
    """Run export_dataframe()."""
    export_dataframe()

if __name__ == '__main__':
    clean_add_features()
