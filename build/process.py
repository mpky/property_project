import os
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


def clean_add_features():
    # Read in preprocessed simplefilter
    print("Reading in preprocessed h5 file")
    corp_prop_merged = pd.read_hdf('./data/preprocessed/bexar_preprocessed.h5')

    print("Engineering features")
    # Filter down to residential objects only
    corp_prop_merged = corp_prop_merged[corp_prop_merged.prop_type_cd == 'R'].reset_index(
        drop=True)

    # Convert SOS Charter date to datetime type
    corp_prop_merged['SOS Charter Date'] = pd.to_datetime(
        corp_prop_merged['SOS Charter Date'], format='%Y%m%d')
    # Convert deed date
    corp_prop_merged['deed_dt'] = pd.to_datetime(
        corp_prop_merged['deed_dt'], format='%m%d%Y')

    # Add main three features from Dutch paper
    # Owner is a just-established company
    # Subtract the two dates and see if the difference is within 365 days
    corp_prop_merged['deed_charter_diff'] = corp_prop_merged['deed_dt'] - \
        corp_prop_merged['SOS Charter Date']

    # If this is <= 365, consider "just-established"
    corp_prop_merged['just_established_owner'] = np.where(
        corp_prop_merged['deed_charter_diff'] <= '365 days', 1, 0)
    # 16,349 properties in the dataset are considered to be owned by 'just-established' companies

    # Foreign Owner
    corp_prop_merged['foreign_based_owner'] = np.where(
        corp_prop_merged['py_addr_country'] != 'US', 1, 0)

    # Unusual Price Fluctuations
    # YoY % difference from 2018 to 2019
    corp_prop_merged['yoy_diff_2019'] = (
        corp_prop_merged.market_value - corp_prop_merged.bexar_2018_market_value) / corp_prop_merged.bexar_2018_market_value

    # YoY % difference from 2017 to 2018
    corp_prop_merged['yoy_diff_2018'] = (corp_prop_merged.bexar_2018_market_value -
                                         corp_prop_merged.bexar_2017_market_value) / corp_prop_merged.bexar_2017_market_value

    # YoY % difference from 2016 to 2017
    corp_prop_merged['yoy_diff_2017'] = (corp_prop_merged.bexar_2017_market_value -
                                         corp_prop_merged.bexar_2016_market_value) / corp_prop_merged.bexar_2016_market_value

    # YoY % difference from 2015 to 2016
    corp_prop_merged['yoy_diff_2016'] = (corp_prop_merged.bexar_2016_market_value -
                                         corp_prop_merged.bexar_2015_market_value) / corp_prop_merged.bexar_2015_market_value

    # Drop unnecessary columns
    corp_prop_merged.drop(
        columns=['sup_num', 'sup_action', 'sup_cd', 'sup_desc', 'udi_group'], inplace=True)

    # make a new column for owners based outside of the state of Texas
    corp_prop_merged['out_of_state_owner'] = np.where(
        corp_prop_merged['py_addr_state'] != 'TX', 1, 0)

    # Feature for if the owner has decided to make confidential their information
    # Convert from T/F to 1/0
    corp_prop_merged['py_confidential_flag'] = np.where(
        corp_prop_merged['py_confidential_flag'] == 'T', 1, 0)

    # Do the same for flag suppression on address for owner
    corp_prop_merged['py_address_suppress_flag'] = np.where(
        corp_prop_merged['py_address_suppress_flag'] == 'T', 1, 0)

    # Binarize if the owner's address is deliverable by mail
    corp_prop_merged['py_addr_ml_deliverable'] = np.where(
        corp_prop_merged['py_addr_ml_deliverable'] == 'Y', 1, 0)

    # Binarize if the property has an entity agent and drop entity agent columns
    corp_prop_merged['entity_agent_binary'] = np.where(
        corp_prop_merged.entity_agent_id == 0, 0, 1)

    # Dropping all other entity agent columns
    corp_prop_merged.drop(columns=[
        'entity_agent_id', 'entity_agent_name',
        'entity_agent_addr_line1', 'entity_agent_addr_line2',
        'entity_agent_addr_line3', 'entity_agent_city',
        'entity_agent_state', 'entity_agent_country'],
        inplace=True)

    # Dropping all chief appraiser data
    corp_prop_merged.drop(columns=[
        'ca_agent_id', 'ca_agent_name', 'ca_agent_addr_line1',
        'ca_agent_addr_line2', 'ca_agent_addr_line3', 'ca_agent_city',
        'ca_agent_state', 'ca_agent_country', 'ca_agent_zip'],
        inplace=True)

    # Make two columns for legal entities and those that are "likely" legal
    # entities based on name
    corp_prop_merged['owner_legal_person'] = np.where(
        (corp_prop_merged.dba.notna() | corp_prop_merged['Taxpayer Name'].notna()), 1, 0)

    # Name-based legal entities (minus trusts)
    terms = [
        'SA DE CV', 'PARTNERSHIP', 'LP', 'LLP', 'SOCIEDAD',
        'LLC', 'CORP', 'COMPANY', 'LTD', 'INC', 'JOINT VENTURE',
        'REAL ESTATE', 'HOLDING', 'GROUP'
    ]

    corp_prop_merged['owner_likely_company'] = np.where(
        corp_prop_merged.cleaned_name.str.contains('|'.join(terms)) == True, 1, 0)

    # Trusts are not covered by the GTO but are a common vehicle for money laundering
    # Will make a separate column for them
    corp_prop_merged['owner_is_trust'] = np.where(
        corp_prop_merged.cleaned_name.str.contains(r'TRUST') == True, 1, 0)

    # Convert appraiser confidential to 1/0
    corp_prop_merged['appr_confidential_flag'] = np.where(
        corp_prop_merged['appr_confidential_flag'] == 'T', 1, 0)

    # Will use get dummies to make binary columns for each status
    bexar_sos_status = pd.get_dummies(
        corp_prop_merged['SOS Status Code'], prefix='sos_status_code_')
    corp_prop_merged = pd.concat([corp_prop_merged, bexar_sos_status], axis=1)

    # Feature for if the owner owns multiple properties
    taxpayer_count = corp_prop_merged['Taxpayer Number'].value_counts()
    multiple_prop_list = taxpayer_count[taxpayer_count > 1].index.values

    # Multiple appearances of the same py_owner_id will also indicate the owner owns multiple properties
    py_owner_id_count = corp_prop_merged.py_owner_id.value_counts()
    multi_prop_list2 = py_owner_id_count[py_owner_id_count>1].index.values
    # Binarize based on the two columns
    corp_prop_merged['owner_owns_multiple'] = np.where((corp_prop_merged['py_owner_id'].isin(multi_prop_list2))
                                                       | (corp_prop_merged['Taxpayer Number'].isin(multiple_prop_list)), 1, 0)

    # Convert from T/F to 1/0
    corp_prop_merged['partial_owner'] = np.where(
        corp_prop_merged['partial_owner'] == 'T', 1, 0)

    # Trim down market values that are above the 99.9% percentile and over 100
    # On the top end, it's mostly military bases and on the low it's publicly-
    # owned land
    top_pctile = corp_prop_merged.market_value.quantile(0.999)
    corp_prop_merged = corp_prop_merged[(corp_prop_merged.market_value > 100) & (
        corp_prop_merged.market_value < top_pctile)].reset_index(drop=True)

    # Price per square foot
    corp_prop_merged['price_psf'] = corp_prop_merged.market_value / \
        corp_prop_merged.Sq_ft
    # Fill the divide by zero infs with nan
    corp_prop_merged.loc[np.isinf(
        corp_prop_merged['price_psf']), 'price_psf'] = np.nan

    # GTO stipulations - Price above 300k and company owner
    corp_prop_merged['two_gto_reqs'] = np.where(((corp_prop_merged.owner_legal_person == 1) |
                                                 (corp_prop_merged.owner_likely_company)) &
                                                (corp_prop_merged.market_value >= 300000), 1, 0)

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
