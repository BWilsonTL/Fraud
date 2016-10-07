
import os
import sys
import pandas as pd
import numpy as np
from core.data import frameformatting as ff
from core.data import frameprep as fp
from core.data import fieldtools as ft
from core import sfconnector as sf
import datetime
from ggplot import *
from sklearn import preprocessing

def static_join(fraud_path, email_path):
    file_1 = pd.read_csv(fraud_path)
    file_2 = pd.read_csv(email_path, sep='\t')
    file_join = pd.merge(file_1, file_2, left_on='Email_ADD', right_on='EMAIL_ADD', how='right')
    return file_join[['MEMBER_SK', 'Blocked', 'VRP', 'Credit_Count', 'Action']]


mbr_attr_qry = """
SELECT DISTINCT
    D_MBR.MEMBER_SK,
    D_MBR.JOIN_DT,
    D_MBR.EMAIL_DOMAIN_TXT,
    D_MBR.LAST_LOGIN_DT,
    D_MBR.LOGIN_CNT,
    D_MBR.EMAIL_UNSUB_DT,
    D_MBR.LOYALTY_TIER_NM,
    D_MBR.GENDER_CD,
    D_MBR.MARITAL_STATUS,
    D_MBR.AGE_RANGE,
    D_MBR.INCOME_RANGE,
    D_MBR.FIRST_ORDER_DT,
    D_MBR.LAST_ORDER_DT,
    D_MBR.C_ORDER_CNT,
    D_MBR.ORDER_QTY,
    D_MBR.CANCEL_QTY,
    D_MBR.RETURN_QTY,
    D_MBR.ORDER_AMT,
    D_MBR.CANCEL_AMT,
    D_MBR.RETURN_AMT,
    D_IP_GEOLOCATION.COUNTRY_CD "COUNTRY",
    D_IP_GEOLOCATION.SUBDIVISION_1_NM "STATE",
    D_IP_GEOLOCATION.CITY_NM "CITY",
    D_IP_GEOLOCATION.POSTAL_CD "ZIP"
FROM F_SITE_ACTIVITY
    LEFT JOIN D_IP_GEOLOCATION
    ON F_SITE_ACTIVITY.IP_ADDR_SK = D_IP_GEOLOCATION.IP_ADDR_SK
        LEFT JOIN D_MBR
        ON F_SITE_ACTIVITY.MEMBER_SK = D_MBR.MEMBER_SK
    WHERE FIRST_ORDER_DT < current_date
"""

# query snowflake for session activity.
session_activity_qry = """
SELECT
    MAX(SESSION_SK) "SESSION_SK",
    MIN(ACTIVITY_DT) as ACTIVITY_START,
    MAX(ACTIVITY_DT) as ACTIVITY_END,
    MAX(MEMBER_SK) "MEMBER_SK",
    COUNT(BOUTIQUE_SK) "ITEM_VIEWS",
    SUM(ACTIVITY_CNT) "TOTAL_ACTIVITY",
    MAX(DEVICE_CATEGORY_ID) "DEVICE_ID"
FROM A_BTQ_VISIT_HRLY
GROUP BY A_BTQ_VISIT_HRLY.MEMBER_SK, A_BTQ_VISIT_HRLY.SESSION_SK
"""
home_path = os.path.expanduser('~')
static_path = os.path.join(home_path, 'Documents/Fraud_2')
f_path = os.path.join(static_path, 'Non_Receipt_Data.csv')
e_path = os.path.join(static_path, 'SK_TO_EMAIL_ADDR')
sf_uname, sf_pwd = sf.account_details(os.path.join(home_path, 'Documents/snowflakeauth.txt'))

age_dict = {
    '18 - 25': '20',
    '26 - 35': '30',
    '36 - 45': '40',
    '46 - 55': '50',
    '56 plus': '60',
    'n/a': '35'
}


def member_prep():
    todaydate = datetime.date.today()
    # import the static table
    fraud_import = static_join(f_path, e_path)
    # Clean up the user table.
    fraud_cleaned = fp.df_field_fill_clean(fraud_import, ['Blocked', 'VRP', 'Action'], 'NA')
    fraud_cleaned = fp.df_field_fill_clean(fraud_cleaned, ['Credit_Count'], 0.0)
    # query snowflake for member attributes
    member_attributes_raw = sf.SnowConnect(mbr_attr_qry, sf_uname, sf_pwd,
                                       'ADW_QA_DB', 'ADM', 'ADW_QA_QRY_RL', 'QUERY_WH').execute_query()
    # create features in member_attributes
    member_attributes = member_attributes_raw.copy()
    # cleanup the login count field
    member_attributes = fp.df_field_fill_clean(member_attributes, 'LOGIN_CNT', 0)
    # create bool column for email subscribe
    member_attributes['EMAIL_SUB'] = (member_attributes['EMAIL_UNSUB_DT'] > todaydate).astype(int)
    member_attributes.drop('EMAIL_UNSUB_DT', axis=1, inplace=True)
    # Get the total shipped order qty for the member
    member_attributes['Effective_Order_QTY'] = (member_attributes['ORDER_QTY']
                                                - member_attributes['CANCEL_QTY']
                                                - member_attributes['RETURN_QTY'])

    member_attributes['Effective_Order_AMT'] = (member_attributes['ORDER_AMT']
                                                - member_attributes['CANCEL_AMT']
                                                - member_attributes['RETURN_AMT'])
    # Convert the age groupings
    member_attributes['Ages'] = member_attributes['AGE_RANGE'].apply(age_dict.get)

    # preserve only the fields that are useful:
    member_filter = member_attributes[['MEMBER_SK', 'JOIN_DT', 'LAST_LOGIN_DT', 'LOGIN_CNT',
                                       'EMAIL_SUB', 'LOYALTY_TIER_NM', 'GENDER_CD',
                                       'MARITAL_STATUS', 'Ages', 'COUNTRY',
                                       'ORDER_QTY', 'CANCEL_QTY', 'RETURN_QTY', 'ORDER_AMT',
                                       'CANCEL_AMT', 'RETURN_AMT', 'Effective_Order_QTY',
                                       'Effective_Order_AMT']]
    # calculate the datediff between datetimes / dates.
    ft.deltadate(member_filter, 'Active_Age_Days',
                 member_filter['LAST_LOGIN_DT'], member_filter['JOIN_DT'], 'D')
    ft.deltadate(member_filter, 'Account_Age', todaydate, member_filter['JOIN_DT'], 'D')
    ft.deltadate(member_filter, 'recency', todaydate, member_filter['LAST_LOGIN_DT'], 'D')
    # drop the datetime fields
    date_fields = ['LAST_LOGIN_DT', 'JOIN_DT']
    for i in date_fields:
        member_filter.drop(i, axis=1, inplace=True)
    # basis expansion of grouped categorical data
    base_list = ['MARITAL_STATUS', 'Ages', 'LOYALTY_TIER_NM', 'GENDER_CD', 'COUNTRY']
    for i in base_list:
        ff.basis_expansion(member_filter, i)
    # bin expansion of continuous data
    be_list = [['Active_Age_Days', 4, 'auto'], ['Account_Age', 5, 'auto'], ['recency', 4, 'auto'],
               ['LOGIN_CNT', 5, 'auto']]
    for i in be_list:
        ft.bin_expansion(member_filter, i[0], ft.bin_generator(i[1]), source_del=True, bin_scale=i[2])
    # do column standardization for some fields.
    scaler = preprocessing.MinMaxScaler()
    scaling_list = ['ORDER_QTY', 'CANCEL_QTY', 'RETURN_QTY', 'ORDER_AMT', 'CANCEL_AMT', 'RETURN_AMT',
                    'Effective_Order_QTY', 'Effective_Order_AMT']
    member_filter[scaling_list] = scaler.fit_transform(member_filter[scaling_list])

    return member_filter

member_data = member_prep()





def session_prep():

    # get stats on activity for members
    session_activity_raw = sf.SnowConnect(session_activity_qry, sf_uname, sf_pwd,
                                      'ADW_PRD_DB', 'ADM', 'ADW_PRD_QRY_RL', 'QUERY_WH').execute_query()

    # create a copy
    scaler = preprocessing.MinMaxScaler()
    session_activity = session_activity_raw.copy()
    session_list = ['BOUTIQUE_CNT', 'TOTAL_ACTIVITY']
    session_activity[session_list] = scaler.fit_transform(session_activity[session_list])

    # calculate the time duration of the session
    ft.deltadate(session_activity, 'Session_Time', session_activity['ACTIVITY_END'],
                 session_activity['ACTIVITY_START'], 's')


    """
    ggplot(aes(x='LOGIN_CNT'), data=member_attributes) + geom_histogram()
    ggplot(member_attributes, aes(x='Active_Age_Days')) + geom_histogram()
    """


def predictor_merge():
    pass


class Main(object):
    def __init__(self):
        pass

    def main(self):
        pass



"""

Test script for predicting charge-back events.


steps:
1. clean the classifier data
2. join to all member data
3. pull member usage information

"""

if __name__ == '__main__':

    pass

