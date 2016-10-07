# Chargeback analysis and batch processing job

import sys
import os
import pandas as pd
from core.data import frameformatting as ff
import numpy as np
import ggplot
from sklearn.cluster import KMeans
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier

fraud_path = os.path.join(os.path.expanduser('~'), 'Documents/Fraud_2/Non_Receipt_Data.csv')
fraud_input = pd.read_csv(fraud_path)

emails_path = os.path.join(os.path.expanduser('~'), 'Documents/Fraud_2/SK_TO_EMAIL_ADDR')
email_input = pd.read_csv(emails_path, sep='\t')

# join the data on email.
joined_email = pd.merge(email_input, fraud_input, left_on='EMAIL_ADD',
                        right_on='Email_ADD', how='left')

charge_back_data = joined_email[['MEMBER_SK', 'Blocked', 'VRP', 'Credit_Count', 'Action']]

# clean up the dataset
charge_back_data['Blocked'].fillna('NA', inplace=True)
charge_back_data['Blocked'] = charge_back_data['Blocked'].str.upper()
charge_back_data['VRP'].fillna('NA', inplace=True)
charge_back_data['VRP'] = charge_back_data['VRP'].str.upper()
charge_back_data['Credit_Count'].fillna(0, inplace=True)
charge_back_data['Action'].fillna('NA', inplace=True)
charge_back_data['Action'] = charge_back_data['Action'].str.upper()


def cat_fact(df, field):
    labels, levels = pd.factorize(df[field])
    # create fields for each of the levels
    total_cnt = levels.size
    for i in range(total_cnt):
        field_name = field + '_' + levels[i]
        df.loc[df[field] == levels[i], field_name] = 1
        df[field_name].fillna(0, inplace=True)
    df.drop([field], inplace=True, axis=1)
    return df

cat_fact(charge_back_data, 'Blocked')
cat_fact(charge_back_data, 'VRP')
cat_fact(charge_back_data, 'Action')

ff.basis_expansion(charge_back_data, 'Blocked', replace=False, sort_flag=True, order='descending')




################ demo kmeans clustering ##########################

dataz = charge_back_data.ix[:, 1:]

train, test = train_test_split(dataz, test_size=0.2)



estimator_list = {
                  'kmeans_3': KMeans(n_clusters=3),
                  'kmeans_5': KMeans(n_clusters=5)
                  }


for name, estimator in estimator_list.items():
    fitted = estimator.fit(train)
    labels = estimator.labels_
    train[name] = labels
    print(fitted)

data_group_3 = train.groupby(by='kmeans_3').sum()
print(data_group_3)
data_group_5 = train.groupby(by='kmeans_5').sum()
print(data_group_5)

# check the power of it ->   m_eval = fitted.predict(test)

target = train['Blocked_YES']
training = train[['Credit_Count', 'VRP_NA', 'VRP_YES', 'VRP_NO', 'Action_NA',
                             'Action_OK TO CREDIT', 'Action_BLOCKED, DO NOT CREDIT',
                             'Action_SEND TO CORP', 'Action_DO NOT CREDIT']]
d_test = test[['Credit_Count', 'VRP_NA', 'VRP_YES', 'VRP_NO', 'Action_NA',
                             'Action_OK TO CREDIT', 'Action_BLOCKED, DO NOT CREDIT',
                             'Action_SEND TO CORP', 'Action_DO NOT CREDIT']]

rfmodel = RandomForestClassifier(n_estimators=100)
rfmodel.fit(training, target)
test['Predicted_val'] = rfmodel.predict(d_test)


compare = test[['Blocked_YES', 'Predicted_val']]


