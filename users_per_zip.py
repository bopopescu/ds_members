from sqlalchemy import create_engine
import psycopg2
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', 50)
sns.set_style('whitegrid')
localdb = create_engine(
    'postgresql://',
    echo=False,
    pool_recycle=300,
    creator=lambda _: psycopg2.connect(service="rockets-local"))

df = pd.read_sql_query("SELECT * FROM members.good_customers", localdb)
df = df.loc[df['num_kids'] < 6, ]
df['domain'] = df['email'].apply(lambda x: x.split('@')[1])

d = df.groupby(['zipcode', 'is_good_customer'])['user_id'].count().\
    reset_index().\
    rename(columns={'user_id': 'users'})

d = d.pivot(index='zipcode', columns='is_good_customer', values='users').\
    fillna(0).\
    reset_index()

d.columns = ['zip', 'bad', 'good']

for col in d.columns:
    if d[col].dtype == 'float':
        d[col] = d[col].astype('int')

census = pd.read_sql("SELECT * FROM members.census", localdb)
census.drop('index', axis=1, inplace=True)

c = census.loc[:, ['zip', 'households', 'med_fam_income']]
c.fillna(0, inplace=True)

t = pd.merge(d, c, how='outer')
t['bad'].fillna(0, inplace=True)
t['good'].fillna(0, inplace=True)
t['tot'] = t['bad'] + t['good']

t['tot_penetration'] = t['tot'] / t['households']
t['good_penetration'] = t['good'] / t['households']

dt = t.loc[t['households'] > 0, ]

rich_zips = dt['med_fam_income'].map(lambda x: x > 110000)

dt.loc[(dt['tot_penetration'] < dt['tot_penetration'].quantile(.99)
        ) & rich_zips, 'tot_penetration'].hist(bins=20)

dt.loc[dt['tot_penetration'] < dt['tot_penetration'].quantile(.99),
       'tot_penetration'].hist(bins=20)
