from sqlalchemy import create_engine
import psycopg2
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

import scikitplot as skplt



pd.set_option('display.max_columns', 50)
sns.set_style('whitegrid')
localdb = create_engine(
    'postgresql://',
    echo=False,
    pool_recycle=300,
    creator=lambda _: psycopg2.connect(service="rockets-local"))


df = pd.read_sql_query("SELECT * FROM members.good_customers", localdb)

df['minutes_to_convert'] = (
    df['became_member_at'] - df['created_at']) / pd.Timedelta(
        minutes=1, seconds=0)
df['hours_to_convert'] = (
    df['became_member_at'] - df['created_at']) / pd.Timedelta(
        hours=1, seconds=0)

df = df.loc[df['num_kids'] < 6, ]
df['domain'] = df['email'].apply(lambda x: x.split('@')[1])

jan = df['created_at'].map(lambda x: x > pd.to_datetime('2018-01-01'))
jun = df['created_at'].map(lambda x: x < pd.to_datetime('2018-06-01'))

domains = df['domain'].map(lambda x: x in ['gmail.com', 'yahoo.com', 'hotmail.com', 'aol.com', 'me.com'])

jj = df[jan & jun & domains]

jj.dropna(inplace=True)

train, test = train_test_split(jj, test_size=0.2, random_state=42)
train.reset_index(drop=True, inplace=True)
test.reset_index(drop=True, inplace=True)

X = train.loc[:, ['service_fee', 'service_fee_enabled',
    'num_boys', 'num_girls', 'is_referred',
    'first_click_source', 'OS',
    'employed_civil_percent', 'unemploym_rate_civil',
    'employed_female_percent', 'med_hh_income',
    'med_fam_income', 'households_density',
    'married_couples__density', 'families_density',
    'households_with_kids_density', 'females_density',
    'members_in_zip', 'minutes_to_convert', 'domain']]
Y = train['avg_keep_rate']

x = test.loc[:, ['service_fee', 'service_fee_enabled',
    'num_boys', 'num_girls', 'is_referred',
    'first_click_source', 'OS',
    'employed_civil_percent', 'unemploym_rate_civil',
    'employed_female_percent', 'med_hh_income',
    'med_fam_income', 'households_density',
    'married_couples__density', 'families_density',
    'households_with_kids_density', 'females_density',
    'members_in_zip', 'minutes_to_convert', 'domain']]
y = test['avg_keep_rate']

rf = RandomForestRegressor(n_estimators=100,
                           min_samples_leaf=1,
                           n_jobs=-1)

Xcols = X.columns.tolist()
categoricals = [Xcols.index(col) for col in Xcols if X[col].dtypes in ['object', 'bool']]

Xcopy = X.copy()
Ycopy = Y.copy()

encoders = {}
for col in X.columns:
    if X[col].dtypes in ['object', 'bool']:
        encoders[col] = LabelEncoder()
        encoders[col].fit(X[col].values)
        X[col] = encoders[col].transform(X[col])
enc = OneHotEncoder(categorical_features=categoricals, sparse=False)
enc.fit(X.as_matrix())
Xenc = enc.transform(X)

for col in x.columns:
    if x[col].dtypes in ['object', 'bool']:
        x[col] = encoders[col].transform(x[col])
xenc = enc.transform(x)

rf.fit(Xenc, Y)