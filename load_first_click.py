import pandas as pd
from sqlalchemy import create_engine
import psycopg2

localdb = create_engine(
    'postgresql://',
    echo=False,
    pool_recycle=300,
    creator=lambda _: psycopg2.connect(service="rockets-local"))

fc = pd.read_csv(
    "/Users/emmanuele/Data/First Click Source Medium - User Summary.csv",
    dtype={'User ID': 'str'},
    index_col=None)

fc = fc.loc[fc['User ID'].notnull(),
            ['User ID', 'First Click Classification.1']]
fc.columns = ['user_id', 'first_click_source']

fc.to_sql(
    "first_click_attribution",
    localdb,
    schema='members',
    if_exists='replace',
    index=False)

# data.to_sql(
#     "good_customers",
#     localdb,
#     schema="members",
#     if_exists='replace',
#     index=False)
