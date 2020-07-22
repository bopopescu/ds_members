import psycopg2
import pandas as pd
from sqlalchemy import create_engine
from dateutil.relativedelta import relativedelta

from query_dbs import kids_and_zip
from query_dbs import avg_keep_rates
from query_dbs import good_customers
from query_dbs import referrals
from query_dbs import first_clicks
from query_dbs import user_agents

from ua_parser import user_agent_parser as up

subordinate = psycopg2.connect(service="rockets-subordinate")
segment = psycopg2.connect(service='rockets-segment')

localdb = create_engine(
    'postgresql://',
    echo=False,
    pool_recycle=300,
    creator=lambda _: psycopg2.connect(service="rockets-local"))

gc = pd.read_sql_query(good_customers, subordinate)
kz = pd.read_sql_query(kids_and_zip, subordinate)
kr = pd.read_sql_query(avg_keep_rates, subordinate)
refs = pd.read_sql_query(referrals, subordinate)
uas = pd.read_sql_query(user_agents, segment)
census = pd.read_sql("SELECT * FROM members.census", localdb)
census.drop('index', axis=1, inplace=True)

df = pd.merge(kr, kz, how='right')
df['created_at'] = pd.to_datetime(df['created_at'])
df = pd.merge(df, refs)
idx1 = df.loc[(df['user_id'] == 698) & (df['is_referred'] == False), ].index
idx2 = df.loc[(df['user_id'] == 142711) & (df['is_referred'] == False), ].index
df.drop(idx1, inplace=True)
df.drop(idx2, inplace=True)

fc = pd.read_sql_query(first_clicks, localdb)
df = pd.merge(df, fc, how='left')

df = pd.merge(df, uas, how='left')
df['device'] = \
    df['context_user_agent'].apply(lambda x: up.ParseDevice(x)['family']
                                   if pd.notnull(x) else None)
df['OS'] = df['context_user_agent'].apply(
    lambda x: up.ParseOS(x)['family'] if pd.notnull(x) else None)
df.loc[df['OS'] == 'Mac OS X', 'device'] = 'Mac'
df.drop('context_user_agent', axis=1, inplace=True)

df['zipcode'] = df['zipcode'].str[:5]

mems_per_zip = \
    df.groupby('zipcode')['user_id'].\
    count().\
    reset_index().\
    rename(columns={'user_id': 'members_in_zip'})

df = pd.merge(df, gc, how='left')
df = pd.merge(df, census, how='left', left_on='zipcode', right_on='zip')
df.drop('zip', axis=1, inplace=True)

df = pd.merge(df, mems_per_zip)

# Cleanup
two_weeks_ago = df['created_at'].max() - relativedelta(weeks=2)

# remove new customers
df = df.loc[df['created_at'] < two_weeks_ago, ]

df['is_good_customer'].fillna(False, inplace=True)
df.reset_index(drop=True, inplace=True)

# remove recent boxes (within last two weeks)
to_remove = df.loc[(df['avg_keep_rate'].isnull()) &
                   (df['is_good_customer'] == True), ].index
df.drop(to_remove, inplace=True)
df.reset_index(drop=True, inplace=True)

to_remove = df.loc[(df['is_member'] == False) &
                   (df['state'] == 'subscription_member'), ].index
df.drop(to_remove, inplace=True)
df.reset_index(drop=True, inplace=True)

to_remove = df.loc[(df['num_boxes'].isnull()) &
                   (df['state'] == 'subscription_member'), 'user_id'].index
df.drop(to_remove, inplace=True)
df.reset_index(drop=True, inplace=True)


# is_good = df['is_good_customer'].map(lambda x: x == True)
# is_first_half_2018 = df['created_at'].map(
#     lambda x: x >= pd.to_datetime('2018-01-01') and x <= pd.to_datetime('2018-06-02')
# )

# census = pd.read_csv(
#     "/Users/emmanuele/Data/census.csv", dtype={'zip': 'str'}, index_col=None)

# df.to_sql(
#     "good_customers",
#     localdb,
#     schema="members",
#     if_exists='replace',
#     index=False)

# Experian 2017 data. Not useful at the moment.
# experian = """
#     SELECT user_id,
#            event_body->'ELSGenericMessage'->'Stage2Data'->'Livu'->'MOSAIC_TYPE' AS mosaic_type
#     FROM user_messages
#     WHERE event_name ilike '%experian%';
# """
# hearken = psycopg2.connect(service="rockets-hearken")
# exper = pd.read_sql_query(experian, hearken)
# df = pd.merge(gc, exper)
# df = df.loc[df['mosaic_type'] != '', ]
