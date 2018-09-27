import psycopg2
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from query_dbs import query_kid_preferences
from df2gspread import gspread2df as g2d
from df2gspread import df2gspread as d2g

from sklearn.preprocessing import Imputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from ua_parser import user_agent_parser as up

pd.set_option('display.max_columns', 100)
slave = psycopg2.connect(service="rockets-slave")
segment = psycopg2.connect(service='rockets-segment')
stitch = psycopg2.connect(service='rockets-stitch')

localdb = create_engine(
    'postgresql://',
    echo=False,
    pool_recycle=300,
    creator=lambda _: psycopg2.connect(service="rockets-local"))

b = pd.read_sql_query(
    """
    WITH nkids AS (SELECT user_id,
                        sum(
                            CASE WHEN gender :: text = 'boys' :: text THEN 1 ELSE 0 END) AS  num_boys,
                        sum(
                            CASE WHEN gender :: text = 'girls' :: text THEN 1 ELSE 0 END) AS num_girls,
                        count(*) AS                                                          num_kids
                FROM kid_profiles
                GROUP BY user_id
                ORDER BY user_id)

    SELECT b.id               AS box_id,
        u.id               AS user_id,
        p.id               AS kid_id,
        b.state            AS box_state,
        u.state            AS user_state,
        service_fee_amount,
        u.created_at,
        b.created_at       AS box_created_at,
        u.became_member_at,
        a.zipcode,
        b.approved_at,
        nkids.num_boys,
        nkids.num_girls,
        nkids.num_kids
    FROM boxes b
            JOIN kid_profiles p ON b.kid_profile_id = p.id
            JOIN users u ON u.id = p.user_id
            JOIN spree_addresses a ON u.ship_address_id = a.id
            JOIN nkids ON nkids.user_id = u.id

    WHERE b.state NOT IN ('new_invalid', 'canceled', 'delivered', 'shipped', 'skipped', 'final')
        AND service_fee_amount = 5
        AND b.season_id = 9
        AND u.became_member_at IS NOT NULL
        AND b.created_at :: date > (current_date - interval '2 days') :: date
        AND u.email NOT ILIKE '%@rocketsofawesome.com';
""", slave)

b['days_to_convert'] = (b['became_member_at'] - b['created_at']).dt.days

c = pd.read_sql(
    """
    SELECT zip,
        unemploym_rate_civil,
        med_hh_income,
        married_couples_density,
        females_density,
        n_kids / area as kids_density,
        households_density,
        kids_school_perc,
        kids_priv_school_perc,
        smocapi_20,
        smocapi_25,
        smocapi_30,
        smocapi_35,
        smocapi_high,
        grapi_15,
        grapi_20,
        grapi_25,
        grapi_30,
        grapi_35,
        grapi_high
    FROM dwh.census
""", localdb)

d = pd.merge(b, c, how='left', left_on='zipcode', right_on='zip')
d.drop(['zip'], axis=1, inplace=True)

kids_list = '(' + ', '.join([str(x) for x in b['kid_id'].unique()]) + ')'

kps = query_kid_preferences(kids_list)

kps['color_count'] = kps.iloc[:, 1:11].notnull().sum(axis=1)
kps['blacklist_count'] = kps.iloc[:, 11:25].notnull().sum(axis=1)
kps['outfit_count'] = kps.iloc[:, 25:54].notnull().sum(axis=1)
kps['style_count'] = kps.iloc[:, 57:66].notnull().sum(axis=1)

kps['color_count'] = kps['color_count'].apply(lambda x: 1 if x > 0 else 0)
kps['blacklist_count'] = kps['blacklist_count'].apply(
    lambda x: 1 if x > 0 else 0)
kps['outfit_count'] = kps['outfit_count'].apply(lambda x: 1 if x > 0 else 0)
kps['style_count'] = kps['style_count'].apply(lambda x: 1 if x > 0 else 0)
kps['note_length'] = kps['note'].apply(lambda x: len(x) if x else 0)
kps['swim_count'] = kps['swim'].apply(lambda x: 1 if pd.notnull(x) else 0)
kps['neon_count'] = kps['neon'].apply(lambda x: 1 if pd.notnull(x) else 0)
kps['text_on_clothes_count'] = kps['text_on_clothes'].apply(
    lambda x: 1 if pd.notnull(x) else 0)
kps['backpack_count'] = kps['backpack'].apply(
    lambda x: 1 if pd.notnull(x) else 0)
kps['teams_count'] = kps['teams'].apply(lambda x: 1 if pd.notnull(x) else 0)

kps['n_preferences'] = kps.loc[:, [
    'color_count', 'blacklist_count', 'outfit_count', 'style_count',
    'swim_count', 'neon_count', 'text_on_clothes_count', 'backpack_count',
    'teams_count'
]].sum(axis=1)

kps.drop([
    'color_count', 'blacklist_count', 'outfit_count', 'style_count',
    'swim_count', 'neon_count', 'text_on_clothes_count', 'backpack_count',
    'teams_count'
],
         axis=1,
         inplace=True)

d = pd.merge(
    d, kps.loc[:, ['kid_id', 'note_length', 'n_preferences']], how='left')

users_list = '(' + ', '.join([str(x) for x in b['user_id'].unique()]) + ')'

cc = pd.read_sql_query(
    """
    SELECT user_id,
        cc_type,
        (make_date(cast(year as int), cast(month as int), 1) 
            + interval '1 month' - interval '1 day') :: date AS exp_date
    FROM spree_credit_cards
    WHERE "default" = TRUE
        AND user_id IN {users}
""".format(users=users_list), slave)

d = pd.merge(d, cc, how='left')
d['days_to_exp'] = (
    pd.to_datetime(d['exp_date']) - d['became_member_at']).dt.days

adds = pd.read_sql_query(
    """
    SELECT id as user_id,
        ship_address_id,
        bill_address_id
    FROM users
    WHERE id IN {users}
""".format(users=users_list), slave)
adds['diff_addresses'] = (
    adds['ship_address_id'] - adds['bill_address_id']).map(lambda x: x > 0)
adds.drop(['ship_address_id', 'bill_address_id'], axis=1, inplace=True)

d = pd.merge(d, adds, how='left')

uas = pd.read_sql_query(
    """
    SELECT id :: int as user_id,
        context_user_agent
    FROM javascript.users
    WHERE id ~ '^\d+$'
        AND context_user_agent IS NOT NULL
        AND CAST(id AS INT) in {users};
""".format(users=users_list), segment)

d = pd.merge(d, uas, how='left')
d['OS'] = d['context_user_agent'].apply(
    lambda x: up.ParseOS(x)['family'] if pd.notnull(x) else None)
d.loc[d['OS'] == 'Mac OS X', 'OS'] = 'Mac'
d.loc[d['OS'] == 'Linux', 'OS'] = 'Other'
d.loc[d['OS'] == 'Chrome OS', 'OS'] = 'Other'
d.drop('context_user_agent', axis=1, inplace=True)

funding = pd.read_sql_query(
    """
    SELECT ev.data__object__metadata__quark_user_id :: INT AS user_id,
        cc.funding
    FROM stitch_stripe.stripe_customers__cards__data cc
            JOIN stitch_stripe.stripe_events ev ON cc.customer = ev.data__object__customer
    WHERE ev.data__object__metadata__quark_user_id IS NOT NULL;
""", stitch)

d = pd.merge(d, funding, how='left')
d = d.loc[d['funding'] != 'unknown', :]
d = d.loc[(d['funding'].notnull()) & (d['OS'].notnull()), ]

df = d.loc[:, [
    'funding', 'OS', 'med_hh_income', 'kids_school_perc',
    'kids_priv_school_perc', 'smocapi_20', 'smocapi_25', 'smocapi_30',
    'smocapi_35', 'grapi_15', 'grapi_20', 'grapi_25', 'grapi_30', 'grapi_35',
    'num_kids', 'num_girls', 'days_to_convert', 'unemploym_rate_civil',
    'married_couples_density', 'n_preferences', 'cc_type', 'diff_addresses',
    'days_to_exp', 'note_length'
]]

imp = Imputer(strategy='median', axis=0, missing_values='NaN')

for col in df[[
        'med_hh_income', 'kids_school_perc', 'kids_priv_school_perc',
        'smocapi_20', 'smocapi_25', 'smocapi_30', 'smocapi_35', 'grapi_15',
        'grapi_20', 'grapi_25', 'grapi_30', 'grapi_35', 'num_kids',
        'days_to_convert', 'unemploym_rate_civil', 'married_couples_density'
]].columns:
    df[col] = imp.fit_transform((df[[col]]))

df.dropna(inplace=True)

rf = joblib.load("/Users/emmanuele/Code/ds_members/rf.pkl")
encoders = joblib.load("/Users/emmanuele/Code/ds_members/lab_encs.pkl")
enc = joblib.load("/Users/emmanuele/Code/ds_members/oh_enc.pkl")
poly = joblib.load("/Users/emmanuele/Code/ds_members/poly.pkl")
xcopy = df.copy()
for col in xcopy.columns:
    if xcopy[col].dtypes in ['object', 'bool']:
        xcopy[col] = encoders[col].transform(xcopy[col])
xenc = enc.transform(xcopy)
xenc = poly.transform(xenc)
preds = rf.predict(xenc)
probs = rf.predict_proba(xenc)

d['probs'] = [p[1] for p in probs]
d.sort_values(by='probs', ascending=False, inplace=True)

risky = d.loc[d['probs'] >= 0.4, ['user_id', 'num_kids', 'zipcode'
                                 ]].drop_duplicates()


def admin_link(user_id):
    link = 'https://admin.rocketsofawesome.com/customers/' + str(
        user_id) + '/edit'
    return link


def zip_link(zipcode):
    link = 'https://www.unitedstateszipcodes.org/' + str(zipcode) + '/'
    return link


risky['admin_link'] = risky['user_id'].apply(admin_link)
risky['zip_link'] = risky['zipcode'].apply(zip_link)

risky['created_at'] = pd.Timestamp.now()

s_id = '1AdUDx4-xDF9PvYCu-JpwTnkqs3wvWxrQ4GTdehGl-AU'
wks_name = 'risky queue'

t = g2d.download(s_id, wks_name, col_names=True, row_names=False)
t['user_id'] = t['user_id'].astype('int')
t['num_kids'] = t['num_kids'].astype('int')
t['created_at'] = pd.to_datetime(t['created_at'])
# t.drop('', axis=1, inplace=True)

risky = risky.loc[(risky['user_id'].isin(set(t['user_id'].unique()))) == False,]

n = pd.concat([t, risky], sort=False)
n = n.reset_index().drop('index', axis=1)
n['Notes'].fillna('', inplace=True)
n['Notes'].replace('None', '', inplace=True)

if n.shape[0] > t.shape[0]:
    risky.to_sql(
        "risky", localdb, schema='dwh', if_exists='append', index=False)
    d2g.upload(n, s_id, wks_name, clean=False, col_names=True, row_names=False)
