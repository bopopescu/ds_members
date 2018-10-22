import argparse
import psycopg2
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from oauth2client.service_account import ServiceAccountCredentials
from df2gspread import gspread2df as g2d
from df2gspread import df2gspread as d2g
import yaml

from sklearn.preprocessing import Imputer
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.externals import joblib
from ua_parser import user_agent_parser as up

pd.set_option('display.max_columns', 100)

parser = argparse.ArgumentParser()
parser.add_argument(
    '-l',
    '--local',
    help='run the prediction locally, to test',
    action='store_true')

args = parser.parse_args()

if args.local:
    stitch = create_engine(
        'postgresql://',
        echo=False,
        pool_recycle=300,
        echo_pool=True,
        creator=lambda _: psycopg2.connect(service='rockets-stitch'))

else:
    with open('/home/ec2-user/Configs/Shoplets/config_prod.yaml') as c:
        conf = yaml.load(c)['rockets-stitch']

    conn = psycopg2.connect(
        dbname=conf['dbname'],
        user=conf['user'],
        password=conf['password'],
        host=conf['dbhost'],
        port=5432,
        sslmode='require')

    stitch = create_engine(
        'postgresql://',
        echo=False,
        pool_recycle=300,
        echo_pool=True,
        creator=lambda _: conn)

u = pd.read_sql_query(
    """
        SELECT DISTINCT bc.user_id,
                        bc.state,
                        ch.channel,
                        u.num_boys,
                        u.num_girls,
                        u.num_kids,
                        u.service_fee_enabled,
                        CASE WHEN ship_address_id != bill_address_id THEN TRUE ELSE FALSE END AS diff_addresses,
                        date_part('days', u.became_member_at - u.created_at)                  AS days_to_convert,
                        unemploym_rate_civil,
                        med_hh_income,
                        married_couples_density,
                        females_density,
                        n_kids / area                                                         AS kids_density,
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
                        grapi_high,
                        left(u.zipcode, 5)                                             AS zipcode
        FROM dw.fact_user_box_count bc
                LEFT JOIN dw.fact_channel_attribution ch ON ch.user_id = bc.user_id
                JOIN dw.fact_active_users u ON bc.user_id = u.user_id
                JOIN stitch_quark.spree_addresses a ON u.ship_address_id = a.id
                JOIN dw.dim_census c ON left(a.zipcode, 5) = c.zip
                JOIN dw.fact_boxes b ON bc.box_id = b.box_id
        WHERE bc.season_id = 10
        AND user_box_rank = 1
        AND bc.state NOT IN ('new_invalid', 'canceled', 'delivered', 'shipped', 'in_fulfillment', 'skipped', 'final')
        AND (b.shipping_window_id IN (SELECT current_shipping_window_id FROM dw.dim_shipping_windows WHERE current_window
                                        UNION (SELECT next_shipping_window_id
                                            FROM dw.dim_shipping_windows
                                            WHERE current_window)) OR b.shipping_window_id IS NULL)
        AND u.email NOT ILIKE '%%@rocketsofawesome.com';
""", stitch)


# Cleanup of multiple entries
u_counts = u.groupby('user_id')['user_id'].count().sort_values(
    ascending=False)
idx = u_counts[u_counts == 2].index
mult_u = u.loc[u['user_id'].isin(idx),]
single_u = u.loc[u['user_id'].isin(idx) == False,]
single_u.drop('state', axis=1, inplace=True)

mult_u.drop(['state'], axis=1, inplace=True)
mult_u.drop_duplicates(inplace=True)
users = pd.concat([single_u, mult_u], axis=0)

users_list = '(' + ', '.join([str(x) for x in users['user_id'].unique()
                                ]) + ')'

cc = pd.read_sql_query(
    """
    SELECT u.user_id,
        cc.cc_type,
        ((make_date(cast(year AS int), cast(month AS int), 1)
                + INTERVAL '1 month' - INTERVAL '1 day') :: DATE - u.became_member_at :: date) / 30 AS months_to_exp,
        ccd.funding
    FROM stitch_quark.spree_credit_cards cc
            JOIN dw.fact_active_users u ON cc.user_id :: BIGINT = u.user_id
            join stitch_stripe.stripe_customers__cards__data ccd on cc.gateway_customer_profile_id = ccd.customer
    WHERE "default" = TRUE
    AND cc.user_id :: BIGINT IN {users}
""".format(users=users_list), stitch)

d = pd.merge(users, cc)

uas = pd.read_sql_query(
    """
    SELECT user_id,
        context_user_agent
    FROM dw.fact_first_click_first_pass
    WHERE user_id in {users}
""".format(users=users_list), stitch)

d = pd.merge(d, uas, how='left')

d.dropna(inplace=True)

d['OS'] = d['context_user_agent'].apply(
    lambda x: up.ParseOS(x)['family'] if pd.notnull(x) else None)
d.loc[d['OS'] == 'Mac OS X', 'OS'] = 'Mac'
d.loc[d['OS'].isin(['Mac', 'Windows', 'iOS', 'Android']) ==
        False, 'OS'] = 'Other'
d.drop('context_user_agent', axis=1, inplace=True)

df = d.loc[:, [
    'user_id', 'funding', 'OS', 'service_fee_enabled', 'med_hh_income',
    'kids_school_perc', 'kids_priv_school_perc', 'smocapi_20', 'smocapi_25',
    'smocapi_30', 'smocapi_35', 'grapi_15', 'grapi_20', 'grapi_25', 'grapi_30',
    'grapi_35', 'num_kids', 'num_girls', 'days_to_convert',
    'unemploym_rate_civil', 'married_couples_density', 'cc_type',
    'diff_addresses', 'months_to_exp', 'channel'
]]

df.dropna(inplace=True)

if args.local:
    xgb = joblib.load('xgb.pkl')
    encoders = joblib.load("lab_encs_xgb.pkl")
    enc = joblib.load("oh_enc_xgb.pkl")

else:
    xgb = joblib.load('/home/ec2-user/ds_members/xgb.pkl')
    encoders = joblib.load("/home/ec2-user/ds_members/lab_encs_xgb.pkl")
    enc = joblib.load("/home/ec2-user/ds_members/oh_enc_xgb.pkl")

xcopy = df.copy()
xcopy.drop('user_id', axis=1, inplace=True)
xcopy.loc[xcopy['OS'].isin(['iOS', 'Windows', 'Android', 'Mac', 'Other']) ==
          False, 'OS'] = 'Other'
for col in xcopy.columns:
    if xcopy[col].dtypes in ['object', 'bool']:
        xcopy[col] = encoders[col].transform(xcopy[col])
xenc = enc.transform(xcopy)
preds = xgb.predict(xenc)
probs = xgb.predict_proba(xenc)

d['probs'] = [p[1] for p in probs]
d.sort_values(by='probs', ascending=False, inplace=True)

risky = d.loc[d['probs'] >= 0.4, [
    'user_id',
    'num_kids',
    'zipcode',
    'probs'
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

scope = [
    'https://spreadsheets.google.com/feeds',
    'https://www.googleapis.com/auth/drive'
]
credentials = ServiceAccountCredentials.from_json_keyfile_name(
    'risky-users-e25b80bbf7bb.json', scope)

t = g2d.download(
    s_id, wks_name, col_names=True, row_names=False, credentials=credentials)
t['user_id'] = t['user_id'].astype('int')
t['num_kids'] = t['num_kids'].astype('float').astype('int')
t['created_at'] = pd.to_datetime(t['created_at'])

risky = risky.loc[(risky['user_id'].isin(set(t['user_id'].unique()))) == False,]

n = pd.concat([t, risky], sort=False)
n = n.reset_index().drop('index', axis=1)
n['Notes'].fillna('', inplace=True)
n['Notes'].replace('None', '', inplace=True)

if n.shape[0] > t.shape[0] and args.local == False:
    n.to_sql("pred_risky", stitch, schema='dw', if_exists='append', index=False)
    d2g.upload(
        n,
        s_id,
        wks_name,
        clean=False,
        col_names=True,
        row_names=False,
        credentials=credentials)
