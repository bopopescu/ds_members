import pandas as pd
from oauth2client.service_account import ServiceAccountCredentials
from df2gspread import gspread2df as g2d
from df2gspread import df2gspread as d2g
import psycopg2
from sqlalchemy import create_engine
import argparse
from datetime import datetime

stitch = create_engine(
    'postgresql://',
    echo=False,
    pool_recycle=300,
    echo_pool=True,
    creator=lambda _: psycopg2.connect(service='rockets-stitch'))

parser = argparse.ArgumentParser()
parser.add_argument(
    '-s',
    '--save',
    help='save the changes to the spreadsheet and db',
    action='store_true')

args = parser.parse_args()

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

u = '(' + ', '.join([str(x) for x in set(t['user_id'])]) + ')'

users = pd.read_sql_query(
    """
    SELECT user_id,
        max(user_box_rank) AS user_box_rank
    FROM dw.fact_user_box_count
    WHERE season_id = 10
        AND user_id IN {users}
    GROUP BY 1
""".format(users=u), stitch)

return_users = set(users.loc[users['user_box_rank'] > 1, 'user_id'])

w = pd.read_sql_query(
    """
    SELECT *
    FROM stitch_quark.whitelisted_users
    WHERE user_id IN {users}
""".format(users=u), stitch)

whitelist_users = set(w['user_id'])

d = datetime(2018, 10, 25, 20)
n = t.loc[(t['user_id'].isin(return_users) == False) |
          (t['created_at'] < d) |
          (t['user_id'].isin(whitelist_users) == False), ]

if args.save:
    n.to_sql(
        "pred_risky", stitch, schema='dw', if_exists='replace', index=False)
    d2g.upload(
        n,
        s_id,
        wks_name,
        clean=True,
        col_names=True,
        row_names=False,
        credentials=credentials)
