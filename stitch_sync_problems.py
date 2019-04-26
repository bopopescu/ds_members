import psycopg2
import argparse
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy import String, Integer

pd.set_option('display.max_columns', 500)

redshift = create_engine(
    'postgresql://',
    echo=True,
    pool_recycle=300,
    echo_pool=True,
    creator=lambda _: psycopg2.connect(service='rockets-redshift'))

slave = create_engine(
    'postgresql://',
    echo=False,
    pool_recycle=300,
    echo_pool=True,
    creator=lambda _: psycopg2.connect(service='rockets-slave'))

# rboxes = pd.read_sql_query("""
#     SELECT id
#     FROM quark.boxes
#     WHERE season_id = 11
#         AND state = 'in_fulfillment'
#         -- AND updated_at <= GETDATE() - INTERVAL '7 days'
# """, redshift)


# sboxes = pd.read_sql_query(
#     """
#     SELECT id
#     FROM boxes
#     WHERE season_id = 11
#         AND state = 'in_fulfillment'
#         -- AND updated_at <= now() - INTERVAL '7 days'
# """, slave)

# ids = set(sboxes['id'].unique()) - set(rboxes['id'].unique())
# id_list = '(' + ', '.join([str(x) for x in ids]) + ')'

# inters = set(sboxes['id'].unique()).intersection(set(rboxes['id'].unique()))

# sdiffs = pd.read_sql_query("""
#     SELECT id,
#         state,
#         updated_at,
#         deleted_at
#     FROM boxes
#     WHERE id IN {id_list}
# """.format(id_list=id_list), slave)


# rdiffs = pd.read_sql_query(
#     """
#     SELECT id,
#         state,
#         updated_at,
#         deleted_at
#     FROM quark.boxes
#     WHERE id IN {id_list}
# """.format(id_list=id_list), redshift)


sboxstates = pd.read_sql_query(
    """
    SELECT state,
        count(*) AS count_slave,
        now() :: date as today
    FROM boxes
    WHERE season_id = 11
    GROUP BY 1, 3
""", slave)

rboxstates = pd.read_sql_query(
    """
    SELECT state,
        count(*) AS count_redshift,
        sysdate :: date as today
    FROM quark.boxes
    WHERE season_id = 11
    GROUP BY 1, 3
""", redshift)

d = rboxstates.merge(sboxstates, how='outer', on=['state', 'today'])
d.fillna(0, inplace=True)
d['count_redshift'] = d['count_redshift'].astype('int')
d['count_slave'] = d['count_slave'].astype('int')

d['delta_values'] = d['count_redshift'] - d['count_slave']


d.to_sql(
    'check_boxes', con=redshift, schema='qa', if_exists='append', index=False)
