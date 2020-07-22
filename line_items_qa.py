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

subordinate = create_engine(
    'postgresql://',
    echo=True,
    pool_recycle=300,
    echo_pool=True,
    creator=lambda _: psycopg2.connect(service='rockets-subordinate')
)

sli_redshift = pd.read_sql_query("""
    SELECT sli.id,
        sli.order_id,
        sli.quantity,
        sli.variant_id,
        sv.sku,
        TRUE AS redshift
    FROM quark.spree_line_items sli
    JOIN quark.spree_variants sv ON sli.variant_id = sv.id
""", redshift)

sli_db = pd.read_sql_query("""
    SELECT
        sli.order_id,
        sli.quantity,
        sli.variant_id,
        sv.sku,
        TRUE AS db
    FROM spree_line_items sli
    JOIN spree_variants sv ON sli.variant_id = sv.id
""", subordinate)


d = pd.merge(sli_redshift, sli_db, how='left', on=['order_id', 'quantity', 'variant_id', 'sku'])
df = d.loc[d['db'].isnull(), ['id', 'order_id', 'variant_id', 'sku']].reset_index().drop('index', axis=1)

df.to_sql(
    'spree_line_items_hard_deletes',
    con=redshift,
    schema='qa',
    if_exists='replace',
    index=False,
    chunksize=1000
)