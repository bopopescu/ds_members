import psycopg2
import argparse
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

pd.set_option('display.max_columns', 500)

redshift = create_engine(
    'postgresql://',
    echo=False,
    pool_recycle=300,
    echo_pool=True,
    creator=lambda _: psycopg2.connect(service='rockets-redshift'))

slave = create_engine(
    'postgresql://',
    echo=False,
    pool_recycle=300,
    echo_pool=True,
    creator=lambda _: psycopg2.connect(service='rockets-slave'))

rboxes = pd.read_sql_query("""
    SELECT id
    FROM quark.boxes
    WHERE season_id = 11
        AND state = 'in_fulfillment'
""", redshift)


sboxes = pd.read_sql_query("""
    SELECT id
    FROM boxes
    WHERE season_id = 11
        AND state = 'in_fulfillment'
""", slave)
