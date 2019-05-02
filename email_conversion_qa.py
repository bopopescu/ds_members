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

daily = pd.read_sql_query(
    """
    SELECT sysdate AS now,
        fewso.*
    FROM dw.fact_emails_with_shop_orders fewso
    WHERE datafields__createdat > '2019-04-01'
""", redshift)

daily.rename(columns={'message_id': '__sdc_primary_key'}, inplace=True)
daily.drop('time_delta', axis=1, inplace=True)

daily.to_sql(
    'test_email_conversion',
    con=redshift,
    schema='qa',
    if_exists='append',
    index=False)

daily['completed_date'] = daily['completed_at'].dt.date
daily_summary = daily.groupby([
    'now', 'completed_date'
])['order_id'].count().reset_index().rename(columns={'order_id': 'n_orders'})

daily_summary.to_sql(
    'test_email_conversion_summary',
    con=redshift,
    schema='qa',
    if_exists='append',
    index=False)

d = pd.read_sql_query(
    """
    SELECT order_id,
        completed_at :: date
    FROM qa.test_email_conversion
    WHERE now :: date = sysdate :: date
    AND completed_at :: date < sysdate :: date

    EXCEPT

    SELECT order_id,
        completed_at :: date
    FROM qa.test_email_conversion
    WHERE now :: date = (sysdate - INTERVAL  '1 day') :: date
        AND completed_at :: date < sysdate  :: date
""", redshift)

s = pd.read_sql_query(
    """
    SELECT 
        sysdate AS now,
        flso.*
    FROM dw.fact_latest_state_orders flso
    WHERE order_type = 'shop'
    AND state = 'complete'
    AND completed_at > '2019-04-15'
""", redshift)

s.to_sql(
    'test_email_conversion_orders',
    con=redshift,
    schema='qa',
    if_exists='append',
    index=False
)