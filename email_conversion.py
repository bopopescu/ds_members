import psycopg2
import argparse
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from matplotlib import pyplot as plt

redshift = create_engine(
    'postgresql://',
    echo=False,
    pool_recycle=300,
    echo_pool=True,
    creator=lambda _: psycopg2.connect(service='rockets-redshift'))

parser = argparse.ArgumentParser()
parser.add_argument(
    '-s',
    '--shop',
    help='plot the email to shop conversion',
    action='store_true')

parser.add_argument(
    '-b',
    '--box',
    help='plot the email to became member conversion',
    action='store_true'
)

args = parser.parse_args()


if args.shop:

    em = pd.read_sql_query("""
        WITH all_emails AS (

            SELECT d.__sdc_primary_key,
                flso.id,
                d.datafields__email,
                d.datafields__createdat,
                flso.completed_at,
                flso.email
            FROM dw.fact_latest_state_orders flso
                    JOIN stitch_iterable.data d
                        ON flso.email = d.datafields__email AND flso.completed_at > d.datafields__createdat
            WHERE flso.state = 'complete'
                AND order_type = 'shop'
                AND d.eventname = 'emailOpen'

            ),

            last_emails AS (

                SELECT id,
                    MAX(datafields__createdat) AS last_ts
                from all_emails
                GROUP BY 1

            ),

        select_last_email AS (

            SELECT all_emails.*
            FROM all_emails
            JOIN last_emails ON all_emails.id = last_emails.id and all_emails.datafields__createdat = last_emails.last_ts

        ),

        first_order AS (
            SELECT __sdc_primary_key,
                min(completed_at) AS first_ts
            from select_last_email
            group by 1
        )

        SELECT select_last_email.*
        FROM select_last_email
        JOIN first_order ON select_last_email.__sdc_primary_key = first_order.__sdc_primary_key and select_last_email.completed_at = first_order.first_ts
    """, redshift)

    em['datafields__createdat'] = pd.to_datetime(em['datafields__createdat'])
    em['days_diff'] = (em['completed_at'].dt.date - em['datafields__createdat'].dt.date).dt.days


    plt.figure(figsize=(10, 6))
    plt.title('Days between email open and ecomm order')
    plt.xlabel('Days')
    em['days_diff'].hist(bins=np.arange(0, 50))
    plt.savefig('email_convert_days.jpg', bbox_inches='tight')


if args.box:

    dm = pd.read_sql_query("""
        WITH all_emails AS (

            SELECT d.__sdc_primary_key,
                dm.user_id,
                d.datafields__email,
                d.datafields__createdat,
                dm.became_member_at
            FROM dw.dim_members dm
                    JOIN stitch_iterable.data d
                        ON dm.email = d.datafields__email AND dm.became_member_at > d.datafields__createdat

            ),

            last_emails AS (

                SELECT user_id,
                    MAX(datafields__createdat) AS last_ts
                from all_emails
                GROUP BY 1

            ),

        select_last_email AS (

            SELECT all_emails.*
            FROM all_emails
            JOIN last_emails ON all_emails.user_id = last_emails.user_id and all_emails.datafields__createdat = last_emails.last_ts

        )


        SELECT select_last_email.*
        FROM select_last_email
    """, redshift)

    dm['datafields__createdat'] = pd.to_datetime(dm['datafields__createdat'])
    dm['days_diff'] = (dm['became_member_at'].dt.date - dm['datafields__createdat'].dt.date).dt.days


    plt.figure(figsize=(10, 6))
    plt.title('Days between email open and ecomm order')
    plt.xlabel('Days')
    dm['days_diff'].hist(bins=np.arange(0, 100))
    plt.savefig('email_convert_box_days.jpg', bbox_inches='tight')
