"""Which segment is falling in volume based on visitors' data?"""

import pandas as pd

import psycopg2
from sqlalchemy import create_engine

REDSHIFT = create_engine(
    'postgresql://',
    echo=True,
    pool_recycle=300,
    echo_pool=True,
    creator=lambda _: psycopg2.connect(service='rockets-redshift'))


fb_region = pd.read_sql_query("""
    SELECT region,
        sum(reach) AS reach,
        sum(impressions) AS impressions,
        sum(clicks) AS clicks
    FROM fb_ads.ads_insights_region
    WHERE date_start > '2019-02-05'
    GROUP BY 1
""", REDSHIFT)

c = pd.read_csv("/Users/emmanuele/Data/nst-est2018-alldata.csv", index_col=None)
c = c.iloc[5:, [4, 15]]

fb_region.loc[fb_region['region'] ==
              'Washington, District of Columbia', 'region'] = 'Washington'


df_region = pd.merge(fb_region, c, how='left', left_on='region', right_on='NAME')
df_region.dropna(inplace=True)
df_region['reach_idx'] = df_region['reach'] / df_region['POPESTIMATE2018']
df_region['impressions_idx'] = df_region['impressions'] / df_region['reach']
df_region['clicks_idx'] = df_region['clicks'] / df_region['impressions']


fb_age_gender = pd.read_sql_query(
    """
    SELECT date_start AS date,
        age,
        gender,
        reach,
        impressions,
        clicks
    FROM fb_ads.ads_insights_age_and_gender
""", REDSHIFT)

fb_age_gender['date'] = pd.to_datetime(fb_age_gender['date'])
