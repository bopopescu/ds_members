import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import psycopg2
from sqlalchemy import create_engine

sns.set_style('whitegrid')
pd.set_option('display.max_columns', 500)

stitch = create_engine(
    'postgresql://',
    echo=False,
    pool_recycle=300,
    echo_pool=True,
    creator=lambda _: psycopg2.connect(service='rockets-stitch'))

d = pd.read_sql_query("""
    WITH w AS (
        SELECT b.box_id,
            b.user_id,
            b.season_id,
            b.state,
            CASE WHEN bc.first_season < b.season_id THEN TRUE ELSE FALSE END AS repeat
        FROM dw.fact_boxes b
                JOIN dw.fact_user_box_count bc ON b.box_id = bc.box_id
                JOIN dw.fact_active_users u ON b.user_id = u.user_id
        WHERE b.state IN ('payment_failed', 'uncollected', 'final')
            AND (u.became_member_at < TIMESTAMP '2018-07-16' AT TIME ZONE 'America/New_York' OR u.became_member_at >= TIMESTAMP '2018-07-28' AT TIME ZONE 'America/New_York'))

    SELECT season_id,
        repeat,
        COUNT(*) AS count_all,
        count(DISTINCT case when state in ('payment_failed', 'uncollected') THEN box_id else null end) as count_fail,
        100.0 * count(DISTINCT case when state in ('payment_failed', 'uncollected') THEN box_id else null end) / count(distinct box_id) as pay_fail_rate
    FROM w
    group by 1, 2
""", stitch)

g = sns.FacetGrid(data=d, hue='repeat', height=8, aspect=1.4)
g.map(plt.bar, 'season_id', 'pay_fail_rate')

f = pd.read_sql_query("""
    SELECT season_id,
        COUNT(*)                                                           AS count_all,
        count(DISTINCT CASE WHEN b.state IN ('payment_failed', 'uncollected') THEN box_id ELSE NULL END)                                  AS count_fail,
        100.0 * count(DISTINCT CASE WHEN b.state IN ('payment_failed', 'uncollected') THEN box_id ELSE NULL END) / count(DISTINCT box_id) AS pay_fail_rate

    FROM dw.fact_boxes b
    join dw.fact_active_users u ON b.user_id = u.user_id
    WHERE b.state IN ('payment_failed', 'uncollected', 'final')
        AND (u.became_member_at < TIMESTAMP '2018-07-16' AT TIME ZONE 'America/New_York' OR u.became_member_at >= TIMESTAMP '2018-07-28' AT TIME ZONE 'America/New_York')

    GROUP BY 1
""", stitch)


s10 = pd.read_sql_query("""
    SELECT b.box_id,
        b.user_id,
        b.state,
        bc.first_season
    FROM dw.fact_boxes b
    JOIN dw.fact_user_box_count bc on b.box_id = bc.box_id
    WHERE b.state IN ('payment_failed', 'uncollected', 'final')
        AND b.season_id = 10
""", stitch)

plt.hist(s10.loc[s10['state'].isin(['payment_failed', 'uncollected']), 'first_season'])