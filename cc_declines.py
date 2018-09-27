import psycopg2
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

sns.set_style('whitegrid')
stitch = psycopg2.connect(service='rockets-stitch')

d = pd.read_sql_query(
    """
    SELECT data__object__failure_message as reason,
        data__object__metadata__quark_user_id :: int as user_id
    FROM stitch_stripe.stripe_events
    WHERE data__object__metadata__quark_user_id IS NOT NULL
    and type = 'charge.failed';
""", stitch)

dd = pd.read_sql_query(
    """
    SELECT count(*),
        data__object__failure_message reason,
        funding
    FROM stitch_stripe.stripe_events ev
            JOIN stitch_stripe.stripe_customers__cards__data cc ON cc.customer = ev.data__object__customer
    WHERE type = 'charge.failed'
    AND ev.data__object__metadata__quark_user_id IS NOT NULL
    GROUP BY 2, 3;
""", stitch)

# f, ax = plt.subplots(figsize=(14, 10))
g = sns.FacetGrid(
    data=dd, col='funding', col_wrap=2, palette='Set2', aspect=1.3, height=4, sharex=False)
g = g.map(
    sns.barplot,
    'count',
    'reason',
    order=d['reason'].value_counts().index,
    orient='h')

# f, ax = plt.subplots(figsize=(14, 10))
# sns.countplot(
#     y='reason',
#     data=d,
#     order=d['reason'].value_counts().index,
#     orient='h',
#     ax=ax)
# f.savefig(
#     "cc_failures_reasons.jpg", transparent=False, dpi=80, bbox_inches='tight')

# t = pd.DataFrame(d.groupby('user_id')['reason'].count()).rename(columns={
#     'reason': 'count'
# }).reset_index()

# f, ax = plt.subplots(figsize=(14, 10))
# sns.countplot(
#     y='count',
#     data=t,
#     order=t['count'].value_counts().index[:10],
#     orient='h',
#     ax=ax)
# ax.set_ylabel("number of cc failures per user")
# ax.set_xlabel("number of users")
# f.savefig(
#     "cc_failures_per_user.jpg", transparent=False, dpi=80, bbox_inches='tight')
