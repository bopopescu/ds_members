"""
Plotting box and sku values for items kept and not kept.
"""

import psycopg2
import pandas as pd
import numpy as np
from uncertainties import ufloat
from sqlalchemy import create_engine
from matplotlib import pyplot as plt
import seaborn as sns
from plot_utils import binned_df

sns.set_style('whitegrid')

slave = psycopg2.connect(service="rockets-slave")

localdb = create_engine(
    'postgresql://',
    echo=False,
    pool_recycle=300,
    creator=lambda _: psycopg2.connect(service="rockets-local"))


def calc_ratio_error(row):
    """Calculate the error bar on a ratio.
    
    Arguments:
        row {pandas row} --
    """
    t = ufloat(row['total'], row['s_total'])
    k = ufloat(row['ka'], row['s_ka'])
    r = k / t if t > 0 else ufloat(0, 0)

    return r.std_dev


b = pd.read_sql_query(
    """
    SELECT SUM(
            CASE
                WHEN ri.id IS NULL OR ri.reception_status = 'expired' THEN 1
                ELSE 0 END)   AS items_kept,
        SUM(sp.amount)        AS gmv,
        o.payment_total       AS revenue,
        b.id                  AS box_id,
        b.kid_profile_id      AS kid_id,
        CASE
            WHEN u.service_fee_enabled IS TRUE THEN b.service_fee_amount
            ELSE 0 END        AS fee,
        b.approved_at :: date AS box_date
    FROM boxes b
            JOIN spree_orders o ON b.order_id = o.id
            JOIN users u ON o.user_id = u.id
            JOIN spree_line_items si ON si.order_id = o.id
            LEFT JOIN spree_inventory_units iu ON iu.line_item_id = si.id
            LEFT JOIN spree_return_items ri ON ri.inventory_unit_id = iu.id
            LEFT JOIN spree_variants v ON v.id = si.variant_id
            LEFT JOIN spree_prices sp ON v.id = sp.variant_id
    WHERE b.state = 'final'  -- otherwise kept is meaningless
    AND v.sku <> 'X001-K09-A'
    AND b.approved_at < (CURRENT_DATE - INTERVAL '2 weeks') :: date
    AND b.approved_at > '2018-01-01'
    GROUP BY 3, 4, 5, 6, 7;
""", slave)
b['box_date'] = pd.to_datetime(b['box_date'])

s = pd.read_sql_query(
    """
SELECT CASE
	       WHEN ri.id IS NULL OR ri.reception_status = 'expired' THEN TRUE
	       ELSE FALSE END    AS kept,
       sp.amount             AS gmv,
       b.id                  AS box_id,
       b.kid_profile_id      AS kid_id,
       CASE
	       WHEN u.service_fee_enabled IS TRUE THEN b.service_fee_amount
	       ELSE 0 END        AS fee,
       b.approved_at :: date AS box_date
FROM boxes b
	     JOIN spree_orders o ON b.order_id = o.id
	     JOIN users u ON o.user_id = u.id
	     LEFT JOIN spree_line_items si ON si.order_id = o.id
	     LEFT JOIN spree_inventory_units iu ON iu.line_item_id = si.id
	     LEFT JOIN spree_return_items ri ON ri.inventory_unit_id = iu.id
	     LEFT JOIN spree_variants v ON v.id = si.variant_id
	     LEFT JOIN spree_prices sp ON v.id = sp.variant_id
WHERE b.state = 'final'  -- otherwise kept is meaningless
  AND v.sku <> 'X001-K09-A'
  AND b.approved_at < (CURRENT_DATE - INTERVAL '2 weeks') :: date
  AND b.approved_at > '2018-01-01'
order by b.id;
""", slave)
s['box_date'] = pd.to_datetime(s['box_date'])

ka = s.groupby('box_id')['kept'].sum().reset_index().rename(
    columns={'kept': 'n_kept'})

nine_items = set(ka.loc[ka['n_kept'] == 9, 'box_id'].unique())
ka = ka.loc[(ka['box_id'].isin(nine_items)) == False,]
s = pd.merge(s, ka)
s['keep_all'] = s['n_kept'].apply(lambda x: True if x == 8. else False)
s.drop('n_kept', axis=1, inplace=True)

b = b.loc[(b['box_id'].isin(nine_items)) == False,]
b['keep_all'] = b['items_kept'].apply(lambda x: True if x == 8 else False)

# remove few outliers
t = b.loc[b['gmv'] < 300,]
print(t['gmv'].median())
print(t.loc[t['keep_all'] == True, 'gmv'].median())
print(t.loc[t['keep_all'] == False, 'gmv'].median())

# by box
kws = dict(linewidth=.5, edgecolor="w", alpha=0.5)
g = sns.FacetGrid(b, hue="keep_all", palette='Set1', aspect=1.3, height=4)
g = g.map(
    plt.hist, 'gmv', bins=range(150, 300, 10), density=False,
    **kws).set_xlabels("Dollar Price")
g.fig.suptitle("Box total price")
# g.fig.savefig("box_price.pdf", transparent=False, dpi=80, bbox_inches='tight')

# non keep-all boxes
n = s.loc[s['keep_all'] == False,]
g = sns.FacetGrid(n, hue='kept', palette="Set3", aspect=1.3, height=4)
g = g.map(
    plt.hist, 'gmv', bins=range(10, 50, 2), density=False,
    **kws).set_xlabels("Dollar Price")
g.fig.suptitle("SKU price for non keep-all boxes")
# g.fig.savefig("sku_price.pdf", transparent=False, bbox_inches='tight')

# by box ratio
bbinn_all = binned_df(b, 'gmv', 150, 300)
bbinn_all.rename(columns={'gmv': 'total'}, inplace=True)

bbinn_ka = binned_df(b.loc[b['keep_all'],], 'gmv', 150, 300)
bbinn_ka.rename(columns={'gmv': 'ka'}, inplace=True)
bbinn_ka.drop('gmv_mid', axis=1, inplace=True)

bbinn = pd.merge(bbinn_all, bbinn_ka, left_index=True, right_index=True)
bbinn['ratio'] = bbinn['ka'] / bbinn['total']

bbinn['s_total'] = np.sqrt(bbinn['total'])
bbinn['s_ka'] = np.sqrt(bbinn['ka'])
bbinn.fillna(0, inplace=True)
bbinn['s_ratio'] = bbinn.apply(calc_ratio_error, axis=1)

bbinn.sort_values(by='gmv_mid', inplace=True)

fig, ax = plt.subplots(nrows=1, ncols=1)
ax.errorbar(
    bbinn['gmv_mid'],
    bbinn['ratio'],
    yerr=bbinn['s_ratio'],
    fmt='o',
    color='C1')
ax.set_xlabel('Box Price')
ax.set_ylim(0, 1)
ax.set_xlim(150, 310)
ax.set_title("Keep All rate V box price")
ax.hlines(0.5, 150, 310, color='C5')
fig.savefig('box_price_ratio.pdf', bbox_inches='tight')
fig.savefig('box_price_ratio.jpg', bbox_inches='tight')

# non-keep all items ratio
n_all = binned_df(n, 'gmv', 10, 50)
n_all.rename(columns={'gmv': 'total'}, inplace=True)
n_ka = binned_df(n.loc[n['kept'],], 'gmv', 10, 50)
n_ka.rename(columns={'gmv': 'ka'}, inplace=True)
n_ka.drop('gmv_mid', axis=1, inplace=True)

nn = pd.merge(n_all, n_ka, left_index=True, right_index=True)
nn['ratio'] = nn['ka'] / nn['total']

nn['s_total'] = np.sqrt(nn['total'])
nn['s_ka'] = np.sqrt(nn['ka'])
nn.fillna(0, inplace=True)
nn['s_ratio'] = nn.apply(calc_ratio_error, axis=1)

nn.sort_values(by='gmv_mid', inplace=True)

fig, ax = plt.subplots(nrows=1, ncols=1)
ax.errorbar(
    nn['gmv_mid'],
    nn['ratio'],
    yerr=nn['s_ratio'],
    fmt='o',
    color='C1')
ax.set_xlabel('SKU Price')
ax.set_ylim(0, 1)
ax.set_xlim(10, 50)
ax.set_title("Keep Rate V sku price")
ax.hlines(0.5, 10, 50, color='C5')
fig.savefig('sku_price_ratio.pdf', bbox_inches='tight')
fig.savefig('sku_price_ratio.jpg', bbox_inches='tight')
