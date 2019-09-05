import psycopg2
from sqlalchemy import create_engine

import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

import plotly
from fbprophet import Prophet

REDSHIFT = create_engine(
        'postgresql://',
        echo=True,
        pool_recycle=300,
        echo_pool=True,
        creator=lambda _: psycopg2.connect(service='rockets-redshift')
)


def mock_df(ts, start, end):
    dfs = []

    for cw in ts['style_number'].unique():



        d = pd.DataFrame({'shipped_on': pd.date_range(start, end), 'zeros': 0, 'cw': cw})
        dfs.append(d)


    return pd.concat(dfs)


def complete_time_series(ts):
    """[summary]
        
        Arguments:
            ts {[type]} -- [description]
        """
    z = mock_df(
            ts,
            pd.to_datetime(ts['shipped_on']).min(),
            pd.to_datetime(ts['shipped_on']).max()
    )
    g = pd.merge(z, ts, how='left')
    g.fillna(0, inplace=True)
    g.drop('zeros', axis=1, inplace=True)

    return g




# backpacks
# d = pd.read_sql_query("""
#     SELECT fi.sku,
#         fi.order_id,
#         fi.state,
#         fo.shipped_at :: date AS shipped_on,
#         fi.quantity
#     FROM dw.fact_items fi
#             JOIN dw.fact_orders fo ON fi.order_id = fo.id
#     WHERE (reception_status IS NULL OR reception_status = 'expired')
#     AND fi.state = 'shipped'
#     AND sku IN ('A234-C01-A', 'A235-F02-A')
# """, REDSHIFT)

# b = d.groupby('shipped_on')['quantity']\
#         .sum()\
#         .reset_index()\
#         .sort_values(by='shipped_on')

# b['cumsum'] = b['quantity'].cumsum()
# b.plot('shipped_on', 'cumsum')


# best sellers
bs = pd.read_sql_query("""
        SELECT dc.style_number || '-' || dc.color_code AS cw,
        sum(fi.quantity)      AS sold
        FROM dw.fact_items fi
                JOIN dw.fact_orders fo ON fi.order_id = fo.id
                JOIN dw.dim_canon dc ON fi.sku = dc.sku
        WHERE fi.state = 'shipped'
                AND fo.season_id = 13
                AND (fi.reception_status IS NULL OR fi.reception_status = 'expired')
        GROUP BY 1
        ORDER BY 2 DESC
        LIMIT 10
""", REDSHIFT)



# time series
ts = pd.read_sql_query("""
        -- SELECT dc.style_number || '-' || dc.color_code AS cw,
        SELECT dc.style_number,
                fo.shipped_at :: date AS shipped_on,
                sum(fi.quantity)      AS sold
        FROM dw.fact_items fi
                JOIN dw.fact_orders fo ON fi.order_id = fo.id
                JOIN dw.dim_canon dc ON fi.sku = dc.sku
        WHERE fi.state = 'shipped'
                -- AND fo.season_id = 13
                AND (fi.reception_status IS NULL OR fi.reception_status = 'expired')
        GROUP BY 1, 2
        ORDER BY 1, shipped_on
""", REDSHIFT)

ts['shipped_on'] = pd.to_datetime(ts['shipped_on'])

# rainbow = ts.loc[ts['cw'] == '5699-K09', ]
# r = complete_time_series(rainbow)
# r['cumsum'] = r['sold'].cumsum()

bomber = ts.loc[ts['style_number'].str.contains('2010'), ]
b = complete_time_series(bomber)
b['cumsum'] = b['sold'].cumsum()

train = b.loc[b['shipped_on'] < '2018-09-01', ['shipped_on', 'cumsum']]
train.rename(columns={'shipped_on': 'ds', 'cumsum': 'y'}, inplace=True)


m = Prophet()
m.fit(train)
future = m.make_future_dataframe(periods=365)
forecast = m.predict(future)

forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
fig1 = m.plot(forecast)
fig2 = m.plot_components(forecast)


def ts_plot(data, var='cumsum', title=None, hue=None, palette=None, style=None):
    """Time series plot
        
        Arguments:
            data {dataframe} -- time series
        
        Keyword Arguments:
            var {str} -- variable to plot (default: {'cumsum'})
            title {str} -- title (default: {None})
            hue {str} -- hue column (default: {None})
            palette {str} -- seaborn palette (default: {None})
            style {[type]} -- [description] (default: {None})
        """

    fig, ax = plt.subplots(figsize=(12, 7))

    ax = sns.lineplot(
            x="shipped_on", y=var, hue=hue, style=style, palette=palette, data=data, ax=ax)
    ax.figure.autofmt_xdate()
    ax.set_title(title)

    # return fig
