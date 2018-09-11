from sqlalchemy import create_engine
import psycopg2
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', 50)
sns.set_style('whitegrid')
localdb = create_engine(
    'postgresql://',
    echo=False,
    pool_recycle=300,
    creator=lambda _: psycopg2.connect(service="rockets-local"))


def plotVar(data, var):
    """Plot 1D histograms of categorical features

    Arguments:
        data {pd.DataFrame} -- Dataframe
        var {str} -- feature name

    Returns:
        tuple(fig, axis) -- matplotlib fig, ax objects
    """
    fig, ax = plt.subplots(figsize=(16, 10))
    sns.countplot(
        x=var,
        data=data,
        order=data[var].value_counts().iloc[:30, ].index,
        ax=ax)

    labels = ax.get_xticklabels()
    ax.set_xticklabels(labels, rotation=90, fontsize=18)
    ax.set_xlabel(var, fontsize=18)
    ax.set_ylabel('count', fontsize=16)
    ax.set_title(var, fontsize=20)

    return fig, ax


def plot_categorical(data, var, label='is_good_customer', yscale=None, q=1.):
    """1D plot of categorical data

    Makes two plots: the top one is a stacked hist of good V bad customers.
    The bottom one is the ratio of True V total customers per bin.

    Arguments:
        data {DataFrame} --
        var {str} -- var name

    Keyword Arguments:
        yscale {[type]} -- [description] (default: {None})
        q {number} -- [description] (default: {1.})
    """
    f, (ax1, ax2) = plt.subplots(
        2, 1, sharex='col', sharey=False, figsize=(14, 9))

    t = data.groupby([var,
                      label])['user_id'].count().reset_index()
    t = t.pivot(index=var, columns=label, values='user_id')
    t.fillna(0, inplace=True)
    t['tot'] = t[True] + t[False]
    for col in t.columns:
        t[col] = t[col].astype('int')
    t['ratio'] = t[True] / t['tot']
    t.sort_values(by='tot', ascending=False, inplace=True)
    qt = t.loc[t['tot'].cumsum() <= q * t['tot'].sum(), ]
    barWidth = 1

    r = qt.index.values
    print(r)
    bars1 = qt[True].values
    bars2 = qt[False].values

    ax1.bar(
        r,
        bars1,
        color='#557f2d',
        edgecolor='white',
        width=barWidth,
        label='Good Customer')
    ax1.bar(
        r,
        bars2,
        bottom=bars1,
        color='#7f6d5f',
        edgecolor='white',
        width=barWidth,
        label='Bad Customer')

    ax2.plot(
        r, qt['ratio'].values, marker='o', linestyle='None', color='#557f2d')
    ax1.legend()
    plt.xticks(rotation=90)
    ax2.set_title("Good Customers Fraction")
    ax2.set_ylim(bottom=0, top=1)
    if yscale:
        ax1.set(yscale='log')

    plt.ion()
    plt.show()


def plot_integer(data, var, yscale=None):
    g = sns.FacetGrid(
        data,
        row='is_good_customer',
        height=3.5,
        aspect=2.5,
        sharey=False,
        sharex=False)
    g.map(plt.hist, var).add_legend()

    return g


def split_quantile(data, var, q=.9):
    h_lo = \
        data.loc[data[var] <= data[var].quantile(q), [var, 'is_good_customer']]
    h_hi = \
        data.loc[data[var] > data[var].quantile(q), [var, 'is_good_customer']]
    return h_lo, h_hi


def binned_df(data):

    var = data.columns[0]

    bin_size = (data[var].max() - data[var].min()) / 20
    bins_range = np.arange(data[var].min(), data[var].max() + bin_size,
                           bin_size)

    binned_true = pd.cut(data.loc[data['is_good_customer'] == True, var],
                         bins_range)
    binned_true.dropna(inplace=True)
    binned_false = pd.cut(data.loc[data['is_good_customer'] == False, var],
                          bins_range)
    binned_false.dropna(inplace=True)
    binned_tot = pd.cut(data[var], bins_range)
    binned_tot.dropna(inplace=True)

    h_true = pd.value_counts(binned_true)
    h_false = pd.value_counts(binned_false)
    h_tot = pd.value_counts(binned_tot)
    h = pd.DataFrame(pd.concat([h_true, h_false, h_tot], axis=1))
    h.columns = ['true', 'false', 'tot']
    h['ratio'] = h['true'] / h['tot']
    h[str(var) + '_mid'] = [point.mid for point in h.index]

    return h


def plot_floating_split(data, var, quantile):
    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
        2, 2, sharex='col', sharey=False, figsize=(14, 8))

    h_lo, h_hi = split_quantile(data, var, quantile)
    df_lo = binned_df(h_lo)
    df_hi = binned_df(h_hi)

    rlo = df_lo[str(var) + '_mid'].values
    barWidth_lo = (rlo.max() - rlo.min()) / 20
    bars1 = df_lo['true'].values
    bars2 = df_lo['false'].values

    ax1.bar(
        rlo,
        bars1,
        color='#557f2d',
        edgecolor='white',
        width=barWidth_lo,
        label='Good Customer')
    ax1.bar(
        rlo,
        bars2,
        bottom=bars1,
        color='#7f6d5f',
        edgecolor='white',
        width=barWidth_lo,
        label='Bad Customer')
    ax3.plot(
        rlo,
        df_lo['ratio'].values,
        marker='o',
        linestyle='None',
        color='#557f2d')
    ax1.legend()
    ax3.set_title("Good Customers Fraction")
    ax3.set_ylim(bottom=0, top=1)

    rhi = df_hi[str(var) + '_mid'].values
    barWidth_hi = (rhi.max() - rhi.min()) / 20
    bars3 = df_hi['true'].values
    bars4 = df_hi['false'].values

    ax2.bar(
        rhi,
        bars3,
        color='#557f2d',
        edgecolor='white',
        width=barWidth_hi,
        label='Good Customer')
    ax2.bar(
        rhi,
        bars4,
        bottom=bars3,
        color='#7f6d5f',
        edgecolor='white',
        width=barWidth_hi,
        label='Bad Customer')
    ax4.plot(
        rhi,
        df_hi['ratio'].values,
        marker='o',
        linestyle='None',
        color='#557f2d')
    ax2.legend()
    ax4.set_title("Good Customers Fraction")
    ax4.set_ylim(bottom=0, top=1)

    f.suptitle(var)

    plt.ion()
    plt.show()


def plot_floating(data, var):
    f, (ax1, ax2) = plt.subplots(
        2, 1, sharex='col', sharey=False, figsize=(14, 8))

    h = data.loc[:, [var, 'is_good_customer']]
    dfb = binned_df(h)

    r = dfb[str(var) + '_mid'].values
    barWidth = (r.max() - r.min()) / 20
    bars1 = dfb['true'].values
    bars2 = dfb['false'].values

    ax1.bar(
        r,
        bars1,
        color='#557f2d',
        edgecolor='white',
        width=barWidth,
        label='Good Customer')
    ax1.bar(
        r,
        bars2,
        bottom=bars1,
        color='#7f6d5f',
        edgecolor='white',
        width=barWidth,
        label='Bad Customer')
    ax2.plot(
        r, dfb['ratio'].values, marker='o', linestyle='None', color='#557f2d')
    ax1.legend()
    ax2.set_title("Good Customers Fraction")
    ax2.set_ylim(bottom=0, top=1)

    f.suptitle(var)

    plt.ion()
    plt.show()
    return f


df = pd.read_sql_query("SELECT * FROM members.good_customers", localdb)

df['minutes_to_convert'] = (
    df['became_member_at'] - df['created_at']) / pd.Timedelta(
        minutes=1, seconds=0)
df['hours_to_convert'] = (
    df['became_member_at'] - df['created_at']) / pd.Timedelta(
        hours=1, seconds=0)

df = df.loc[df['num_kids'] < 6, ]
df['domain'] = df['email'].apply(lambda x: x.split('@')[1])

targ = pd.read_sql_query("SELECT * FROM members.targets", localdb)
df = pd.merge(df, targ, how='left', left_on='zipcode', right_on='ZipCode')
df['target'] = df['ZipCode'].apply(lambda x: True if x == x else False)
df.drop('ZipCode', axis=1, inplace=True)

jan = df['created_at'].map(lambda x: x > pd.to_datetime('2018-01-01'))
jun = df['created_at'].map(lambda x: x < pd.to_datetime('2018-06-01'))

# df.loc[df['minutes_to_convert'] < 20, 'minutes_to_convert'].hist(bins=20)

# var = 'families_density'
# plt.figure(figsize=(14, 10))
# # f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
# #     2, 2, sharex='col', sharey=False, figsize=(14, 8))
# barWidth = 1

# h_lo, h_hi = split_quantile(df, var)
# df_lo = binned_df(h_lo)
# df_hi = binned_df(h_hi)

# r = df_lo[str(var) + '_mid'].values
# bars1 = df_lo['true'].values
# bars2 = df_lo['false'].values

# plt.bar(r, bars1, color='#7f6d5f', edgecolor='white', width=barWidth)
# plt.bar(r, bars2, bottom=bars1, color='#557f2d', edgecolor='white', width=barWidth)
# plt.show()
