from sqlalchemy import create_engine
import psycopg2
import pandas as pd
from dateutil.relativedelta import relativedelta
from datetime import date
from matplotlib import pyplot as plt
import seaborn as sns

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


df = pd.read_sql_query("SELECT * FROM members.good_customers", localdb)

df['minutes_to_become'] = (
    df['became_member_at'] - df['created_at']) / pd.Timedelta(
        minutes=1, seconds=0)
df['hours_to_become'] = (
    df['became_member_at'] - df['created_at']) / pd.Timedelta(
        hours=1, seconds=0)


two_weeks_ago = df['created_at'].max() - relativedelta(weeks=2)

# remove new customers
df = df.loc[df['created_at'] < two_weeks_ago, ]

df['is_good_customer'].fillna(False, inplace=True)

df.loc[df['minutes_to_become'] < 20, 'minutes_to_become'].hist(bins=20)
