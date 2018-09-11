import argparse
from sqlalchemy import create_engine
import psycopg2
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report

# import scikitplot.plotters as skplt
# from scikitplot import classifier_factory
import scikitplot as skplt

pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)

sns.set_style('whitegrid')
localdb = create_engine(
    'postgresql://',
    echo=False,
    pool_recycle=300,
    creator=lambda _: psycopg2.connect(service="rockets-local"))
slave = psycopg2.connect(service="rockets-slave")

parser = argparse.ArgumentParser()
parser.add_argument(
    "-c",
    '--createDataset',
    help="query slave and create copy dataset to local db",
    action='store_true')

parser.add_argument(
    "-m",
    '--trainModel',
    help="query local dataset and train the model",
    action="store_true")

parser.add_argument(
    "-p",
    "--plotStuff",
    help="query local dataset and plot features",
    action="store_true")

args = parser.parse_args()


def wide_to_long(data, var_name):
    """From wide to long format.
    
    Arguments:
        data {pd.DataFrame} -- green, blue, ..., is_good_kid
        var_name {str} -- 'color', 'outfit', ...
    
    Returns:
        pd.DataFrame -- Long format data frame
    """
    s = data.melt(id_vars=['is_good_kid'], var_name=var_name)
    colors = s.groupby([var_name, 'value']).agg({
        'is_good_kid': ['sum', 'count']
    }).reset_index().rename(columns={
        'sum': 'good_kids',
        'count': 'all_kids'
    })
    colors.columns = [var_name, 'value', 'good_kids', 'all_kids']
    colors['good_kids'] = colors['good_kids'].astype('int')
    colors['bad_kids'] = colors['all_kids'] - colors['good_kids']

    return colors


def plot_stack(data, xaxis_var, hue='good_kids', title=None, horiz=False):
    """
    Make a stacked bar chart.
    
    Arguments:
        data {pd.DataFrame} -- in long format
        xaxis_var {str} -- x axis variable
    
    Keyword Arguments:
        hue {str} -- good or bad kid... (default: {'good_kids'})
        title {str} -- title to display (default: {None})
    """
    if title is None:
        title = str(xaxis_var) + " " + str(hue)

    barWidth = 1
    f, ax1 = plt.subplots(figsize=(14, 9))
    r = data[xaxis_var].unique()
    vals = data['value'].unique()
    if horiz == True:
        cumsum = np.zeros(len(r))
        for val in vals:
            y = data.loc[data['value'] == val, hue].values
            ax1.barh(
                r,
                y,
                left=cumsum,
                edgecolor='white',
                label=val)
            cumsum += y
        ax1.invert_yaxis()  # labels read top-to-bottom

    else:
        cumsum = np.zeros(len(r))
        for val in vals:
            y = data.loc[data['value'] == val, hue].values
            ax1.bar(
                r,
                y,
                bottom=cumsum,
                edgecolor='white',
                width=barWidth,
                label=val)
            cumsum += y

    ax1.legend()
    ax1.set_title(title, fontsize=18)
    plt.ion()
    plt.show()


if args.createDataset:

    with open("./sql/kids.sql", 'r') as q:
        k = pd.read_sql_query(q.read(), slave)

    # let's throw away those kids with null size preferences
    # s = set(k.loc[(k['top_size'].isnull()) | (k['bottom_size'].isnull()), 'id'])
    # ss = pd.read_sql_query("""
    # SELECT * FROM kid_profiles
    # WHERE id IN ({0})
    # """.format(', '.join([str(l) for l in s])), slave)

    k = k.loc[(k['top_size'].notnull()) & (k['bottom_size'].notnull()),]
    k['top_size'] = k['top_size'].astype('int')
    k['bottom_size'] = k['bottom_size'].astype('int')

    for col in k.columns:
        if k[col].dtype == 'object':
            k[col] = k[col].fillna('no_preference')
            k[col] = k[col].replace('', 'no_preference')

    with open("./sql/avg_kr_kid.sql", 'r') as q:
        akr = pd.read_sql_query(q.read(), slave)

    with open("./sql/is_good_kid.sql", 'r') as q:
        gk = pd.read_sql_query(q.read(), slave)

    with open("./sql/has_siblings.sql", 'r') as q:
        sk = pd.read_sql_query(q.read(), slave)

    st = pd.read_sql_query(
        """
        SELECT DISTINCT p.id    AS kid_id,
                        p.created_at :: date,
                        round((DATE_PART('year', p.created_at :: date) - DATE_PART('year', p.birthdate :: date)) * 12 +
                              DATE_PART('month', p.created_at :: date) - DATE_PART('month', p.birthdate :: date)) :: integer AS age_months,
                        u.id    AS user_id,
                        u.state AS state
        FROM kid_profiles p
                 INNER JOIN users u ON p.user_id = u.id;
        """, slave)

    d = pd.merge(akr, gk)
    d = pd.merge(d, k)
    d = pd.merge(d, sk)
    d = pd.merge(d, st)
    d['created_at'] = pd.to_datetime(d['created_at'])
    d.loc[d['note'] == '\n', 'has_note'] = False

    d.to_sql(
        "good_kids",
        localdb,
        schema="members",
        if_exists='replace',
        index=False)

elif args.trainModel:
    d = pd.read_sql_query("SELECT * FROM members.good_kids", localdb)

    # remove kids with older siblings and first box season 9
    is_recent = d['first_box_season'].map(lambda x: x == 9)
    has_older = d['has_older_siblings'].map(lambda x: x == True)
    d = d.loc[-(is_recent & has_older),]

    # look only at 80% of the data, for an eventual classification model
    train, test = train_test_split(d, test_size=0.2, random_state=42)
    train.reset_index(drop=True, inplace=True)
    test.reset_index(drop=True, inplace=True)

    X = train.drop(
        [
            'kid_id', 'avg_keep_rate', 'num_boxes', 'first_box_season',
            'is_good_kid', 'user_id', 'birthdate', 'note', 'style_other',
            'created_at', 'state'
        ],
        axis=1)
    Y = train['is_good_kid']

    x = test.drop(
        [
            'kid_id', 'avg_keep_rate', 'num_boxes', 'first_box_season',
            'is_good_kid', 'user_id', 'birthdate', 'note', 'style_other',
            'created_at', 'state'
        ],
        axis=1)
    y = test['is_good_kid']

    Xcols = X.columns.tolist()
    categoricals = [
        Xcols.index(col) for col in Xcols
        if X[col].dtypes in ['object', 'bool']
    ]

    Xcopy = X.copy()
    Ycopy = Y.copy()

    encoders = {}
    for col in X.columns:
        if X[col].dtypes in ['object', 'bool']:
            encoders[col] = LabelEncoder()
            encoders[col].fit(X[col].values)
            X[col] = encoders[col].transform(X[col])
    enc = OneHotEncoder(categorical_features=categoricals, sparse=False)
    # enc.fit(X.as_matrix())
    # Xenc = enc.transform(X)
    enc.fit(X)
    Xenc = enc.transform(X)

    for col in x.columns:
        if x[col].dtypes in ['object', 'bool']:
            x[col] = encoders[col].transform(x[col])
    xenc = enc.transform(x)

    # Scale
    # scaler = StandardScaler(with_mean=False).fit(Xenc)
    # Xscaled = scaler.transform(Xenc)
    # xscaled = scaler.transform(xenc)

    rf = RandomForestClassifier()
    # classifier_factory(rf)
    rf.fit(X, Y)

    # predicted_probas = rf.predict_proba(xscaled)

elif args.plotStuff:
    d = pd.read_sql_query("SELECT * FROM members.good_kids", localdb)

    # remove kids with older siblings and first box season 9
    is_recent = d['first_box_season'].map(lambda x: x == 9)
    has_older = d['has_older_siblings'].map(lambda x: x == True)
    d = d.loc[-(is_recent & has_older),]

    s = d.loc[:, [
        'blue', 'green', 'orange', 'pink', 'purple', 'red', 'silver', 'white',
        'yellow', 'is_good_kid'
    ]]

    colors = wide_to_long(s, 'color')
    # plot_stack(
    #     colors,
    #     'color',
    #     hue='good_kids',
    #     title="Color preferences for good kids")

    ofb = d.loc[:, [
        'of_basics', 'of_boys_active_1', 'of_boys_active_2',
        'of_boys_athleisure', 'of_boys_essential', 'of_boys_essentials_1',
        'of_boys_performance_active', 'of_boys_preppy_1', 'of_boys_preppy_2',
        'of_boys_preppy_classic', 'of_boys_preppy_cool', 'of_boys_trendy',
        'of_boys_trendy_1', 'of_boys_trendy_2', 'of_classic', 'of_cool',
        'is_good_kid'
    ]]
    ofbw = wide_to_long(ofb, 'outfit')
    # plot_stack(ofs, 'outfit', hue='good_kids', title='Outfit preference')

    ofg = d.loc[:, [
        'of_basics', 'of_girls_active', 'of_girls_active_1',
        'of_girls_essential', 'of_girls_essentials_1', 'of_girls_girly',
        'of_girls_preppy', 'of_girls_preppy_1', 'of_girls_preppy_2',
        'of_girls_trendy_1', 'of_girls_trendy_2', 'of_girls_trendy_awesome',
        'of_girls_trendy_cool', 'of_sporty', 'is_good_kid'
    ]]
    ofgw = wide_to_long(ofg, 'outfit')