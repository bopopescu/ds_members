import psycopg2
import argparse
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from ua_parser import user_agent_parser as up
from query_dbs import user_agents

from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score
import scikitplot as skplt

sns.set_style('whitegrid')

slave = psycopg2.connect(service="rockets-slave")
segment = psycopg2.connect(service='rockets-segment')
localdb = create_engine(
    'postgresql://',
    echo=False,
    pool_recycle=300,
    creator=lambda _: psycopg2.connect(service="rockets-local"))

parser = argparse.ArgumentParser()
parser.add_argument(
    '-b', '--baseline', help='train baseline rf', action='store_true')
parser.add_argument(
    '-g', '--randomgrid', help='train random grid search', action='store_true')
parser.add_argument(
    '-u',
    '--upsample',
    help='train on a minority upsampled set',
    action='store_true')

args = parser.parse_args()


def split_quantile(data, var, q=.9):

    h = data.sort_values(by=var, ascending=True)
    tot = h.shape[0]
    q_idx = int(np.ceil(q * tot))

    h_lo = h.iloc[:q_idx, :]

    return h_lo


def binned_df(data, var):

    bin_size = (data[var].max() - data[var].min()) / 20
    bins_range = np.arange(data[var].min(), data[var].max() + bin_size,
                           bin_size)

    binned = pd.cut(data.loc[:, var], bins_range)
    binned.dropna(inplace=True)

    h = pd.value_counts(binned)
    h = pd.DataFrame(h)
    h[str(var) + '_mid'] = [p.mid for p in h.index]

    return h


def plot_floating(data, var, quantile=None):
    f, ax1 = plt.subplots(1, 1, figsize=(14, 8))
    h_lo = data.copy()

    if quantile:
        h_lo = split_quantile(data, var, quantile)

    df = binned_df(h_lo, var)
    r = df[str(var) + '_mid'].values
    barWidth = (r.max() - r.min()) / 20
    bars = df[var].values

    ax1.bar(r, bars, color='#557f2d', edgecolor='white', width=barWidth)
    f.suptitle(var)
    plt.ion()
    plt.show()


def cat_encode(X, Y, x, y):
    Xcols = X.columns.tolist()
    categoricals = [
        Xcols.index(col) for col in Xcols
        if X[col].dtypes in ['object', 'bool']
    ]

    Xcopy = X.copy()
    Ycopy = Y.copy()
    xcopy = x.copy()
    ycopy = y.copy()

    encoders = {}
    for col in Xcopy.columns:
        if Xcopy[col].dtypes in ['object', 'bool']:
            encoders[col] = LabelEncoder()
            encoders[col].fit(Xcopy[col].values)
            Xcopy[col] = encoders[col].transform(Xcopy[col])
    enc = OneHotEncoder(categorical_features=categoricals, sparse=False)
    # enc.fit(X.as_matrix())
    # Xenc = enc.transform(X)
    enc.fit(Xcopy)
    Xenc = enc.transform(Xcopy)

    for col in xcopy.columns:
        if xcopy[col].dtypes in ['object', 'bool']:
            xcopy[col] = encoders[col].transform(xcopy[col])
    xenc = enc.transform(xcopy)

    return Xenc, xenc



def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


try:
    d
    print("Found")

except NameError:
    b = pd.read_sql_query(
        """
        SELECT b.id               AS box_id,
            u.id               AS user_id,
            p.id AS kid_id,
            b.state AS box_state,
            CASE
                WHEN b.state IN ('payment_failed', 'uncollected', 'lost', 'auth_failed') THEN TRUE
                ELSE FALSE END AS is_fraud,
            u.state AS user_state,
            service_fee_amount,
            u.created_at,
            u.became_member_at,
            a.zipcode,
            b.approved_at,
            nkids.num_boys,
            nkids.num_girls,
            nkids.num_kids
        FROM boxes b
                JOIN kid_profiles p ON b.kid_profile_id = p.id
                JOIN users u ON u.id = p.user_id
                JOIN spree_addresses a ON u.ship_address_id = a.id
                JOIN (SELECT user_id,
                            sum(
                                CASE WHEN gender :: text = 'boys' :: text THEN 1 ELSE 0 END) AS  num_boys,
                            sum(
                                CASE WHEN gender :: text = 'girls' :: text THEN 1 ELSE 0 END) AS num_girls,
                            count(*) AS                                                          num_kids
                    FROM kid_profiles
                    GROUP BY user_id
                    ORDER BY user_id) nkids ON nkids.user_id = u.id

        WHERE b.state NOT IN ('new', 'new_invalid', 'canceled')
            AND service_fee_amount = 5
            AND u.email NOT ILIKE '%@rocketsofawesome.com';
    """, slave)

    b['days_to_convert'] = (b['became_member_at'] - b['created_at']).dt.days

    c = pd.read_sql(
        """
        SELECT zip,
            unemploym_rate_civil,
            med_fam_income,
            married_couples__density,
            females_density
        FROM members.census
    """, localdb)

    d = pd.merge(b, c, how='left', left_on='zipcode', right_on='zip')
    d.drop(['zip'], axis=1, inplace=True)

    kids_list = '(' + ', '.join([str(x) for x in b['kid_id'].unique()]) + ')'

    kps = pd.read_sql_query(
        "SELECT * FROM members.kid_preferences WHERE kid_id IN {profiles}"
        .format(profiles=kids_list), localdb)

    kps['color_count'] = kps.iloc[:, 1:11].notnull().sum(axis=1)
    kps['blacklist_count'] = kps.iloc[:, 11:25].notnull().sum(axis=1)
    kps['outfit_count'] = kps.iloc[:, 25:54].notnull().sum(axis=1)
    kps['style_count'] = kps.iloc[:, 57:66].notnull().sum(axis=1)

    kps['color_count'] = kps['color_count'].apply(lambda x: 1 if x > 0 else 0)
    kps['blacklist_count'] = kps['blacklist_count'].apply(
        lambda x: 1 if x > 0 else 0)
    kps['outfit_count'] = kps['outfit_count'].apply(lambda x: 1 if x > 0 else 0)
    kps['style_count'] = kps['style_count'].apply(lambda x: 1 if x > 0 else 0)
    kps['note_length'] = kps['note'].apply(lambda x: len(x) if x else 0)
    # kps['note_count'] = kps['note'].apply(lambda x: 1 if pd.notnull(x) else 0)
    kps['swim_count'] = kps['swim'].apply(lambda x: 1 if pd.notnull(x) else 0)
    kps['neon_count'] = kps['neon'].apply(lambda x: 1 if pd.notnull(x) else 0)
    kps['text_on_clothes_count'] = kps['text_on_clothes'].apply(
        lambda x: 1 if pd.notnull(x) else 0)
    kps['backpack_count'] = kps['backpack'].apply(
        lambda x: 1 if pd.notnull(x) else 0)
    kps['teams_count'] = kps['teams'].apply(lambda x: 1 if pd.notnull(x) else 0)

    kps['n_preferences'] = kps.loc[:, [
        'color_count', 'blacklist_count', 'outfit_count', 'style_count',
        'swim_count', 'neon_count', 'text_on_clothes_count', 'backpack_count',
        'teams_count'
    ]].sum(axis=1)

    kps.drop([
        'color_count', 'blacklist_count', 'outfit_count', 'style_count',
        'swim_count', 'neon_count', 'text_on_clothes_count', 'backpack_count',
        'teams_count'
    ],
             axis=1,
             inplace=True)

    d = pd.merge(
        d, kps.loc[:, ['kid_id', 'note_length', 'n_preferences']], how='left')

    users_list = '(' + ', '.join([str(x) for x in b['user_id'].unique()]) + ')'

    cc = pd.read_sql_query(
        """
        SELECT user_id,
            cc_type,
            (make_date(cast(year as int), cast(month as int), 1) 
                + interval '1 month' - interval '1 day') :: date AS exp_date
        FROM spree_credit_cards
        WHERE "default" = TRUE
            AND user_id IN {users}
    """.format(users=users_list), slave)

    d = pd.merge(d, cc, how='left')
    d['days_to_exp'] = (
        pd.to_datetime(d['exp_date']) - d['became_member_at']).dt.days

    adds = pd.read_sql_query(
        """
        SELECT id as user_id,
            ship_address_id,
            bill_address_id
        FROM users
        WHERE id IN {users}
    """.format(users=users_list), slave)
    adds['diff_addresses'] = (
        adds['ship_address_id'] - adds['bill_address_id']).map(lambda x: x > 0)
    adds.drop(['ship_address_id', 'bill_address_id'], axis=1, inplace=True)

    d = pd.merge(d, adds, how='left')

df = d.loc[:, [
    'num_kids', 'days_to_convert', 'unemploym_rate_civil', 'med_fam_income',
    'married_couples__density', 'females_density', 'n_preferences', 'cc_type',
    'days_to_exp', 'diff_addresses', 'note_length', 'is_fraud'
]]

imp = Imputer(strategy='median', axis=0, missing_values='NaN')

for col in df[[
        'unemploym_rate_civil', 'med_fam_income', 'married_couples__density',
        'females_density'
]].columns:
    df[col] = imp.fit_transform((df[[col]]))

df.dropna(inplace=True)

train, test = train_test_split(df, test_size=0.2, random_state=42)
X = train.drop('is_fraud', axis=1)
Y = train['is_fraud']
x = test.drop('is_fraud', axis=1)
y = test['is_fraud']

rf = RandomForestClassifier()
Xenc, xenc = cat_encode(X, Y, x, y)

if args.baseline:
    rf.fit(Xenc, Y)
    preds = rf.predict(Xenc)
    probs = rf.predict_proba(Xenc)
    print(accuracy_score(Y, preds))
    # Keep only the positive class
    probs = [p[1] for p in probs]
    print(roc_auc_score(Y, probs))

    test_preds = rf.predict(xenc)
    test_probs = rf.predict_proba(xenc)

    # test_probs = [p[1] for p in test_probs]
    # predictions = cross_val_predict(rf, x, y)

    skplt.metrics.plot_confusion_matrix(y, test_preds, normalize=True)
    skplt.metrics.plot_precision_recall(y, test_probs)
    skplt.metrics.plot_roc(y, test_probs)
    clf_names = ['Random Forest']
    skplt.metrics.plot_calibration_curve(y, [test_probs], clf_names)
    plot_learning_curve(rf, "Random Forest", Xenc, Y)

if args.randomgrid:
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start=1000, stop=2500, num=4)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    class_weight = ['balanced_subsample', 'balanced']
    class_weight.append(None)

    random_grid = {
        'n_estimators': n_estimators,
        # 'max_features': max_features,
        'max_depth': max_depth,
        'min_samples_split': min_samples_split,
        'min_samples_leaf': min_samples_leaf
        # 'bootstrap': bootstrap,
        # 'class_weight': class_weight
    }

    rf_random = RandomizedSearchCV(
        estimator=rf,
        scoring='f1',
        param_distributions=random_grid,
        n_iter=50,
        cv=3,
        verbose=2,
        random_state=42,
        n_jobs=-1)
    # Fit the random search model
    rf_random.fit(X, Y)

    random_preds = rf_random.predict(x)
    random_probs = rf_random.predict_proba(x)
    skplt.metrics.plot_roc(y, random_probs)
    skplt.metrics.plot_precision_recall(y, random_probs)
    skplt.metrics.plot_confusion_matrix(y, random_preds, normalize=True)

if args.upsample:
    train_majority = train.loc[train['is_fraud'] == False,]
    train_minority = train.loc[train['is_fraud'] == True,]

    train_minority_upsampled = resample(
        train_minority,
        replace=True,  # sample with replacement
        n_samples=train_majority.shape[0],  # to match majority class
        random_state=123)  # reproducible results

    train_upsampled = pd.concat([train_majority, train_minority_upsampled])
    Xup = train_upsampled.drop('is_fraud', axis=1)
    Yup = train_upsampled['is_fraud']

    rfup = RandomForestClassifier()
    rfup.fit(Xup, Yup)

    preds_up = rfup.predict(x)
    probs_up = rfup.predict_proba(x)
    skplt.metrics.plot_roc(y, probs_up)
    skplt.metrics.plot_precision_recall(y, probs_up)
    skplt.metrics.plot_confusion_matrix(y, preds_up, normalize=True)
