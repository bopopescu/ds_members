"""Model fraudulent $5 accounts.

Returns:
    None -- Saves random forest to pickle file

"""

import psycopg2
import argparse
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from ua_parser import user_agent_parser as up
from query_dbs import query_kid_preferences

from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import PolynomialFeatures
from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import learning_curve

from sklearn.svm import SVC
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score
import scikitplot as skplt
from sklearn.externals import joblib

from rfpimp import importances
from rfpimp import plot_importances
from rfpimp import plot_corr_heatmap

sns.set_style('whitegrid')
pd.set_option('display.max_columns', 500)

stitch = create_engine(
    'postgresql://',
    echo=False,
    pool_recycle=300,
    echo_pool=True,
    creator=lambda _: psycopg2.connect(service='rockets-stitch'))

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
parser.add_argument(
    '-p', '--poly', help='polynomial features', action='store_true')
parser.add_argument('-x', '--xgb', help='train xgb', action='store_true')
parser.add_argument(
    '-m',
    '--importances',
    help='calculate and plot feature importances',
    action='store_true')
parser.add_argument(
    '-s', '--svm', help='classify with SVM', action='store_true')
parser.add_argument(
    '-d', '--dumpfile', help='Save model to pkl file', action='store_true')

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


def cat_encode(X, x):
    Xcols = X.columns.tolist()
    categoricals = [
        Xcols.index(col)
        for col in Xcols
        if X[col].dtypes in ['object', 'bool']
    ]

    Xcopy = X.copy()
    xcopy = x.copy()

    encoders = {}
    for col in Xcopy.columns:
        if Xcopy[col].dtypes in ['object', 'bool']:
            encoders[col] = LabelEncoder()
            encoders[col].fit(Xcopy[col].values)
            Xcopy[col] = encoders[col].transform(Xcopy[col])
    enc = OneHotEncoder(categorical_features=categoricals, sparse=False)
    enc.fit(Xcopy)
    Xenc = enc.transform(Xcopy)

    for col in xcopy.columns:
        if xcopy[col].dtypes in ['object', 'bool']:
            xcopy[col] = encoders[col].transform(xcopy[col])
    xenc = enc.transform(xcopy)

    if args.dumpfile:
        joblib.dump(encoders, 'lab_encs_xgb.pkl')
        joblib.dump(enc, 'oh_enc_xgb.pkl')

    return Xenc, xenc


def plot_learning_curve(estimator,
                        title,
                        X,
                        y,
                        ylim=None,
                        cv=None,
                        n_jobs=1,
                        train_sizes=np.linspace(.1, 1.0, 5)):
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

    plt.fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color="r")
    plt.fill_between(
        train_sizes,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
        color="g")
    plt.plot(
        train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(
        train_sizes,
        test_scores_mean,
        'o-',
        color="g",
        label="Cross-validation score")

    plt.legend(loc="best")
    return plt


try:
    d
    print("Found")

except NameError:

    # b = pd.read_sql_query("""
    #     SELECT b.box_id,
    #         b.user_id,
    #         b.kid_profile_id AS kid_id,
    #         b.state                                                               AS box_state,
    #         CASE
    #             WHEN b.state IN ('payment_failed', 'uncollected', 'lost', 'auth_failed') THEN TRUE
    #             ELSE FALSE END                                                    AS is_fraud,
    #         -- u.state                                                               AS user_state,
    #         service_fee_amount,
    #         u.service_fee_enabled,
    #         left(a.zipcode, 5)                                                    AS zipcode,
    #         became_member_at,
    #         u.num_boys,
    #         u.num_girls,
    #         u.num_kids,
    #         CASE WHEN ship_address_id != bill_address_id THEN TRUE ELSE FALSE END AS diff_addresses,
    #         ch.channel,
    #         date_part('days', u.became_member_at - u.created_at)                  AS days_to_convert

    #     FROM dw.fact_boxes b
    #             JOIN dw.fact_active_users u ON u.user_id = b.user_id
    #             JOIN stitch_quark.spree_addresses a ON u.ship_address_id = a.id
    #             LEFT JOIN dw.fact_channel_attribution ch ON ch.user_id = b.user_id
    #     WHERE b.state IN ('payment_failed', 'uncollected', 'lost', 'auth_failed', 'final')
    #     AND season_id BETWEEN 7 AND 9
    #     AND became_member_at >= '2018-01-01'
    #     AND service_fee_amount = 5
    #     AND u.email NOT ILIKE '%%@rocketsofawesome.com'
    # """, stitch)

    u = pd.read_sql_query(
        """
        SELECT DISTINCT 
            bc.user_id,
            bc.state,
            ch.channel,
            u.num_boys,
            u.num_girls,
            u.num_kids,
            u.service_fee_enabled,
            CASE WHEN ship_address_id != bill_address_id THEN TRUE ELSE FALSE END AS diff_addresses,
            date_part('days', u.became_member_at - u.created_at)                  AS days_to_convert,
            unemploym_rate_civil,
            med_hh_income,
            married_couples_density,
            females_density,
            n_kids / area                                                         AS kids_density,
            households_density,
            kids_school_perc,
            kids_priv_school_perc,
            smocapi_20,
            smocapi_25,
            smocapi_30,
            smocapi_35,
            smocapi_high,
            grapi_15,
            grapi_20,
            grapi_25,
            grapi_30,
            grapi_35,
            grapi_high,
            CASE WHEN bc.state IN ('payment_failed', 'uncollected', 'lost', 'auth_failed') THEN TRUE ELSE FALSE END AS is_fraud

        FROM dw.fact_user_box_count bc
                LEFT JOIN dw.fact_channel_attribution ch ON ch.user_id = bc.user_id
                JOIN dw.fact_active_users u ON bc.user_id = u.user_id
                JOIN stitch_quark.spree_addresses a ON u.ship_address_id = a.id
                JOIN dw.dim_census c ON left(a.zipcode, 5) = c.zip
        WHERE season_id BETWEEN 7 AND 9
            AND user_box_rank = 1
            AND bc.state NOT IN ('new', 'shipped', 'needs_review', 'canceled', 'skipped')
            AND u.email NOT ILIKE '%%@rocketsofawesome.com'
    """, stitch)

    # Cleanup of multiple entries
    u_counts = u.groupby('user_id')['user_id'].count().sort_values(
        ascending=False)
    idx = u_counts[u_counts == 2].index
    mult_u = u.loc[u['user_id'].isin(idx),]
    single_u = u.loc[u['user_id'].isin(idx) == False,]
    single_u.drop('state', axis=1, inplace=True)

    fraud_or_not = mult_u.groupby('user_id')['is_fraud'].sum()
    frauds = fraud_or_not[fraud_or_not > 0].index
    not_frauds = fraud_or_not[fraud_or_not == 0].index
    mult_u.drop(['state', 'is_fraud'], axis=1, inplace=True)
    mult_u.drop_duplicates(inplace=True)
    mult_u['is_fraud'] = False
    mult_u.loc[mult_u['user_id'].isin(frauds), 'is_fraud'] = True
    users = pd.concat([single_u, mult_u], axis=0)

    # d = pd.merge(b, c, how='left', left_on='zipcode', right_on='zip')
    # d.drop(['zip'], axis=1, inplace=True)

    # kids_list = '(' + ', '.join([str(x) for x in b['kid_id'].unique()]) + ')'

    # kps = query_kid_preferences(kids_list)

    # kps['color_count'] = kps.iloc[:, 1:11].notnull().sum(axis=1)
    # kps['blacklist_count'] = kps.iloc[:, 11:25].notnull().sum(axis=1)
    # kps['outfit_count'] = kps.iloc[:, 25:54].notnull().sum(axis=1)
    # kps['style_count'] = kps.iloc[:, 57:66].notnull().sum(axis=1)

    # kps['color_count'] = kps['color_count'].apply(lambda x: 1 if x > 0 else 0)
    # kps['blacklist_count'] = kps['blacklist_count'].apply(
    #     lambda x: 1 if x > 0 else 0)
    # kps['outfit_count'] = kps['outfit_count'].apply(lambda x: 1 if x > 0 else 0)
    # kps['style_count'] = kps['style_count'].apply(lambda x: 1 if x > 0 else 0)
    # kps['note_length'] = kps['note'].apply(lambda x: len(x) if x else 0)
    # kps['swim_count'] = kps['swim'].apply(lambda x: 1 if pd.notnull(x) else 0)
    # kps['neon_count'] = kps['neon'].apply(lambda x: 1 if pd.notnull(x) else 0)
    # kps['text_on_clothes_count'] = kps['text_on_clothes'].apply(
    #     lambda x: 1 if pd.notnull(x) else 0)
    # kps['backpack_count'] = kps['backpack'].apply(
    #     lambda x: 1 if pd.notnull(x) else 0)
    # kps['teams_count'] = kps['teams'].apply(lambda x: 1 if pd.notnull(x) else 0)

    # kps['n_preferences'] = kps.loc[:, [
    #     'color_count', 'blacklist_count', 'outfit_count', 'style_count',
    #     'swim_count', 'neon_count', 'text_on_clothes_count', 'backpack_count',
    #     'teams_count'
    # ]].sum(axis=1)

    # kps.drop([
    #     'color_count', 'blacklist_count', 'outfit_count', 'style_count',
    #     'swim_count', 'neon_count', 'text_on_clothes_count', 'backpack_count',
    #     'teams_count'
    # ],
    #          axis=1,
    #          inplace=True)

    # d = pd.merge(
    #     d, kps.loc[:, ['kid_id', 'note_length', 'n_preferences']], how='left')

    users_list = '(' + ', '.join([str(x) for x in users['user_id'].unique()
                                 ]) + ')'

    cc = pd.read_sql_query(
        """
        SELECT u.user_id,
            cc.cc_type,
            ((make_date(cast(year AS int), cast(month AS int), 1)
                    + INTERVAL '1 month' - INTERVAL '1 day') :: DATE - u.became_member_at :: date) / 30 AS months_to_exp,
            ccd.funding
        FROM stitch_quark.spree_credit_cards cc
                JOIN dw.fact_active_users u ON cc.user_id :: BIGINT = u.user_id
                join stitch_stripe.stripe_customers__cards__data ccd on cc.gateway_customer_profile_id = ccd.customer
        WHERE "default" = TRUE
        AND cc.user_id :: BIGINT IN {users}
    """.format(users=users_list), stitch)

    d = pd.merge(users, cc)

    uas = pd.read_sql_query(
        """
        SELECT user_id,
            context_user_agent
        FROM dw.fact_first_click_first_pass
        WHERE user_id in {users}
    """.format(users=users_list), stitch)

    d = pd.merge(d, uas, how='left')

    d.dropna(inplace=True)

    d['OS'] = d['context_user_agent'].apply(
        lambda x: up.ParseOS(x)['family'] if pd.notnull(x) else None)
    d.loc[d['OS'] == 'Mac OS X', 'OS'] = 'Mac'
    d.loc[d['OS'].isin(['Mac', 'Windows', 'iOS', 'Android']) ==
          False, 'OS'] = 'Other'
    d.drop('context_user_agent', axis=1, inplace=True)

df = d.loc[:, [
    'user_id', 'funding', 'OS', 'service_fee_enabled', 'med_hh_income',
    'kids_school_perc', 'kids_priv_school_perc', 'smocapi_20', 'smocapi_25',
    'smocapi_30', 'smocapi_35', 'grapi_15', 'grapi_20', 'grapi_25', 'grapi_30',
    'grapi_35', 'num_kids', 'num_girls', 'days_to_convert',
    'unemploym_rate_civil', 'married_couples_density', 'cc_type',
    'diff_addresses', 'months_to_exp', 'channel', 'is_fraud'
]]

# imp = Imputer(strategy='median', axis=0, missing_values='NaN')

# for col in df[[
#         'med_hh_income', 'kids_school_perc', 'kids_priv_school_perc',
#         'smocapi_20', 'smocapi_25', 'smocapi_30', 'smocapi_35', 'grapi_15',
#         'grapi_20', 'grapi_25', 'grapi_30', 'grapi_35', 'num_kids',
#         'days_to_convert', 'unemploym_rate_civil', 'married_couples_density'
# ]].columns:
#     df[col] = imp.fit_transform((df[[col]]))

# df.dropna(inplace=True)

tr_id, ts_id = train_test_split(
    df['user_id'].unique(), test_size=0.2, random_state=3)
train = df.loc[df['user_id'].isin(tr_id),]
test = df.loc[df['user_id'].isin(ts_id),]

X = train.drop(['user_id', 'is_fraud'], axis=1)
Y = train['is_fraud']
x = test.drop(['user_id', 'is_fraud'], axis=1)
y = test['is_fraud']

Xenc, xenc = cat_encode(X, x)

if args.poly:
    poly = PolynomialFeatures(2)
    Xenc = poly.fit_transform(Xenc)
    xenc = poly.transform(xenc)

if args.baseline:
    rf = RandomForestClassifier(n_jobs=-1)
    rf.fit(Xenc, Y)
    preds = rf.predict(Xenc)
    probs = rf.predict_proba(Xenc)
    print('train accuracy score: {0:2f}'.format(accuracy_score(Y, preds)))

    cv_preds = cross_val_predict(rf, Xenc, Y, cv=5)
    print('train cv accuracy score: {0:2f}'.format(accuracy_score(Y, cv_preds)))
    # Keep only the positive class
    probs = [p[1] for p in probs]
    print('train roc score: {0:2f}'.format(roc_auc_score(Y, probs)))

    test_preds = rf.predict(xenc)
    test_probs = rf.predict_proba(xenc)
    print('test accuracy score: {0:2f}'.format(accuracy_score(y, test_preds)))

    test_probs_1 = [p[1] for p in test_probs]
    print('test roc score: {0:2f}'.format(roc_auc_score(y, test_probs_1)))

    # predictions = cross_val_predict(rf, x, y)

    skplt.metrics.plot_confusion_matrix(y, test_preds, normalize=True)
    skplt.metrics.plot_confusion_matrix(y, test_preds, normalize=False)
    skplt.metrics.plot_precision_recall(y, test_probs)
    skplt.metrics.plot_roc(y, test_probs)
    clf_names = ['Random Forest']
    skplt.metrics.plot_calibration_curve(y, [test_probs], clf_names)
    plot_learning_curve(rf, "Random Forest", Xenc, Y)


if args.xgb:
    xgb = XGBClassifier(n_jobs=2, scale_pos_weight=(Y.shape[0] - Y.sum())/Y.sum())
    xgb.fit(Xenc, Y)
    preds = xgb.predict(Xenc)
    probs = xgb.predict_proba(Xenc)
    print('train accuracy score: {0:2f}'.format(accuracy_score(Y, preds)))

    cv_preds = cross_val_predict(xgb, Xenc, Y, cv=5)
    print('train cv accuracy score: {0:2f}'.format(accuracy_score(Y, cv_preds)))
    # Keep only the positive class
    probs = [p[1] for p in probs]
    print('train roc score: {0:2f}'.format(roc_auc_score(Y, probs)))

    test_preds = xgb.predict(xenc)
    test_probs = xgb.predict_proba(xenc)
    print('test accuracy score: {0:2f}'.format(accuracy_score(y, test_preds)))

    test_probs_1 = [p[1] for p in test_probs]
    print('test roc score: {0:2f}'.format(roc_auc_score(y, test_probs_1)))

    # predictions = cross_val_predict(xgb, x, y)

    skplt.metrics.plot_confusion_matrix(y, test_preds, normalize=True)
    skplt.metrics.plot_confusion_matrix(y, test_preds, normalize=False)
    skplt.metrics.plot_precision_recall(y, test_probs)
    skplt.metrics.plot_roc(y, test_probs)
    clf_names = ['XGB']
    skplt.metrics.plot_calibration_curve(y, [test_probs], clf_names)
    plot_learning_curve(xgb, "XGB", Xenc, Y)


if args.dumpfile:
    if args.baseline:
        joblib.dump(rf, "rf.pkl")
    if args.xgb:
        joblib.dump(xgb, 'xgb.pkl')
    if args.poly:
        joblib.dump(poly, 'poly.pkl')


if args.svm:
    svm = SVC(
        kernel='linear',
        class_weight='balanced',  # penalize
        probability=True)
    svm.fit(Xenc, Y)
    preds = svm.predict(Xenc)
    probs = svm.predict_proba(Xenc)
    print(accuracy_score(Y, preds))
    # Keep only the positive class
    probs = [p[1] for p in probs]
    print(roc_auc_score(Y, probs))

    test_preds = svm.predict(xenc)
    test_probs = svm.predict_proba(xenc)

    # test_probs = [p[1] for p in test_probs]
    # predictions = cross_val_predict(svm, x, y)

    skplt.metrics.plot_confusion_matrix(y, test_preds, normalize=True)
    skplt.metrics.plot_precision_recall(y, test_probs)
    skplt.metrics.plot_roc(y, test_probs)
    clf_names = ['SVM']
    skplt.metrics.plot_calibration_curve(y, [test_probs], clf_names)
    plot_learning_curve(svm, "SVM", Xenc, Y)

if args.importances:
    rf = RandomForestClassifier(n_jobs=-1)
    Xnum = X.drop(['cc_type', 'diff_addresses'], axis=1)
    xnum = x.drop(['cc_type', 'diff_addresses'], axis=1)
    Xpoly = poly.fit_transform(Xnum)
    xpoly = poly.transform(xnum)
    Xpoly = pd.DataFrame(Xpoly, columns=poly.get_feature_names(Xnum.columns))
    xpoly = pd.DataFrame(xpoly, columns=poly.get_feature_names(xnum.columns))
    rf.fit(Xnum, Y)
    # here we assume there are no categorical variables
    imp = importances(rf, xnum, y)  # permutation
    plot_importances(imp, figsize=(8, 12))
    plot_corr_heatmap(Xnum, figsize=(11, 11))

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
