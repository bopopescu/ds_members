import time

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_distances_argmin
from sqlalchemy import create_engine
import psycopg2

sns.set_style('whitegrid')

localdb = create_engine(
    'postgresql://',
    echo=False,
    pool_recycle=300,
    creator=lambda _: psycopg2.connect(service='rockets-local'))

gc_sql = """
    SELECT * FROM members.good_customers;
"""

data = pd.read_sql_query(gc_sql, localdb)
data.dropna(inplace=True)

# Plot correlations
corr = data.corr()
# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})


# Cluster
X = data.loc[:, ['service_fee', 'num_kids', 'unemploym_rate_civil',
                 'employed_female_percent', 'med_hh_income',
                 'households_density']]


n_clusters = 3

k_means = KMeans(init='k-means++', n_clusters=n_clusters, n_init=10)
t0 = time.time()
k_means.fit(X)
t_batch = time.time() - t0

# Plot result

# We want to have the same colors for the same cluster from the
# KMeans algorithm. Let's pair the cluster centers per
# closest one.
k_means_cluster_centers = np.sort(k_means.cluster_centers_, axis=0)
k_means_labels = pairwise_distances_argmin(X, k_means_cluster_centers)

X['k_means_labels'] = k_means_labels

fig = plt.figure(figsize=(14, 8))
fig.subplots_adjust(left=0.02, right=0.98, bottom=0.05, top=0.9)
colors = ['#4EACC5', '#FF9C34', '#4E9A06']

xaxis = 'unemploym_rate_civil'
yaxis = 'employed_female_percent'


# KMeans
ax = fig.add_subplot(1, 2, 1)
for k, col in zip(range(n_clusters), colors):
    my_members = k_means_labels == k
    cluster_center = k_means_cluster_centers[k]
    # ax.plot(X[my_members, 0], X[my_members, 1], 'w',
    #         markerfacecolor=col, marker='.')
    ax.plot(
        X.loc[X['k_means_labels'] == k, xaxis],
        X.loc[X['k_means_labels'] == k, yaxis],
        'w',
        markerfacecolor=col,
        marker='o')

    # ax.plot(cluster_center[8], cluster_center[10], 'o', markerfacecolor=col,
    #         markeredgecolor='k', markersize=6)
ax.set_title('N clusters: %.0f' % (n_clusters))
ax.set_xlabel(xaxis)
ax.set_ylabel(yaxis)

plt.show()
