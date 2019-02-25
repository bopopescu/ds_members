import pandas as pd
import psycopg2
from sqlalchemy import create_engine
pd.set_option('max_colwidth', 200)

# stuff taken from https://www.earthdatascience.org/tutorials/get-cenus-data-with-cenpy/
# not really needed
# import cenpy as cen
# import re

# datasets = list(cen.explorer.available(verbose=True).items())

# # df = pd.DataFrame(datasets, columns=['code', 'description'])
# for d in datasets:
#     if re.search('acs', d[1], re.I):
#         print(d)

# dataset = 'ACSDT1Y2016'
# cen.explorer.explain(dataset)

# con = cen.base.Connection(dataset)

localdb = create_engine(
    'postgresql://',
    echo=False,
    pool_recycle=300,
    creator=lambda _: psycopg2.connect(service="rockets-local"))

stitch = create_engine(
    'postgresql://',
    echo=True,
    pool_recycle=300,
    echo_pool=True,
    creator=lambda _: psycopg2.connect(service='rockets-stitch'))

# create_engine('redshift+psycopg2://username@host.amazonaws.com:5439/database')

redshift = create_engine(
    'redshift+psycopg2://',
    echo=True,
    pool_recycle=300,
    echo_pool=True,
    creator=lambda _: psycopg2.connect(service='rockets-redshift')
)

# Load population file
pop = pd.read_csv(
    "~/Data/ACS_16_5YR_S1101/ACS_16_5YR_S1101_with_ann.csv",
    usecols=[
        'GEO.id2', 'HC01_EST_VC02', 'HC02_EST_VC02', 'HC01_EST_VC06',
        'HC01_EST_VC10'
    ],
    skiprows=[1],
    dtype={
        'GEO.id2': 'str',
        'HC01_EST_VC02': 'int',
        'HC02_EST_VC02': 'int',
        'HC01_EST_VC06': "int",
        'HC01_EST_VC10': 'int'
    })

pop.columns = [
    'zip', 'households', 'married_couples', 'families', 'households_with_kids'
]

land = pd.read_csv(
    "~/Data/2017_Gaz_zcta_national.txt",
    sep='\t',
    usecols=['GEOID', 'ALAND_SQMI'],
    dtype={
        'GEOID': 'str',
        'ALAND_SQMI': 'float'
    })

land.columns = ['zip', 'area']

df = pd.merge(pop, land)

empl = pd.read_csv(
    "~/Data/ACS_16_5YR_DP03/ACS_16_5YR_DP03_with_ann.csv",
    usecols=[
        'GEO.id2', 'HC03_VC06', 'HC03_VC12', 'HC01_VC16', 'HC03_VC17',
        'HC01_VC85', 'HC01_VC114'
    ],
    skiprows=[1],
    na_values=['-'],
    dtype={
        'GEO.id2': 'str',
        'HC03_VC06': 'float',
        'HC03_VC12': 'float',
        'HC01_VC16': 'int',
        'HC03_VC17': 'float',
        'HC01_VC85': 'str',
        'HC01_VC114': 'str'
    })

empl.columns = [
    'zip', 'employed_civil_percent', 'unemploym_rate_civil', 'n_females',
    'employed_female_percent', 'med_hh_income', 'med_fam_income'
]

df = pd.merge(df, empl)

df['households_density'] = df['households'] / df['area']
df['married_couples_density'] = df['married_couples'] / df['area']
df['families_density'] = df['families'] / df['area']
df['households_with_kids_density'] = df['households_with_kids'] / df['area']
df['females_density'] = df['n_females'] / df['area']
df.replace({'2,500-': '-999', '250,000+': '999999'}, inplace=True)

df['med_hh_income'] = df['med_hh_income'].astype('float')
df['med_fam_income'] = df['med_fam_income'].astype('float')

n_kids = pd.read_csv(
    "~/Data/ACS_16_5YR_B09001/ACS_16_5YR_B09001_with_ann.csv",
    usecols=['GEO.id2', 'HD01_VD01'],
    skiprows=[1],
    dtype={
        'GEO.id2': 'str',
        'HD01-VD01': 'int'
    })
n_kids.columns = ['zip', 'n_kids']

sch = pd.read_csv(
    "~/Data/ACS_16_5YR_S1401/ACS_16_5YR_S1401_with_ann.csv",
    usecols=[
        'GEO.id2', 'HC01_EST_VC03', 'HC02_EST_VC03', 'HC04_EST_VC03',
        'HC06_EST_VC03'
    ],
    skiprows=[1],
    na_values=['-'],
    dtype={
        'GEO.id2': 'str',
        'HC01_EST_VC03': 'int',
        'HC02_EST_VC03': 'float',
        'HC04_EST_VC03': 'float',
        'HC06_EST_VC03': 'float'
    })
sch.columns = [
    'zip', 'kids_school', 'kids_school_perc', 'kids_publ_school_perc',
    'kids_priv_school_perc'
]

sch.fillna(0, inplace=True)

df = pd.merge(df, n_kids)
df = pd.merge(df, sch)

hh = pd.read_csv(
    "~/Data/ACS_16_5YR_DP04/ACS_16_5YR_DP04_with_ann.csv",
    usecols=[
        'GEO.id2', 'HC03_VC160', 'HC03_VC161', 'HC03_VC162', 'HC03_VC163',
        'HC03_VC164', 'HC03_VC199', 'HC03_VC200', 'HC03_VC201', 'HC03_VC202',
        'HC03_VC203', 'HC03_VC204'
    ],
    skiprows=[1],
    na_values=['-'],
    dtype={
        'GEO.id2': 'str',
        'HC03_VC160': 'float',
        'HC03_VC161': 'float',
        'HC03_VC162': 'float',
        'HC03_VC163': 'float',
        'HC03_VC164': 'float',
        'HC03_VC199': 'float',
        'HC03_VC200': 'float',
        'HC03_VC201': 'float',
        'HC03_VC202': 'float',
        'HC03_VC203': 'float',
        'HC03_VC204': 'float'
    })

hh.columns = [
    'zip', 'smocapi_20', 'smocapi_25', 'smocapi_30', 'smocapi_35',
    'smocapi_high', 'grapi_15', 'grapi_20', 'grapi_25', 'grapi_30', 'grapi_35',
    'grapi_high'
]
hh.fillna(0, inplace=True)
df = pd.merge(df, hh)

df.to_csv("dim_census.csv", index=None)

# df.to_sql(
#     "dim_census",
#     redshift,
#     schema='dw',
#     if_exists='replace',
#     index=False,
#     chunksize=10000)

# df.to_sql(
#     "census",
#     stitch,
#     schema='dw',
#     if_exists='replace',
#     index=False,
#     chunksize=1000)
