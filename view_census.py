import pandas as pd
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

# Load population file
pop = pd.read_csv(
    "~/Data/ACS_16_5YR_S1101/ACS_16_5YR_S1101_with_ann.csv",
    usecols=['GEO.id2', 'HC01_EST_VC02', 'HC02_EST_VC02', 'HC01_EST_VC06', 'HC01_EST_VC10'],
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
