import pandas as pd

targ = pd.read_csv("~/Data/Target_Stores_in_USA.csv")
targ['ZipCode'] = targ['ZipCode'].str[:5]

t = targ['ZipCode']
t = t.str[:5]
