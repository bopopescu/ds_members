import pandas as pd
import holidays
from datetime import date
us_holidays = holidays.US()

hols = []

for date, name in sorted(holidays.US(years=2015).items()):
    print(date, name)
    hols.append(date)

for date, name in sorted(holidays.US(years=2016).items()):
    print(date, name)
    hols.append(date)

for date, name in sorted(holidays.US(years=2017).items()):
    print(date, name)
    hols.append(date)

for date, name in sorted(holidays.US(years=2018).items()):
    print(date, name)
    hols.append(date)

for date, name in sorted(holidays.US(years=2019).items()):
    print(date, name)
    hols.append(date)

for date, name in sorted(holidays.US(years=2020).items()):
    print(date, name)
    hols.append(date)

for date, name in sorted(holidays.US(years=2021).items()):
    print(date, name)
    hols.append(date)

for date, name in sorted(holidays.US(years=2022).items()):
    print(date, name)
    hols.append(date)

c = pd.bdate_range('2015-01-01', '2022-12-31', freq='C', holidays=hols)
d = pd.DataFrame(c)
d.columns=['date']

d['dow'] = d['date'].dt.day_name()

d.to_csv("business_calendar_2015_2022.csv", index=None)