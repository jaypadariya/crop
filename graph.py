#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 30 12:26:39 2019

@author: ispluser
"""

import matplotlib.pyplot as plt
import pandas as pd

condition_data = pd.read_csv(r'/home/ispluser/Dimple/Wheat_Paddy_ahm/bar_chart.csv')
condition_data.info()


# column chart for acreage:
condition_data[:].plot(x='village', y=['Acreage(2017-2018)', 'Acreage(2018-2019)'], figsize=(10,5), grid=True, kind = 'bar')

# bar chart for acreage:
condition_data[:].plot(x='village', y=['Acreage(2017-2018)', 'Acreage(2018-2019)'], figsize=(20,25), grid=True, kind = 'barh')

## conditioned data analysis:
## bar chart:
#condition_data[:].plot(x='village', y=['Condition(2017-2018)'], figsize=(20,25), grid=True, kind = 'bar')
#condition_data['Condition(2017-2018)'].value_counts().plot(kind = 'bar')
#
#condition_data.plot(kind = 'hist')

import pandas as pd
import matplotlib.pyplot as plt

df = pd.DataFrame(
    {
        "Condition(2017-2018)": ["P", "M", "G", "E"],
        "Condition(2018-2019)": ["P", "M", "G", "E"],
    }
)

categorical_features = ["Condition(2017-2018)", "Condition(2018-2019)"]
fig, ax = plt.subplots(1, len(categorical_features))
for i, categorical_feature in enumerate(df[categorical_features]):
    df[categorical_feature].value_counts().plot("bar", ax=ax[i]).set_title(categorical_feature)
fig.show()

