import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import Imputer


desired_width=500
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_columns',20)

UI = pd.read_csv('C:\\Users\\kai_t\\Desktop\\LA Traffic Data\\traffic-collision-data-from-2010-to-present.csv'
    ,header=None, low_memory=False)

df = pd.DataFrame(UI)
df.columns = df.iloc[0]
df = df.iloc[1:]

####################################################################
###     data to numeric, date time, etc

for n in df.columns:
    df[n] = pd.to_numeric(df[n], errors = 'ignore', downcast='float')

for n in df.columns[1:3]:
    df[n] = pd.to_datetime(df[n], errors='ignore')

df.drop(['DR Number', 'Location', 'LA Specific Plans', 'MO Codes']
        , axis=1, inplace=True)
df.dropna(subset=['Zip Codes','Cross Street'], inplace=True)

'''
Using Imputer to fill in null data
'''

from sklearn.preprocessing import Imputer

########################################################
#   filling numeric data (mean)

def filler_mean(daf):
    col = daf.columns
    indx = daf.index
    lenn = len(daf)
    imp = Imputer(strategy='median', verbose=0, axis=0)
    ff = pd.DataFrame()
    for n in col:
        try:
            i = type(int(daf[n].iloc[2]))
        except ValueError:
            ff = pd.concat([ff, daf[n]], sort=False, axis=1)
            continue
        except TypeError:
            ff = pd.concat([ff, daf[n]], sort=False, axis=1)
            continue

        if i == int:
             if daf[n].isnull().sum() <= lenn:
                imp = imp.fit(daf[[n]])
                imp1 = imp.transform(daf[[n]]).flatten()
                im = pd.DataFrame(imp1, index=indx, columns=[n]).round()
                ff = pd.concat([ff, im], sort=False, axis=1)
             else:
                 pass
        else:
             pass

    return ff

df = filler_mean(df)


##########################################################33
#   define class function for filling categorical data, SeriesImputer(mean)

from sklearn.base import TransformerMixin
class SeriesImputer(TransformerMixin):

    def __init__(self):
        """Impute missing values
        If the Series is of dtype Object, then impute with the most frequent object.
        If the Series is not of dtype Object, then impute with the mean
        """

    def fit(self, X, y=None):
        if X.dtype == np.dtype('O'):
            self.fill = X.value_counts().index[0]
        else:
            self.fill = X.mean()
        return self

    def transform(self, X, y=None):
       return X.fillna(self.fill)

#####################################################
#   fill categorical data with SeriesImputer

def fil_freq(daf):
    col = daf.columns
    lenn = len(daf)
    a = SeriesImputer()
    ff = pd.DataFrame()
    for n in col:
            i = type(daf[n].iloc[2])
            print(i)
            if i == str:
                if daf[n].isnull().sum() <= lenn:
                    io = a.fit_transform(daf[n])
                    ff = pd.concat([ff, io], sort=False, axis=1)
                else:
                    ff = pd.concat([ff, daf[n]], sort=False, axis=1)
            else:
                ff = pd.concat([ff, daf[n]], sort=False, axis=1)
    return ff

df = fil_freq(df)

df.to_csv(r'D:\\Data Modeling\\traffic-collision-data-from-2010-to-present.csv\\filled_traffic.csv',header=True)

#################################################################################
###                 ALL FILLED
