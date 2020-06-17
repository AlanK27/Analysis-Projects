import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

desired_width=500
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_columns',20)

sns.set_style('darkgrid')

UI = pd.read_csv('C:\\Users\\kai_t\\Desktop\\LA Traffic Data\\filled_traffic.csv',header=None)
df = pd.DataFrame(UI)

df.columns = df.loc[0]
df = df.loc[1:]
df = df[df.columns[1:]]

'''
i=0
for n in df.month:
    if n == 6:
        i += 1
        continue
    else:
        print(i)
        break

2594
upper 2595
'''

'''
#   get rid of incomplete June data
#   optional
df = df.loc[2595:]
'''

for n in df.columns[:2]:
    df[n] = pd.to_datetime(df[n])

for n in df.columns[2:]:
    df[n] = pd.to_numeric(df[n], errors='ignore')

#############################################################
#   creating month column based on date occured

i = []
for n in df.index:
    i.append(df['Date Occurred'].loc[n].month)
i = pd.DataFrame(i, columns=['month'], index = df.index)
df = pd.concat([df, i ], axis=1, sort = False)

race_dict = {'H':'Hispanic', 'B':'Black', 'O':'Unknown', 'W':'White', 'X':'Unknown', '-':'Unknown',
             'A':'Asian', 'K':'Asian', 'C':'Asian', 'F':'Asian', 'U':'Pacific Islander',
             'J':'Asian', 'P':'Pacific Islander', 'V':'Asian', 'Z':'Asian',
             'I':'American Indian', 'G':'Pacific Islander', 'S':'Pacific Islander', 'D':'Asian', 'L':'Asian'}
df['Victim Descent'] = df['Victim Descent'].map(race_dict)

print(df.head())
print(df.info())
print(df.describe())

###############################################################
#       Accidents per day

df1 = df[df.columns[:4]]
df2 = pd.DataFrame(df1.resample('D', on='Date Occurred').size())
df2['dates'] = df2.index
df2.columns = ['# of accidents', 'dates']

fig, ax = plt.subplots(figsize=(12, 6))
fig = sns.lineplot(x='dates', y='# of accidents', data=df2)
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))

###############################################################
#       Accidents per Month with Trend

df1 = df[df.columns[:4]]
df_month = pd.DataFrame(df1.resample('3M', on='Date Occurred').size())
df_month['dates'] = df_month.index
df_month.columns = ['# of accidents', 'dates']
df_month = df_month.iloc[2:-1]

df_year = pd.DataFrame(df1.resample('Y', on='Date Occurred').size())
df_year['dates'] = df_year.index
df_year.columns = ['# of accidents', 'dates']
df_year = df_year.iloc[2:-1]

def mov_avg(daf):
    i = []
    for n in range(len(daf)-2):
        i.append(np.sum(daf['# of accidents'].iloc[n:(n+2)])/(2))
    return i

monthly = mov_avg(df_month)
monthly = pd.DataFrame(monthly, index = df_month.index[2:], columns = ['mov avg'])

fig_month = plt.plot(df_month['dates'].iloc[2:], monthly['mov avg'], label = 'Moving Average')
fig1 = plt.plot(df_month['dates'], df_month['# of accidents'], label ='Montly Accidents')
plt.xlabel('Time')
plt.ylabel('# of accidents')
plt.legend()



############################################################################################
##          Yearly data


yearly = mov_avg(df_year)
yearly = pd.DataFrame(yearly, index = df_year.index[2:], columns = ['mov avg'])

fig_year = plt.plot(df_year['dates'].iloc[2:], yearly['mov avg'])
fig2 = plt.plot(df_year['dates'].iloc[:-1], df_year['# of accidents'].iloc[:-1])


###############################################################################################
##           Age and Race Distribution

sns.boxenplot(x='Victim Descent', y ='Victim Age',data=df,palette="Set3")
sns.boxplot(x='Victim Descent', y ='Victim Age',data=df,palette="Set3")
plt.tight_layout()

###################################################################
##            Heatmap of Race and Area

df1 = df[['Area Name','Victim Descent']]
df1['counter'] = 1
io = pd.pivot_table(data=df1, values='counter', index='Area Name', columns ='Victim Descent',aggfunc='count')

sns.heatmap(io, cmap="YlGnBu" )
plt.tight_layout()




