import streamlit as st 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 
import scipy.stats as st
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')



# Loading the dataset

df = pd.read_csv('yellow_tripdata_2020-01.csv')

df.head()



# EDA

df.shape



df.dtypes



df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
df['tpep_dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'])

df['duration'] = df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']
df['duration'] = df['duration'].dt.total_seconds()/60 # coverting in minutes

df

df = df[['passenger_count', 'payment_type', 'fare_amount', 'trip_distance','duration']]

df

df.isnull().sum()



(65441/len(df))*100

We have 1% missing data



df.dropna(inplace=True)

df



df['passenger_count'] = df['passenger_count'].astype('int64')
df['payment_type'] = df['payment_type'].astype('int64')




df[df.duplicated()]



df.drop_duplicates(inplace = True)

df.shape



df['passenger_count'].value_counts(normalize = True)



df['payment_type'].value_counts(normalize = True)



df = df[df['payment_type']<3]
df = df[(df['passenger_count']>0)&(df['passenger_count']<6)]

df.shape



df['payment_type'].replace([1,2],['Online','Cash'], inplace=True)
df.head(2)



df.describe()

**getting rid of negative values**


df = df[df['fare_amount']>0]
df = df[df['trip_distance']>0]
df = df[df['duration']>0]

df.describe()



plt.boxplot(df['fare_amount'])
plt.show()



for col in ['fare_amount','trip_distance','duration']:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    IQR = q3-q1

    lower_bound = q1-1.5*IQR
    upper_bound = q3+1.5*IQR

    df = df[(df[col]>=lower_bound)&(df[col]<=upper_bound)]

df



# Visualization

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.title('Distribution of fare amount')
plt.hist(df[df['payment_type'] == 'Online']['fare_amount'], histtype='barstacked', bins = 20, edgecolor ='k', color = '#FA643F', label ='Online')
plt.hist(df[df['payment_type'] == 'Cash']['fare_amount'], histtype='barstacked', bins = 20, edgecolor ='k', color = '#FFBCAB', label ='Cash')
plt.legend()

plt.subplot(1,2,2)
plt.title('Distribution of trip distance')
plt.hist(df[df['payment_type'] == 'Online']['trip_distance'], histtype='barstacked', bins = 20, edgecolor ='k', color = '#FA643F', label ='Online')
plt.hist(df[df['payment_type'] == 'Cash']['trip_distance'], histtype='barstacked', bins = 20, edgecolor ='k', color = '#FFBCAB', label ='Cash')
plt.legend()

plt.show()

1. Data is not normally distibuted
2. Data is rightly skewed

df.groupby('payment_type').agg({'fare_amount':['mean','std'], 'trip_distance':['mean','std']})





For % we usually use piechart and sometimes donut chart

plt.title("Preference of payment type")
plt.pie(df['payment_type'].value_counts(normalize = True), labels = df['payment_type'].value_counts().index,
        startangle = 90, shadow = True, autopct = "%1.1f%%", colors = ['#FA643F','#FFBCAB'])
plt.show()



df

passenger_count = df.groupby(['payment_type','passenger_count'])[['passenger_count']].count()
passenger_count.rename(columns = {'passenger_count':'count'}, inplace=True)
passenger_count.reset_index(inplace=True)
passenger_count

passenger_count["percent"] = (passenger_count['count']/passenger_count['count'].sum())*100

passenger_count



👇 To make stacked bar chart

df_ = pd.DataFrame(columns = ['payment_type',1,2,3,4,5])
df_['payment_type'] = ['Online','Cash']
df_.iloc[0,1:] = passenger_count.iloc[5:,-1]
df_.iloc[1,1:] = passenger_count.iloc[0:5,-1]


df_





fig, ax = plt.subplots(figsize=(20,6))  

df_.plot(x='payment_type', kind='barh', stacked=True, color=['#FA643F', '#FFBCAB', '#CB8B82', '#F1F1F1', '#FD9F9F'], ax=ax)  # Assign the plot to 'ax'


for p in ax.patches:
    width = p.get_width()
    height = p.get_height()
    x,y = p.get_xy()
    ax.text(x + width/2,
            y + height/2,
            '{:.0f}%'.format(width),
            ha = 'center',
            va = 'center')
        

plt.show()  




# HYPOTHESIS TESTING

**NULL HYPOTHESIS**: There is no difference in avg fare between customers who use credit cards and customers who use cash.

**ALTERNATE HYPOTHESIS**: There is a difference in avg fare between customers who use credit cards and customers who use cash.





**Choosing Test Type**

1. Since fare amount is continous we cannot use Chi-square
2. We perform Anova when the groups are 2 or more than 2
3. Z-test : a. Large dataset  (n > 30)  b: POP_std should be known c. Data should be normally distributed
4. We use T-test when our data is small(n ≤ 30). But more importantly we use it when we don't have POP_std. It should work well with large dataset in      such cases.
  




Choosing Test Type:

1.Chi-Square Test is not suitable because fare amount is continuous (Chi-square is for categorical data).

2. ANOVA is used when comparing means across 2 or more groups (correct).
3. Z-Test:
  a. Requires large dataset (n > 30)
  b. Population standard deviation (POP_STD) must be known
  c. Data should be normally distributed
4. T-Test:
  a. Used when sample size is small (n ≤ 30)
  b. Also works for large datasets when POP_STD is unknown
  c. Assumes normality for small samples, but Central Limit Theorem helps with large samples

sm.qqplot(df['fare_amount'], line = '45')
plt.show()

data is not normal





#### **Samples**

card_sample = df[df['payment_type']=='Online']['fare_amount']
cash_sample = df[df['payment_type']=='Cash']['fare_amount']



# T-test

t_stats, p_value = st.ttest_ind(a = card_sample, b = cash_sample, equal_var = False) 
print ('T Statistic:', t_stats)
print ('p-value:', p_value)

