# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 11:02:55 2020

@author: Moon
"""
import pandas as pd

dfA=pd.read_csv("Rides_DataA.csv")
dfB=pd.read_csv("Rides_DataB.csv")
#combine data dfA and dfB
df=dfA.merge(dfB,left_on="RIDE_ID",right_on='RIDE_ID')
df_2=df[['RIDE_ID', 'started_on', 'completed_on', 'start_location_long', 'start_location_lat',
        'distance_travelled', 'end_location_lat', 'end_location_long','active_driver_id','rider_id',
        'base_fare', 'total_fare', 'rate_per_mile', 'rate_per_minute',
       'time_fare','driving_time_to_rider','driver_id', 'car_id','make', 'model', 'year']]
 
#convert to date time
from datetime import datetime
df_2['started_on']=pd.to_datetime(df_2['started_on'],utc=True) #convert object to date format
df_2['start_date']=df_2['started_on'].dt.date
df_2['start_hr']=df_2['started_on'].dt.hour
 
df_2['completed_on']=pd.to_datetime(df_2['completed_on'],utc=True) #convert object to date format
df_2['complete_hr']=df_2['completed_on'].dt.hour

#find min of strat and max of end time to find out extreme points of schedule
working_hr=df_2.groupby(['driver_id','start_date']).agg({
    'start_hr':['min'],
    'complete_hr':['max']})

#remove one of multi index row and columns
working_hr=working_hr.droplevel(level = 1,axis=1)#second index of row inmulti index removed
working_hr=working_hr.droplevel(level = 1)#second index of column in multi index removed

working_hr=working_hr.reset_index()
working_hr['duration']=working_hr['complete_hr']-working_hr['start_hr']

#just count for each driver how many data available
dri_count=working_hr.groupby(['driver_id'])['driver_id'].count()
dri_count=dri_count.to_frame()
dri_count.rename(columns={'driver_id':'#count_available'},inplace=True)
dri_count=dri_count.reset_index()

#Take more than one row otherwise pdf is not started to plot
dri_count_10=dri_count[dri_count['#count_available']>1]
working_hr_merged=working_hr.merge(dri_count_10,left_on='driver_id',right_on='driver_id')

#positive dureations are taken
working_hr_merged=working_hr_merged[working_hr_merged['duration']>0]
import matplotlib.pyplot as plt

#plot
fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(111)
for i in working_hr_merged['driver_id'].unique():
    working_hr_merged[working_hr_merged['driver_id']==i].duration.plot(ax=ax,kind='density') 
    plt.title('pdf of working hours of drivers') 
    plt.xlabel('Working hours')
    


working_hr_merged..plot(x='year', y='unemployment')