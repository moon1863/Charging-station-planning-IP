# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 10:35:22 2020

@author: Moon
"""

import pandas as pd

#read data
dfA=pd.read_csv("Rides_DataA.csv")
dfB=pd.read_csv("Rides_DataB.csv")


#combine data dfA and dfB
df=dfA.merge(dfB,left_on="RIDE_ID",right_on=['RIDE_ID'])
df_2=df[['RIDE_ID', 'started_on', 'completed_on', 'start_location_long', 'start_location_lat',
       'distance_travelled', 'end_location_lat', 'end_location_long','active_driver_id','rider_id',
       'base_fare', 'total_fare', 'rate_per_mile', 'rate_per_minute','time_fare','driving_time_to_rider','driver_id', 'car_id','make', 'model', 'year']]

#filter EV rows
EV_make=["b'Chevrolet'", "b'Ford'", "b'Honda'", "b'Tesla'", "b'Kia'",
         "b'Tessla'","b'Mitsubishi'","b'Nissan'","b'Toyota'"]        
EV_model=["b'Spark'","b'Bolt'","b'Focus'","b'Clarity'","b'Fit'"
          ,"b'Soul'","b'i-MiEV'","b'Leaf'","b'S'","b'X'","b'Rav-4'"]
df_3=df_2[df_2.make.isin(EV_make)]
df_3=df_3[df_3.model.isin(EV_model)]

#assign EV properties columns
df_EV_capacity_RA=pd.read_csv("EV_capacity_RA.csv")
df_4=df_3.merge(df_EV_capacity_RA,left_on="model",right_on="model")


#energy demand ridewise
df_4['energy_required_KWH']=df_4['distance_travelled']*.001*0.62/df_4['MKWH']

#daily energy demand and charging necessity identification driverwise
from datetime import datetime
df_4['started_on']=pd.to_datetime(df_4['started_on'],utc=True)
df_4['start_date']=df_4['started_on'].dt.date
df_4['start_date']=pd.to_datetime(df_4['start_date'])
df_4['start_hr']=df_4['started_on'].dt.hour

df_4['end_on']=pd.to_datetime(df_4['completed_on'],utc=True)
df_4['end_date']=df_4['end_on'].dt.date
df_4['end_date']=pd.to_datetime(df_4['end_date'])
df_4['end_hr']=df_4['end_on'].dt.hour

#set up coordinate
subset_strt_location = df_4[['start_location_lat','start_location_long']]
df_4['coordinate_start'] = [tuple(x) for x in subset_strt_location.to_numpy()]

subset_end_location = df_4[['end_location_lat','end_location_long']]
df_4['coordinate_end'] = [tuple(x) for x in subset_end_location.to_numpy()]

#create daily data
df_5=df_4.groupby(['driver_id','start_date',
                   'capcaity'])['energy_required_KWH'].sum()
df_5=df_5.add_suffix('').reset_index()

df_5['charge_needed']=df_5.apply(lambda r:1 
                                 if (int(r.capcaity) < r.energy_required_KWH) 
                                 else 0,axis=1)


qry_driver_date_charge_needed=df_5[df_5.charge_needed==1][['driver_id','start_date']]

qry_driver_date_charge_needed['driver_id']=(qry_driver_date_charge_needed
                                               ['driver_id']).astype(int)
qry_driver_date_charge_needed['start_date']=pd.to_datetime(qry_driver_date_charge_needed
                                               ['start_date'])
df6=df_4.merge(qry_driver_date_charge_needed,left_on=['driver_id','start_date']\
             ,right_on=['driver_id','start_date'],how='right')\
    [['driver_id','start_date','end_date','coordinate_start','coordinate_end',\
      'start_hr','end_hr']]

#create_pivot_table
a=pd.pivot_table(df6,index=['driver_id','start_date'],columns=['start_hr'],\
               values=['coordinate_start'],aggfunc="first")

piv_table_unique=pd.pivot_table(df6,index=['driver_id'],columns=['start_hr'],\
               values=['coordinate_start'],aggfunc="first")   
    
piv_table_unique=piv_table_unique.add_suffix('').reset_index()  
piv_table_unique.to_csv("pivot_table_EVCS.csv")  





