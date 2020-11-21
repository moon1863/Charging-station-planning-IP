# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 16:00:07 2020

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
       'base_fare', 'total_fare', 'rate_per_mile', 'rate_per_minute',
       'time_fare','driving_time_to_rider','driver_id', 'car_id','make', 'model', 'year']]

#filter wednesday with maximum trips
from datetime import datetime
df_2['started_on']=pd.to_datetime(df_2['started_on'],utc=True)
df_2['start_date']=df_2['started_on'].dt.date
temp=(df_2['start_date'].value_counts().head(100))
temp.to_csv("temp.csv")
df_2['start_date']=pd.to_datetime(df_2['start_date'])
df_3_22_17=df_2[df_2['start_date']=="2017-03-22"]




df_3_22_17['start_minute']=df_3_22_17['started_on'].dt.minute
df_3_22_17['start_hr']=df_3_22_17['started_on'].dt.hour
df_3_22_17['start_cum_minute']=(df_3_22_17['start_hr']*60)+(df_3_22_17['start_minute'])

# =============================================================================
# df_4['end_on']=pd.to_datetime(df_4['completed_on'],utc=True)
# df_4['end_date']=df_4['end_on'].dt.date
# df_4['end_date']=pd.to_datetime(df_4['end_date'])
# df_4['end_hr']=df_4['end_on'].dt.hour
# =============================================================================

#set up coordinate
subset_strt_location = df_3_22_17[['start_location_lat','start_location_long']]
df_3_22_17['coord_start'] = [tuple(x) for x in subset_strt_location.to_numpy()]

subset_end_location = df_3_22_17[['end_location_lat','end_location_long']]
df_3_22_17['coord_end'] = [tuple(x) for x in subset_end_location.to_numpy()]


df_3=df_3_22_17.groupby(['driver_id'])['model'].first()
df_3=df_3.to_frame()


#assign random model for FV
df_EV_capacity_RA=pd.read_csv("EV_capacity_RA.csv")
EV_model_ar=df_EV_capacity_RA['model'].values
EV_model_li=EV_model_ar.tolist()

FV_model_srs=df_3['model'][~df_3['model'].isin(EV_model_li)].unique()
FV_model_li=FV_model_srs.tolist()


import random
for i in range (0,1131):
    model_replacer=random.choice(EV_model_li)
    df_3.iloc[i]=df_3.iloc[i].replace(to_replace=FV_model_li,value=model_replacer)
    
df_3=df_3.add_suffix('').reset_index()
df_4=df_3_22_17.merge(df_3,left_on="driver_id",right_on="driver_id") 
  
#model_y denotes randomly converted EV model  


#assign EV properties and keep remaining blank
df_5=df_4.merge(df_EV_capacity_RA,left_on="model_y",right_on="model")

#energy demand ridewise
df_5['energy_required_KWH']=df_5['distance_travelled']*.001*0.62/df_5['MKWH']
df_6=df_5.groupby(['driver_id','capacity'])['energy_required_KWH'].sum()
df_6=df_6.add_suffix('').reset_index()

#create daily data
df_6['charge_needed']=df_6.apply(lambda r:1 
                                 if (int(r.capacity) < r.energy_required_KWH) 
                                 else 0,axis=1)








qry_driver_date_charge_needed=df_6[df_6.charge_needed==1][['driver_id']]

qry_driver_date_charge_needed['driver_id']=(qry_driver_date_charge_needed
                                               ['driver_id']).astype(int)
# =============================================================================
# qry_driver_date_charge_needed['start_date']=pd.to_datetime(qry_driver_date_charge_needed
#                                                ['start_date'])
# =============================================================================
df_7=df_5.merge(qry_driver_date_charge_needed,left_on=['driver_id']\
             ,right_on=['driver_id'],how='right')\
    [['driver_id','coord_start','coord_end','start_cum_minute']]

#create_pivot_table
a=pd.pivot_table(df_7,index=['driver_id'],columns=['start_cum_minute'],\
               values=['coord_start'],aggfunc="first")

 
    
# =============================================================================
# piv_table_unique=piv_table_unique.add_suffix('').reset_index()  
# piv_table_unique.to_csv("pivot_table_EVCS.csv")  
# 
# =============================================================================

#sample data 
# =============================================================================
# sample_dic={108:[0,(30.265,-97.732),0,0,0,(30.319,-97.738),0,(30.5,-97.737)]}
# import pandas as pd
# sample_df=pd.DataFrame.from_dict(sample_dic, orient='index')
# list_coordinate={}
# list_position=[]
# for i in range (0,8):
#     if sample_df.iloc[0][i] !=0:
#         list_position.append(i)
#         list_coordinate[i]=sample_df.iloc[0][i]
#     else: sample_df.iloc[0][i]=sample_df.iloc[0][i]    
# 
# dist=[]
# for i in range(0,8):
#     try:
#         dist.append(distance(list_coordinate[list_position[i]],list_coordinate[list_position[i+1]]))          
#     except IndexError:
#         break
#     
# dist_list=[]
# for i in range (0,list_position[0]+1):    
#     dist_list.append(0)
#       
# for i in range (list_position[0]+1,list_position[1]+1):    
#     dist_list.append(dist[0]/(list_position[1]-list_position[0]))
#     
# for i in range (list_position[1]+1,list_position[2]+1):    
#     dist_list.append(dist[1]/(list_position[2]-list_position[1]))  
# =============================================================================

dri_time_location_dict
dri_df=pd.DataFrame.from_dict(dri_time_location_dict, orient='index')
list_coordinate={}
list_position=[]
for i in range (0,24):
    if dri_df.iloc[0][i] !=0:
        list_position.append(i)
        list_coordinate[i]=dri_df.iloc[0][i]
    else: dri_df.iloc[0][i]=dri_df.iloc[0][i]    


list_coordinate.astype(int)
dist=[]
for i in range(0,24):
    try:
        dist.append(distance((int(list_coordinate[list_position[i]])),
                             (int(list_coordinate[list_position[i+1]]))))          
    except IndexError:
        break
    
dist_list=[]
for i in range (0,list_position[0]+1):    
    dist_list.append(0)
      
for i in range (list_position[0]+1,list_position[1]+1):    
    dist_list.append(dist[0]/(list_position[1]-list_position[0]))
    
for i in range (list_position[1]+1,list_position[2]+1):    
    dist_list.append(dist[1]/(list_position[2]-list_position[1]))  



#usung haversine formula find distance
from math import sin, cos, sqrt, atan2, radians

# approximate radius of earth in km
def distance(loc_i,loc_f):
    R = 6373.0
    
    lat1 = radians(loc_i[0])
    lon1 = radians(loc_i[1])
    lat2 = radians(loc_f[0])
    lon2 = radians(loc_f[1])
    
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    
    return R * c