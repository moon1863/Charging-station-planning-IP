# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 16:00:07 2020

@author: Moon
"""

import pandas as pd

# =============================================================================
# #read data
# dfA=pd.read_csv("Rides_DataA.csv")
# dfB=pd.read_csv("Rides_DataB.csv")
# 
# #combine data dfA and dfB
# df=dfA.merge(dfB,left_on="RIDE_ID",right_on=['RIDE_ID'])
# df_2=df[['RIDE_ID', 'started_on', 'completed_on', 'start_location_long', 'start_location_lat',
#        'distance_travelled', 'end_location_lat', 'end_location_long','active_driver_id','rider_id',
#        'base_fare', 'total_fare', 'rate_per_mile', 'rate_per_minute',
#        'time_fare','driving_time_to_rider','driver_id', 'car_id','make', 'model', 'year']]
# 
# #filter wednesday with maximum trips
# from datetime import datetime
# df_2['started_on']=pd.to_datetime(df_2['started_on'],utc=True) #convert object to date format
# df_2['start_date']=df_2['started_on'].dt.date
# temp=(df_2['start_date'].value_counts().head(100))#datewise top 100 trip number generated 
# temp.to_csv("day_vs_trip.csv")# opened in excel and check the day 
# df_2['start_date']=pd.to_datetime(df_2['start_date'])
# df_4_5_17=df_2[df_2['start_date']=="2017-04-05"]#wednesday with max trip
# 
# #generate cumulative minutes from the beginning of day
# df_4_5_17['start_minute']=df_4_5_17['started_on'].dt.minute
# df_4_5_17['start_hr']=df_4_5_17['started_on'].dt.hour
# df_4_5_17['start_cum_minute']=(df_4_5_17['start_hr']*60)+(df_4_5_17['start_minute'])
# df_4_5_17.to_csv("df_4_5_17.csv")
# =============================================================================

#you can start from here. ABove commented section are the base.NEVER DELETE
df_4_5_17=pd.read_csv("df_4_5_17.csv")

#set up coordinate
subset_strt_location = df_4_5_17[['start_location_lat','start_location_long']]
df_4_5_17['coord_start'] = [tuple(x) for x in subset_strt_location.to_numpy()]

subset_end_location = df_4_5_17[['end_location_lat','end_location_long']]
df_4_5_17['coord_end'] = [tuple(x) for x in subset_end_location.to_numpy()]

df_3=df_4_5_17.groupby(['driver_id'])['model'].first()
df_3=df_3.to_frame()

#assign random model for FV
df_EV_capacity_RA=pd.read_csv("EV_capacity_RA.csv")
EV_model_ar=df_EV_capacity_RA['model'].values
EV_model_li=EV_model_ar.tolist()

FV_model_srs=df_3['model'][~df_3['model'].isin(EV_model_li)].unique()
FV_model_li=FV_model_srs.tolist()


import random
for i in range (0,len(df_3)):
    model_replacer=random.choice(EV_model_li)
    df_3.iloc[i]=df_3.iloc[i].replace(to_replace=FV_model_li,value=model_replacer)
    
df_3=df_3.add_suffix('').reset_index()
df_4=df_4_5_17.merge(df_3,left_on="driver_id",
                      right_on="driver_id")  #model_y denotes randomly converted EV model  

#assign EV properties 
df_5=df_4.merge(df_EV_capacity_RA,left_on="model_y",right_on="model")
df_5.drop('Unnamed: 0',axis=1,inplace=True)

# =============================================================================
# #will use later , after usage remove it. energy demand ridewise
# df_5['energy_required_KWH']=df_5['distance_travelled']*.001*0.62/df_5['MKWH']
# df_6=df_5.groupby(['driver_id','capacity'])['energy_required_KWH'].sum()
# df_6=df_6.add_suffix('').reset_index()
# =============================================================================

# =============================================================================
# #create daily data
# df_6['charge_needed']=df_6.apply(lambda r:1 
#                                  if (int(r.capacity) < r.energy_required_KWH) 
#                                  else 0,axis=1)
# 
# qry_driver_date_charge_needed=df_6[df_6.charge_needed==1][['driver_id']]
# 
# qry_driver_date_charge_needed['driver_id']=(qry_driver_date_charge_needed
#                                                ['driver_id']).astype(int)
# 
# df_7=df_5.merge(qry_driver_date_charge_needed,left_on=['driver_id']\
#              ,right_on=['driver_id'],how='right')\
#     [['driver_id','coord_start','coord_end','start_cum_minute']]
# =============================================================================

##find lambda i.e., distance factor
from functions import distance
df_5.insert(31,"shortest_dist","")#extra colum created with unit km

for i in range (0,len(df_5)):#shortest path distance estimated
    df_5['shortest_dist'][i]=distance(df_5['coord_start'][i],df_5['coord_end'][i])

df_5['distance_travelled']=df_5['distance_travelled']/1000#converted to km

df_6=df_5.groupby(['driver_id']).agg({'distance_travelled':sum,
                                      'shortest_dist':sum})
df_6=df_6.add_suffix('').reset_index()
df_6['lambda']=df_6['distance_travelled']/df_6['shortest_dist']

#create df with id,coord and time
df_7=df_5[['driver_id','coord_start','coord_end','start_cum_minute']]

#create_pivot_table
pivot=pd.pivot_table(df_7,index=['driver_id'],columns=['start_cum_minute'],\
               values=['coord_start'],aggfunc="first")
    
#convert pivot header
current_headers=tuple(pivot.columns.to_list())
future_headers_1=df_7['start_cum_minute'].to_list()
future_headers_2=list(set(future_headers_1))
future_headers_2.sort()
pivot.columns=future_headers_2

#pivot to dictionary
pivot=pivot.reset_index()#add driver_id as column

col_name=list(range(0,1440))
temp_pivot=pd.DataFrame(columns=col_name)

concatenated = pd.concat([pivot, temp_pivot], axis=0)

driverid=concatenated['driver_id']
concatenated.drop('driver_id',axis=1,inplace=True)
concatenated.insert(0,'driver_id',driverid)
pivot2=concatenated
pivot_dic=pivot2.set_index('driver_id').T.to_dict()#dictionary with driver key and another dic of time:coord as values

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

dri_df=pd.DataFrame.from_dict(pivot_dic, orient='index')#convert first column as index
dri_df.fillna(0,inplace=True)
dri_df_col_li=dri_df.columns.tolist()
dri_df=dri_df.reset_index()#note: if you use add suffix"" then the column name will be string
dri_df.rename(columns={'index':'driver_id'},inplace=True)

#######DISTANCE
li=dri_df_col_li#old one (errorful we will use if needed later): li=list(range(0,len(dri_df_col_li)))
dict_li=[]
for d in range(0,500):#for full data, use len(dri_df)
    list_coordinate={}
    list_position=[]
    for i in dri_df_col_li:
        if dri_df.iloc[d][i] !=0:
            list_position.append(i)
            list_coordinate[i]=dri_df.iloc[d][i]
        #else: dri_df.iloc[d][i]=dri_df.iloc[d][i]    
    
    
    dist=[]#distance only for points with coord
    for i in range(0,len(list_position)):
        try:
            dist.append(distance(list_coordinate[list_position[i]],
                                 list_coordinate[list_position[i+1]]))          
        except IndexError:
            break
        
    dist_list=[]#for all minutes
    for i in range (0,list_position[0]+1):    
        dist_list.append(0)
    
    for i in range(0,len(list_position)-1):     #find distance for all minutes
        for j in range(list_position[i]+1,(list_position[i+1])+1):    
            dist_list.append(dist[i]/((list_position[i+1])-(list_position[i])))
    
    for i in range (list_position[-1],len(dri_df_col_li)):    
        dist_list.append(0)        
    
    for i in range(0,len(dist_list)-1):         #find cumulative distance              
        dist_list[i+1]=dist_list[i]+dist_list[i+1]
    
    dict_li.append(dict(zip(li,dist_list)))

distance_dict=dict(zip(list(dri_df.driver_id),dict_li)) #combined with corresponding driver id
        
cum_distance_df=pd.DataFrame.from_dict(distance_dict, orient='index')
cum_distance_df.to_csv("cum_distance_df.csv")
cum_distance_df_2=cum_distance_df.reset_index()
cum_distance_df_2.rename(columns={'index':'driver_id'},inplace=True)


#distance converted to routing distance using lambda
df_6_500=df_6.head(500)
cum_distance_df_3=cum_distance_df_2.merge(df_6_500,left_on="driver_id",right_on="driver_id")

for i in range(0,500):
    for j in range (1,1441):
        cum_distance_df_3.iloc[i,j]=(cum_distance_df_3.iloc[i,j])*(cum_distance_df_3.iloc[i,1443]) 

cum_distance_df_4=cum_distance_df_3.drop(columns=["distance_travelled","shortest_dist","lambda"])


#generate SOC
df_5_1=df_5[['driver_id','capacity','MKWH']].drop_duplicates()
soc=cum_distance_df_4.merge(df_5_1,left_on="driver_id",right_on="driver_id")

for i in range(0,500):
    for j in range (1,1441):
        soc.iloc[i,j]=100-(((soc.iloc[i,j])*0.62)/((soc.iloc[i,1442])*(soc.iloc[i,1441])))*100 

soc.drop(columns=["MKWH","capacity"],inplace=True)
soc.to_csv("SOC.csv")
##########time interval from immediate next trip 
from functions import time_diff

s=0
p=['']*500 #for full data, replace 150 with len(dri_df)

for d in range (0,500): #for full data, replace 150 with len(dri_df)
    m=[0]*len(li)#blank list
    
    list_position_ti=[]
    for i in dri_df_col_li:
        if dri_df.iloc[d][i] !=0:
            list_position_ti.append(i)
            list_coordinate[i]=dri_df.iloc[d][i]
        else: dri_df.iloc[d][i]=dri_df.iloc[d][i]  
        
    
    for i in range(0,len(m)):
        try:
            m[list_position_ti[i]]=m[list_position_ti[i]]+list_position_ti[i]
        except IndexError:
            break
    
    
    gap=[] 
    for i in range(0,len(list_position_ti)): #find the gap value
        try:
            gap.append(time_diff(m[list_position_ti[i+1]],m[list_position_ti[i]]))
        except IndexError:
            break
    gap.append(0)
    
    j=0
    for i in range(0,len(m)):
        try:
            if m[i]!=0:
                m[i]=gap[j] 
                j=j+1
            else:
                m[i]=0 
                m[-1]=0
        except IndexError:
            break
    p[s]=m
    s=s+1        

# =============================================================================
# m=[0,0,0,0,0,0,0,0,0]#blank list
# n=[3,1,7,8]#positional number to fitt according to number
# 
# for i in range(0,9):
#     try:
#         m[n[i]]=m[n[i]]+n[i]
#     except IndexError:
#         break
# 
# m=[0,1,0,3,0,0,0,7,8]
# l=[]
# for i in m:
#     if m[i] !=0:
#         l.append(i)
#     else: m[i]=m[i]  #find postition of non zero value  
# 
# l=[1,3,7,8]
# 
# t=[] 
# for i in range(0,len(l)): #find the gap value
#     try:
#         t.append(time_diff(m[l[i+1]],m[l[i]]))          
#     except IndexError:
#         break
# 
# t=[2,4,1]
# j=0
# for i in range(0,len(m)):
#     try:
#         if m[i]!=0:
#             m[i]=t[j] 
#             j=j+1
#         else:
#             m[i]=0 
#             m[-1]=0
#     except IndexError:
#         break
# =============================================================================

# =============================================================================
# for i in range(0,len(list_position)-1):     #find distance for all minutes
#     for j in range(list_position[i]+1,list_position[i+1]+1):    
#         dist_list.append(dist[i]/(list_position[i+1]-list_position[i]))
# =============================================================================

dictionary_time=dict(zip(list(dri_df.driver_id),p))

time_intv_df=pd.DataFrame.from_dict(dictionary_time, orient='index')
time_intv_df_2=time_intv_df.reset_index()
time_intv_df_2.rename(columns={"index":"driver_id"},inplace=True)
time_intv_df_2.to_csv("time_intv_df_2.csv")
#SOC and GT comparison
soc=pd.read_csv("SOC.csv")
soc.drop(columns=["Unnamed: 0"],inplace=True)

gt=pd.read_csv("time_intv_df_2.csv")
gt.drop(columns=["Unnamed: 0"],inplace=True)

list_number=list(range(0,1440))
l=[str(x) for x in list_number]#soc's columns are string therefore, string list created

#soc.iloc[:,1:]=soc.iloc[:,1:].astype(float)
    
for i in range (1,1440):
    soc.loc[(soc.iloc[:,i]<50)&(gt.iloc[:,i]>20),l[i-1]] = "CN"
    
soc.to_csv("CN_SOC_GT.csv")    

   


# =============================================================================
# Useless
#a=list(dri_df.columns)
# a=[str(x) for x in a]
# 
# dri_df_str_col=dri_df.copy()
# dri_df_str_col.columns=a
# dri_df_str_col=dri_df_str_col.reset_index()
# 
# 
# aa2=aa[list(dri_df_str_col.columns)]#df with CN mention and same size with dri_df_str_col
# 
# 
# aa2.insert(loc=0, column='driver_id', value=list(aa['index']))
# 
# 
# #modified df particular drivers
# 
# matched_driver=[70,1418,2483,2648,2862,3241,3662,4172,4442,4656]
# 
# aa2_mdf=aa2.loc[aa2['driver_id'].isin(matched_driver)]
# dri_df_str_col_mdf=dri_df_str_col.loc[dri_df_str_col['index'].isin(matched_driver)]
# =============================================================================


# =============================================================================
# list_number=list(range(0,1440))
# pos=[str(x) for x in list_number]
# =============================================================================

soc2=pd.read_csv("CN_SOC_GT.csv") 
soc2.drop(columns=["Unnamed: 0"],inplace=True)

dri_df_mdf=dri_df.head(500)#for full data, remove head(100)
dri_df_mdf.columns=dri_df_mdf.columns.astype(str)#column converted to str

list_number_2=list(soc2.columns)
for i in range(1,1440):
    dri_df_mdf.loc[(dri_df_mdf.iloc[:,i]!=0)&(soc2.iloc[:,i]!="CN"),
                           list_number_2[i]] = 0
    
dri_df_mdf.to_csv("coord_CN.csv")

list_coordinate_CN={}
list_position_CN=[]
for d in range(0,500):#for full data, use len(dri_df)
    for i in l:
        if dri_df_mdf.iloc[d][i] !=0:
            list_position_CN.append(i)
            list_coordinate_CN[i]=dri_df_mdf.iloc[d][i]#listcoordinate with time as keys

time_coordinate_CN=pd.DataFrame.from_dict(list_coordinate_CN,orient="index")
time_coordinate_CN=time_coordinate_CN.reset_index()
time_coordinate_CN.rename(columns={"index":"driver_id",0:"latitude",1:"longitude"},inplace=True)
time_coordinate_CN.to_csv("time_coordinate_CN.csv")

# =============================================================================
# 
# list_N=[]
# a1 = [[100, 0, 100], [4, 0, 6], [100, 2, 3]]
# df_a = pd.DataFrame(a1, columns=['a', 'b', 'c'])
# 
# for i in range (0,3):
#     for j in range(0,3):
#         if df_a.iloc[i,j] !=0:
#             list_N.append(df_a.iloc[i,j])
# =============================================================================

# =============================================================================
# # extract the charging need locations
# list_N=[]
# for r in range(0,10):
#     for c in range(1,262):
#         if dri_df_str_col_mdf.iloc[r,c]!=0:
#             list_N.append(dri_df_str_col_mdf.iloc[r,c])
# 
# N_df=pd.DataFrame({'N_coord':list_N})            
# N_df.to_csv("N_df.csv")            
# 
# N_lat=[]
# for i in range(0,15):
#     N_lat.append(N_df.iloc[i,0][0])
#     
# N_long=[]
# for i in range(0,15):
#     N_long.append(N_df.iloc[i,0][1])    
# 
# parcel_id=list(range(0,15))
# =============================================================================



#k means clustering 
N_lat=[x[0] for x in list(list_coordinate_CN.values())]#find first element of tuple which is values in dict
N_long=[x[1] for x in list(list_coordinate_CN.values())]#find first element of tuple which is values in dict
parcel_id=list(range(0,len(N_lat)))


df_clst={'parcel_id':parcel_id,'N_lat':N_lat,'N_long':N_long}

df_clst=pd.DataFrame.from_dict(df_clst)
df_clst.to_csv("df_clst.csv")#combined lat,long,clustername
df_clst=pd.read_csv("df_clst.csv")
df_clst=df_clst.drop(columns=["Unnamed: 0"])


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns; sns.set()
import csv

K_clusters = range(1,20)
kmeans = [KMeans(n_clusters=i) for i in K_clusters]
Y_axis = df_clst[['N_lat']]#two bracket makes it dataframe
X_axis = df_clst[['N_long']]
score = [kmeans[i].fit(Y_axis).score(Y_axis) for i in range(len(kmeans))]
# Visualize
plt.plot(K_clusters, score)
plt.xlabel('Number of Clusters')
plt.ylabel('Score')
plt.title('Elbow Curve')
plt.show()


kmeans = KMeans(n_clusters = 10, init ='k-means++')
#kmeans.fit(df_clst[df_clst.columns[1:3]]) # Compute k-means clustering.

df_clst['cluster_label'] = kmeans.fit_predict(df_clst[df_clst.columns[1:3]])
centers = kmeans.cluster_centers_ # Coordinates of cluster centers
centers_tuple_li=[tuple(row) for row in centers]
centers_df=pd.DataFrame({'center_coord':centers_tuple_li})
centers_df.to_csv("centers_df.csv")

labels = kmeans.predict(df_clst[df_clst.columns[1:3]]) # Labels of each point
N_cluster_center_df=pd.DataFrame({'N_cl_cent':list(labels)})            
N_df.to_csv("N_df.csv") 

#plot
df_clst.plot.scatter(x = 'N_lat', y = 'N_long', c=labels, s=50, cmap='viridis')
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)

# =============================================================================
# #filter driver id, capcaity, MKWH
# df5_f=df_5[['driver_id','capacity','MKWH']]
# df5_f=df5_f.drop_duplicates()
# 
# 
# #left merge
# df_8=cum_distance_df_2.merge(df5_f,left_on='index',right_on='driver_id',how='left')
# for i in range(0,26): 
#     df_8.iloc[i,1:1441]=(df_8.iloc[i,1442]-(df_8.iloc[i,1:1441]*0.62/df_8.iloc[i,1443]))/df_8.iloc[i,1442]
#     
# df_8.to_csv("SOC.csv")    
# =============================================================================




