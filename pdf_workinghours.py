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
df_2['start_hr']=(df_2['started_on'].dt.hour)+(df_2['started_on'].dt.minute)/60
 
df_2['completed_on']=pd.to_datetime(df_2['completed_on'],utc=True) #convert object to date format
df_2['complete_hr']=(df_2['completed_on'].dt.hour)+(df_2['completed_on'].dt.minute)/60

df_3=df_2.sort_values(by=['driver_id','start_date'])
df_3=df_3.reset_index()
df_3['start_date']=(df_3['start_date']).astype(str)
iteration=0
# =============================================================================
# end_time_drivers=[]
# start_time_drivers=[]
# duration=[]
# #for driver_id in df_2['driver_id']:
# =============================================================================

#find the break time which is >2.2 hrs
list1=[]
list2=[]  
list3=[]
iteration=0
b=int(0)    
for i in range(0,1494125):
        if (df_3['driver_id'][i]==df_3['driver_id'][i+1]) and (df_3['start_date'][i]==df_3['start_date'][i+1]):
            iteration=iteration+1
            print("iteration: ",iteration)
            
            print("i: ",i)
            #start_time_drivers.append(df_3['start_hr'][i])
                                      
            if (df_3['start_hr'][i+1]-df_3['complete_hr'][i]<2.2):
                b=b+0
            else: b=b+df_3['start_hr'][i+1]-df_3['complete_hr'][i]
            
            print("b: ",b)
            #end_time_drivers.append(df_3['complete_hr'][i+1])
            
            
            if b!=0:
                list1.append(df_3['driver_id'][i+1])
                list2.append(df_3['start_date'][i+1])
                list3.append(b)
                
        else: continue 
        
df_4 = pd.DataFrame(list(zip(list1, list2,list3)), 
               columns =['driver_id', 'date','breaktime'])  
df_4.drop_duplicates() 
df_4.insert(loc=3,column="breaktime_noncum",value=None)
for i in range (1,1261142):
    df_4['breaktime_noncum'][i]=(df_4['breaktime'][i]-df_4['breaktime'][i-1])
    print(i)     
df_5=df_4.groupby(['driver_id','date'])['breaktime_noncum'].sum().to_frame()
df_5=df_5.reset_index()
df_5.to_csv("df_5_breaktime.csv")

                                  
#find min of strat and max of end time to find out extreme points of schedule
working_hr=df_2.groupby(['driver_id','start_date']).agg({
    'start_hr':['min'],
    'complete_hr':['max']},
    )
working_hr=working_hr.reset_index()
#remove one of multi index row and columns
working_hr=working_hr.droplevel(level = 1,axis=1)#second index of row inmulti index removed
#working_hr=working_hr.droplevel(level = 1)#second index of column in multi index removed

working_hr=working_hr.reset_index()

#remove break time from last checkout and checkin time of the day
working_hour_up=pd.merge(working_hr,df_5,how="outer",left_on=['driver_id','start_date'],right_on=['driver_id','date']) 
working_hour_up.replace(to_replace=np.nan,value=0,inplace=True)

working_hour_up['duration']=working_hour_up['complete_hr']-working_hour_up\
    ['start_hr']-working_hour_up['breaktime_noncum']

#just count for each driver how many data available
dri_count=working_hour_up.groupby(['driver_id'])['driver_id'].count()
dri_count=dri_count.to_frame()
dri_count.rename(columns={'driver_id':'#count_available'},inplace=True)
dri_count=dri_count.reset_index()

#Take more than one row otherwise pdf is not started to plot
dri_count_10=dri_count[dri_count['#count_available']>1]
working_hr_merged=working_hour_up.merge(dri_count_10,left_on='driver_id',right_on='driver_id')

working_hr_merged=working_hr_merged[working_hr_merged['duration']>0]
working_hr_merged.to_csv("working_hr_merged.csv")
#IF YOU HAVE RESULTANT CSV YOU START FROM HERE 

import pandas as pd
working_hr_merged=pd.read_csv("working_hr_merged.csv")

#positive dureations are taken

import matplotlib.pyplot as plt

#plot for austine data's working hours
working_hr_merged.duration.plot(kind='density')
plt.xlabel('working hrs')
plt.ylabel('pdf')
plt.title("pdf for working hrs from rideaustin data")

#plot for each driver_id
fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(111)
for i in working_hr_merged['driver_id'].unique():
    working_hr_merged[working_hr_merged['driver_id']==i].duration.plot(ax=ax,kind='density') 
    plt.title('pdf of working hours of drivers') 
    plt.xlabel('Working hours')
    
#find best distribution and statistics/params
import scipy.stats as st

#f0lowwing fucntion copied from stakeoverflow
def get_best_distribution(data):
    dist_names = ["norm", "lognorm"]
    dist_results = []
    params = {}
    for dist_name in dist_names:
        dist = getattr(st, dist_name)
        param = dist.fit(data)

        params[dist_name] = param
        # Applying the Kolmogorov-Smirnov test
        D, p = st.kstest(data, dist_name, args=param)
        print("p value for "+dist_name+" = "+str(p))
        dist_results.append((dist_name, p))

    # select the best fitted distribution
    best_dist, best_p = (max(dist_results, key=lambda item: item[1]))
    # store the name of the best fit and its p value

    print("Best fitting distribution: "+str(best_dist))
    print("Best p value: "+ str(best_p))
    print("Parameters for the best fit: "+ str(params[best_dist]))

    return best_dist, best_p, params[best_dist]    
    
print(get_best_distribution(working_hr_merged['duration']))

#plot the best distributioncurve ,here, normal
#plot pdf of gaussian/normal distribution 
bin_li=[]
mu,sigma=3.37,2.93
len(set(driver_id)=4803
s = np.random.normal(mu, sigma, len(set(driver_id)))
    
# Create the bins and histogram
count, bins, ignored = plt.hist(s, 20)#20 intervals
#bins=if bins is:[1, 2, 3, 4]then the first bin is [1, 2) (including 1, but excluding 2) 
#and the second [2, 3). The last bin, however, is [3, 4], which includes 4., 
#counts are values for each interval
bin_li.append(bins)

    
    # Plot the distribution curve

fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(111)

ax.plot(bin_li[0], 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (bin_li[0] - mu)**2 / (2 * sigma**2)), linewidth=3)


plt.show()
