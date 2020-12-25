# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 11:01:13 2020

@author: Moon
"""

import matplotlib.pyplot as plt
import numpy as np

#read each line of csv file
import csv
data =[]
with open('working_hr_merged.csv', 'r',encoding='utf-8') as f:
    reader = csv.reader(f, delimiter=',')
    for row in reader:
        data.append(row)
    data.remove(data[0])  

#find mean
sum=0    
for i in range(0,len(data)):
    sum=sum+float(data[i][7])    

mean=sum/len(data)

#find std deaviation
sum_2=0
for i in range(0,len(data)):
    sum_2=sum_2+(float(data[i][7])-mean)**2

import math
std=math.sqrt(sum_2/len(data))       

#find total number of drivers
driver_id=[]
for i in range (0,len(data)):
      driver_id.append(data[i][1]) 
len(set(driver_id))

#plot pdf of gaussian distribution 
bin_li=[]
mu=[3,4,5,6]
for rndmean in [3,4,5,6]:                    
    mu, sigma = rndmean, std
    s = np.random.normal(mu, sigma, len(set(driver_id)))
    
    # Create the bins and histogram
    count, bins, ignored = plt.hist(s, 20)#20 intervals
    #bins=if bins is:[1, 2, 3, 4]then the first bin is [1, 2) (including 1, but excluding 2) 
    #and the second [2, 3). The last bin, however, is [3, 4], which includes 4., 
    #counts are values for each interval
    bin_li.append(bins)

mu=[3,4,5,6]    
    # Plot the distribution curve

fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(111)

ax.plot(bin_li[0], 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (bin_li[0] - mu[0])**2 / (2 * sigma**2)), linewidth=3)

ax.plot(bin_li[1], 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (bin_li[1] - mu[1])**2 / (2 * sigma**2)),linewidth=3)

ax.plot(bin_li[2], 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (bin_li[2] - mu[2])**2 / (2 * sigma**2) ),linewidth=3)

ax.plot(bin_li[3], 1/(sigma * np.sqrt(2 * np.pi)) *
np.exp( - (bin_li[3] - mu[3])**2 / (2 * sigma**2) ),linewidth=3)

plt.show()
