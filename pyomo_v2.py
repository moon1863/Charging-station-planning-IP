# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 20:13:55 2020

@author: Moon
"""
import pandas as pd
from random import randint
from pyomo.environ import *

#prepare the data in matrix form
matrix=pd.read_csv("pivot_table_EVCS.csv",header=1)
matrix.drop(['start_hr'],axis=1,inplace=True)

#set sub values
I=matrix.iloc[1:,0].values

dri_time_location_dict=matrix.set_index('Unnamed: 1').transpose().to_dict(orient='dict')

I_list=I.tolist()
T=[str(x) for x in range (24)]

N=[]
for d in I_list:
    for t in T:
        N.append(dri_time_location_dict[d][t])


W={}
for d in I_list:
    for t in T:
        W.update({dri_time_location_dict[d][t]:randint(10,40)})
        
max_N=100        
#construct model
model=ConcreteModel()

model.Drivers=I
model.Locations=N
model.Time=T

model.x=Var(model.Drivers,model.Locations,model.Time,bounds=(0.0,1.0))
model.y=Var(model.Locations,bounds=(0.0,1.0))

model.obj=Objective(expr=sum(W[n]*model.x[i,n,t] for n in model.Locations 
                             for i in model.Drivers for t in model.Time),
                    sense=minimize)

model.single_x=ConstraintList()
for n in model.Locations:
        for t in model.Time:
            model.single_x.add(
                sum(model.x[n,t])==1)
            #KeyError: "Index '('(30.214000000000002, -97.751)', '0')' is not valid for indexed component 'x'"
            
model.single_x_y=ConstraintList()
for i in model.Drivers:
    for n in model.Locations:
        for t in model.Time:
            model.single_x_y.add(
                model.x[i,n,t]<=model.y[n])
    
model.single_y=ConstraintList()
for n in model.Locations:
    model.single_y.add(
        sum(model.y[n])<=100)
    #TypeError: '_GeneralVarData' object is not iterable

model.single_x_a=ConstraintList()
for i in model.Drivers:
    for n in model.Locations:
        for t in model.Time:
            model.single_x_y.add(
                model.x[i,n,t]<=model.a[i,n,t])            
    #a is not defined
    
SolverFactory('glpk').solve(model).write()