# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 20:13:55 2020

@author: Moon
"""
#import
import pandas as pd
from random import randint
from pyomo.environ import *

#import from preprocessed data and preprocess for pyomo
table=pd.read_csv("coord_CN_df.csv")
driver_int_li=list(table['driver_id'])
driver_str_li=[str(x) for x in driver_int_li]

cent_clst=pd.read_csv("centers_df.csv")
cent_int_li=list(cent_clst['center_coord'])
cent_str_li=[str(x) for x in cent_int_li]

minutes_int_li=[517,605,890,910,958,1014,1033,1091,1118,1145,1296,1343,1368,1415,1426]
minutes_str_li=[str(x) for x in minutes_int_li]

c_m=[]
for c in cent_str_li:
    for m in minutes_str_li:
        c_m.append((c,m))

import random
from random import randint
rand_cost=[]
for i in range (0,len(c_m)):
    rand_cost.append(randint(10, 40))
dtab= dict(zip(c_m,rand_cost)) #i.e.,('(23.45,-34.24)','517minutes'):14$        

#creation of a concrete model
model=ConcreteModel()

#define sets
model.i=Set(initialize=driver_str_li,doc="driver list as str")
model.n=Set(initialize=cent_str_li,doc="center of clustered locations as str_tuple")
model.t=Set(initialize=minutes_str_li,doc="minutes when charge needed as str")

#define parameters
model.w=Param(model.n,model.t,initialize=dtab,doc='timevalue for (location,time)')
model.N=Param(initialize=5,doc="max number of charging station")

#define variable
model.x=Var(model.i,model.n,model.t,within=Binary,doc='if the driver charge\
            at that lcoationa and time')
model.a=Var(model.i,model.n,model.t,within=Binary,doc='feasible or not if i_th\
            driver want to charge at that location and time')            
model.y=Var(model.n,within=Binary,doc='if the location has vacant charging station')

#define constraint
def charge_once_rule(mdoel,i):
    return sum(model.x[i,n,t] for n in model.n for t in model.t)==1
model.charge_once=Constraint(model.i,rule=charge_once_rule,doc="assumption:only\
                             charge once in a day")

def able_to_charge_feasloca_rule(model,i,n,t):
    return model.x[i,n,t]-model.a[i,n,t]<=0
model.able_to_charge_feasloca=Constraint(model.i,model.n,model.t,rule=able_to_charge_feasloca_rule,doc="a\
                                         driver only able to charge at feasible location at time")
                                        
def able_to_charge_vacant_rule(model,i,n,t):
    return model.x[i,n,t]-model.y[n]<=0
model.able_to_charge_vacant=Constraint(model.i,model.n,model.t,rule=able_to_charge_vacant_rule,doc="a \
                                       driver only able to charge where chraging\
                                           station is vacant")
                                           
def max_station_rule(model):
    return sum(model.y[n] for n in model.n)<=model.N
model.max_station=Constraint(rule=max_station_rule,doc="maximum number of station\
                             able to build")
                             
#define objective function
def objective_rule(model):
    return sum(model.w[n,t]*model.x[i,n,t] for i in model.i for n in model.n\
               for t in model.t)
model.objective=Objective(rule=objective_rule,sense=minimize,doc="minimize timevalue")

#display of the output
def pyomo_postprocess(options=None,instance=None,results=None):
    model.y.display()
                     
from pyomo.opt import SolverFactory
opt=SolverFactory("glpk")
results=opt.solve(model)
results.write()
#print("\nDisplaying solution\n"+"-"*60)
pyomo_postprocess(None,model,results) 


                  














































# =============================================================================
# #prepare the data in matrix form
# matrix=pd.read_csv("pivot_table_EVCS.csv",header=1)
# matrix.drop(['start_hr'],axis=1,inplace=True)
# 
# #set sub values
# #worng
# I=matrix.iloc[1:,0].values
# 
# dri_time_location_dict=matrix.set_index('Unnamed: 1').transpose().to_dict(orient='dict')
# 
# I_list=I.tolist()
# T=[str(x) for x in range (24)]
# 
# N=[]
# for d in I_list:
#     for t in T:
#         N.append(dri_time_location_dict[d][t])
# 
# 
# W={}
# for d in I_list:
#     for t in T:
#         W.update({dri_time_location_dict[d][t]:randint(10,40)})
#         
# max_N=100        
# #construct model
# model=ConcreteModel()
# 
# model.Drivers=I
# model.Locations=N
# model.Time=T
# 
# model.x=Var(model.Drivers,model.Locations,model.Time,bounds=(0.0,1.0))
# model.y=Var(model.Locations,bounds=(0.0,1.0))
# 
# model.obj=Objective(expr=sum(W[n]*model.x[i,n,t] for n in model.Locations 
#                              for i in model.Drivers for t in model.Time),
#                     sense=minimize)
# 
# model.single_x=ConstraintList()
# for n in model.Locations:
#         for t in model.Time:
#             model.single_x.add(
#                 sum(model.x[n,t])==1)
#             #KeyError: "Index '('(30.214000000000002, -97.751)', '0')' is not valid for indexed component 'x'"
#             
# model.single_x_y=ConstraintList()
# for i in model.Drivers:
#     for n in model.Locations:
#         for t in model.Time:
#             model.single_x_y.add(
#                 model.x[i,n,t]<=model.y[n])
#     
# model.single_y=ConstraintList()
# for n in model.Locations:
#     model.single_y.add(
#         sum(model.y[n])<=100)
#     #TypeError: '_GeneralVarData' object is not iterable
# 
# model.single_x_a=ConstraintList()
# for i in model.Drivers:
#     for n in model.Locations:
#         for t in model.Time:
#             model.single_x_y.add(
#                 model.x[i,n,t]<=model.a[i,n,t])            
#     #a is not defined
#     
# SolverFactory('glpk').solve(model).write()
# =============================================================================
